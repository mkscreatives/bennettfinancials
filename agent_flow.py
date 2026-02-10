#!/usr/bin/env python3
"""Generate a script using a multi-agent prompt flow."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
from textwrap import dedent
from time import sleep
from typing import Any
import urllib.request


@dataclass(frozen=True)
class Agent:
    name: str
    role: str
    prompt_template: str


TEMPLATE_LIBRARY = {
    "generic": dedent(
        """
        #!/usr/bin/env bash
        # README: Generic production script template.
        set -euo pipefail
        readonly NOW="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

        log() { echo "[$1] ${NOW} ${*:2}"; }
        die() { log ERROR "$*"; exit 1; }
        require_cmd() { for cmd in "$@"; do command -v "$cmd" >/dev/null 2>&1 || die "Missing $cmd"; done; }

        usage() {
          cat <<'USAGE'
        Usage: script.sh --input <path> [--output <path>] [--dry-run]
        USAGE
        }

        main() {
          local input="" output="" dry_run=false
          while [[ $# -gt 0 ]]; do
            case "$1" in
              --input) input="$2"; shift 2 ;;
              --output) output="$2"; shift 2 ;;
              --dry-run) dry_run=true; shift ;;
              -h|--help) usage; exit 0 ;;
              *) die "Unknown argument: $1" ;;
            esac
          done
          [[ -n "$input" ]] || die "--input required"
          require_cmd bash
          log INFO "Starting"
          [[ "$dry_run" == true ]] && log INFO "Dry run enabled"
          # TODO: Implement steps.
          cat <<EOF
        Task summary:
        - input: $input
        - output: ${output:-<not set>}
        - dry_run: $dry_run
        - completed_at: $NOW
        EOF
          log INFO "Done"
        }

        main "$@"
        """
    ).strip(),
    "backup": dedent(
        """
        #!/usr/bin/env bash
        # README: Backup-oriented script template.
        set -euo pipefail
        readonly NOW="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
        log() { echo "[$1] ${NOW} ${*:2}"; }
        die() { log ERROR "$*"; exit 1; }
        require_cmd() { for cmd in "$@"; do command -v "$cmd" >/dev/null 2>&1 || die "Missing $cmd"; done; }
        usage() { echo "Usage: backup.sh --input <dir> --output <archive> [--dry-run]"; }
        main() { :; }
        main "$@"
        """
    ).strip(),
    "etl": dedent(
        """
        #!/usr/bin/env bash
        # README: ETL/transform script template.
        set -euo pipefail
        readonly NOW="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
        log() { echo "[$1] ${NOW} ${*:2}"; }
        die() { log ERROR "$*"; exit 1; }
        require_cmd() { for cmd in "$@"; do command -v "$cmd" >/dev/null 2>&1 || die "Missing $cmd"; done; }
        usage() { echo "Usage: etl.sh --input <file> --output <file> [--dry-run]"; }
        main() { :; }
        main "$@"
        """
    ).strip(),
}

WRITER_CHECKLIST = dedent(
    """
    Script quality checklist (mandatory):
    - Keep set -euo pipefail.
    - Provide usage() and --help.
    - Include --dry-run and prevent side effects in dry-run mode.
    - Validate required inputs; use non-zero exits on failures.
    - Add require_cmd checks for dependencies.
    - Use UTC timestamp logging.
    - Avoid destructive defaults.
    - Emit final manifest summary.
    - Include short README comments at top.
    """
).strip()


AGENTS = [
    Agent("planner", "Decompose the task", "You are Planner. Return JSON with keys: steps, assumptions. Task: {task}"),
    Agent("requirements", "Extract requirements", "You are Requirements. Return JSON with keys: functional, constraints. Task: {task}"),
    Agent("outline", "Create outline", "You are Outline. Return JSON with keys: sections, flags, checks. Task: {task}"),
    Agent(
        "script_writer",
        "Write script",
        "You are ScriptWriter. Produce runnable Bash script only. Use selected template + checklist. Task: {task}",
    ),
    Agent(
        "reviewer",
        "Improve script",
        "You are Reviewer. Return improved final Bash script only. Enforce checklist strictly. Task: {task}",
    ),
    Agent(
        "validator",
        "Score quality",
        "You are Validator. Return strict JSON: {\"score\": int, \"issues\": [str], \"approved\": bool}. Task: {task}",
    ),
]

AGENT_INDEX = {a.name: a for a in AGENTS}


def select_template(task: str) -> tuple[str, str]:
    lowered = task.lower()
    if any(word in lowered for word in ["backup", "archive", "compress", "log"]):
        return "backup", TEMPLATE_LIBRARY["backup"]
    if any(word in lowered for word in ["etl", "transform", "csv", "export", "import"]):
        return "etl", TEMPLATE_LIBRARY["etl"]
    return "generic", TEMPLATE_LIBRARY["generic"]


def load_cache(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
    except Exception:
        return {}
    return {}


def save_cache(path: Path, cache: dict[str, str]) -> None:
    path.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def hash_key(*parts: str) -> str:
    return hashlib.sha256("||".join(parts).encode("utf-8")).hexdigest()


def init_memory_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
              task TEXT NOT NULL,
              title TEXT NOT NULL,
              summary TEXT NOT NULL,
              content_hash TEXT NOT NULL UNIQUE
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def normalize_text(text: str) -> str:
    lowered = text.lower()
    lowered = re.sub(r"[^a-z0-9\s]", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def tokenize(text: str) -> set[str]:
    return {token for token in normalize_text(text).split() if len(token) > 2}


def jaccard_similarity(left: str, right: str) -> float:
    lset = tokenize(left)
    rset = tokenize(right)
    if not lset or not rset:
        return 0.0
    return len(lset & rset) / len(lset | rset)


def extract_title_and_summary(content: str) -> tuple[str, str]:
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    title = ""
    for line in lines:
        if line.startswith("#"):
            title = line.lstrip("# ").strip()
            break
    if not title and lines:
        title = lines[0][:120]
    summary = " ".join(lines[:6])[:300] if lines else ""
    return title or "untitled", summary


def fetch_recent_memories(path: Path, limit: int) -> list[dict[str, str]]:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT task, title, summary, created_at FROM memories ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def is_too_similar(path: Path, content: str, threshold: float) -> tuple[bool, list[str]]:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute("SELECT title, summary FROM memories ORDER BY id DESC LIMIT 200").fetchall()
    finally:
        conn.close()

    title, summary = extract_title_and_summary(content)
    new_blob = f"{title}\n{summary}"
    matches: list[str] = []
    for row in rows:
        old_blob = f"{row['title']}\n{row['summary']}"
        if jaccard_similarity(new_blob, old_blob) >= threshold:
            matches.append(str(row["title"]))
    return (len(matches) > 0, matches[:5])


def store_memory(path: Path, task: str, content: str) -> None:
    title, summary = extract_title_and_summary(content)
    fingerprint = hashlib.sha256(normalize_text(content).encode("utf-8")).hexdigest()
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            "INSERT OR IGNORE INTO memories (task, title, summary, content_hash) VALUES (?, ?, ?, ?)",
            (task, title, summary, fingerprint),
        )
        conn.commit()
    finally:
        conn.close()


def build_memory_context(path: Path, limit: int) -> str:
    items = fetch_recent_memories(path, limit)
    if not items:
        return ""
    lines = ["Recent generated items (avoid duplicates):"]
    for item in items:
        lines.append(f"- {item['title']} :: {item['summary']}")
    return "\n".join(lines)


def call_openai(
    messages: list[dict[str, str]],
    model: str,
    base_url: str,
    api_key: str,
    stream: bool,
    max_retries: int,
    timeout: int,
) -> tuple[str, dict[str, Any] | None]:
    payload = {"model": model, "messages": messages, "temperature": 0.2, "stream": stream}
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    attempt = 0
    while True:
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                if stream:
                    chunks: list[str] = []
                    for raw in response:
                        line = raw.decode("utf-8").strip()
                        if not line.startswith("data:"):
                            continue
                        data = line.removeprefix("data:").strip()
                        if data == "[DONE]":
                            break
                        evt = json.loads(data)
                        delta = evt.get("choices", [{}])[0].get("delta", {})
                        if delta.get("content"):
                            chunks.append(delta["content"])
                    return "".join(chunks).strip(), None
                data = json.loads(response.read().decode("utf-8"))
                return data["choices"][0]["message"]["content"].strip(), data.get("usage")
        except Exception as exc:  # noqa: BLE001
            attempt += 1
            if attempt > max_retries:
                raise SystemExit(f"OpenAI API request failed after {max_retries} retries: {exc}")
            sleep(2**attempt)


def validate_json(agent: str, content: str, strict: bool) -> None:
    if not strict:
        return
    try:
        obj = json.loads(content)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{agent} returned invalid JSON: {exc}")
    if agent == "validator":
        required = {"score", "issues", "approved"}
        if not isinstance(obj, dict) or not required.issubset(obj):
            raise SystemExit("validator JSON missing required keys")


def run_agent_prompt(
    agent: Agent,
    task: str,
    model: str,
    base_url: str,
    api_key: str,
    context: str,
    data_context: str,
    selected_template: str,
    json_mode: bool,
    strict_json_all: bool,
    stream: bool,
    max_retries: int,
    timeout: int,
    draft: str | None,
    cache: dict[str, str],
    use_cache: bool,
) -> str:
    system_prompt = agent.prompt_template.format(task=task)
    force_json = strict_json_all or (json_mode and agent.name in {"planner", "requirements", "outline", "validator"})
    if force_json:
        system_prompt = f"{system_prompt}\n\nRespond ONLY as valid JSON."

    user_prompt = dedent(
        f"""
        Task: {task}

        Context:
        {context}

        Data context:
        {data_context}

        Selected template key: {selected_template}

        Script template:
        {TEMPLATE_LIBRARY[selected_template]}

        Checklist:
        {WRITER_CHECKLIST}
        """
    ).strip()
    if draft:
        user_prompt += f"\n\nDraft:\n{draft}"

    cache_key = hash_key(agent.name, model, system_prompt, user_prompt)
    if use_cache and cache_key in cache:
        return cache[cache_key]

    content, usage = call_openai(
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        model=model,
        base_url=base_url,
        api_key=api_key,
        stream=stream,
        max_retries=max_retries,
        timeout=timeout,
    )
    validate_json(agent.name, content, force_json)
    if usage:
        print(f"[usage:{agent.name}] {json.dumps(usage, separators=(',', ':'))}")
    cache[cache_key] = content
    return content


def build_output(
    task: str,
    model: str,
    base_url: str,
    api_key: str,
    data_context: str,
    json_mode: bool,
    strict_json_all: bool,
    stream: bool,
    max_retries: int,
    timeout: int,
    parallel: bool,
    cache: dict[str, str],
    use_cache: bool,
) -> dict[str, str]:
    outputs: dict[str, str] = {}
    context_parts: list[str] = []
    template_key, _template = select_template(task)

    if parallel:
        first_agents = [AGENT_INDEX["planner"], AGENT_INDEX["requirements"]]
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {
                executor.submit(
                    run_agent_prompt,
                    agent,
                    task,
                    model,
                    base_url,
                    api_key,
                    "",
                    data_context,
                    template_key,
                    json_mode,
                    strict_json_all,
                    stream,
                    max_retries,
                    timeout,
                    None,
                    cache,
                    use_cache,
                ): agent
                for agent in first_agents
            }
            for future in futures:
                agent = futures[future]
                outputs[agent.name] = future.result()
                outputs[f"{agent.name}_prompt"] = agent.prompt_template.format(task=task)
        context_parts.extend([f"planner output:\n{outputs['planner']}", f"requirements output:\n{outputs['requirements']}"])
        remaining = [AGENT_INDEX["outline"], AGENT_INDEX["script_writer"], AGENT_INDEX["reviewer"], AGENT_INDEX["validator"]]
    else:
        remaining = AGENTS

    for agent in remaining:
        prompt = agent.prompt_template.format(task=task)
        context = "\n\n".join(context_parts)
        if agent.name == "script_writer":
            draft = run_agent_prompt(
                agent,
                task,
                model,
                base_url,
                api_key,
                context + "\n\nCreate draft first.",
                data_context,
                template_key,
                json_mode,
                strict_json_all,
                stream,
                max_retries,
                timeout,
                None,
                cache,
                use_cache,
            )
            outputs["script_writer_draft"] = draft
            outputs[agent.name] = run_agent_prompt(
                agent,
                task,
                model,
                base_url,
                api_key,
                context,
                data_context,
                template_key,
                json_mode,
                strict_json_all,
                stream,
                max_retries,
                timeout,
                draft,
                cache,
                use_cache,
            )
        elif agent.name == "validator":
            validator_context = context + f"\n\nFinal script:\n{outputs.get('reviewer', '')}"
            outputs[agent.name] = run_agent_prompt(
                agent,
                task,
                model,
                base_url,
                api_key,
                validator_context,
                data_context,
                template_key,
                True,
                True,
                stream,
                max_retries,
                timeout,
                None,
                cache,
                use_cache,
            )
        else:
            outputs[agent.name] = run_agent_prompt(
                agent,
                task,
                model,
                base_url,
                api_key,
                context,
                data_context,
                template_key,
                json_mode,
                strict_json_all,
                stream,
                max_retries,
                timeout,
                None,
                cache,
                use_cache,
            )
        outputs[f"{agent.name}_prompt"] = prompt
        context_parts.append(f"{agent.name} output:\n{outputs[agent.name]}")

    return outputs


def persist_outputs(output_dir: Path, outputs: dict[str, str], json_mode: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for agent in AGENTS:
        suffix = "json" if json_mode or agent.name == "validator" else "txt"
        output_dir.joinpath(f"{agent.name}.{suffix}").write_text(outputs[agent.name] + "\n", encoding="utf-8")
        output_dir.joinpath(f"{agent.name}_prompt.txt").write_text(outputs[f"{agent.name}_prompt"] + "\n", encoding="utf-8")
    if "script_writer_draft" in outputs:
        output_dir.joinpath("script_writer_draft.txt").write_text(outputs["script_writer_draft"] + "\n", encoding="utf-8")


def load_prompt_overrides(path: Path | None) -> dict[str, str]:
    if not path:
        return {}
    if not path.exists():
        raise SystemExit(f"Prompt config not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit("Prompt config must be a JSON object keyed by agent name.")
    return {str(k): str(v) for k, v in data.items()}


def apply_prompt_overrides(overrides: dict[str, str]) -> list[Agent]:
    if not overrides:
        return AGENTS
    return [Agent(name=a.name, role=a.role, prompt_template=overrides.get(a.name, a.prompt_template)) for a in AGENTS]


def load_data_context(path: Path | None) -> str:
    if not path:
        return ""
    if not path.exists():
        raise SystemExit(f"Data context file not found: {path}")
    return path.read_text(encoding="utf-8").strip()


def load_preset(path: Path | None) -> dict[str, Any]:
    if not path:
        return {}
    if not path.exists():
        raise SystemExit(f"Preset config not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit("Preset config must be a JSON object.")
    return data


def run_self_test(model: str, base_url: str, api_key: str, timeout: int) -> None:
    content, _usage = call_openai(
        [{"role": "system", "content": "Reply with OK only."}, {"role": "user", "content": "OK"}],
        model=model,
        base_url=base_url,
        api_key=api_key,
        stream=False,
        max_retries=1,
        timeout=timeout,
    )
    if content.strip().upper() != "OK":
        raise SystemExit(f"Self-test failed: unexpected response: {content!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate scripts with a high-quality multi-agent flow.")
    parser.add_argument("--task", help="Task description for the agents.")
    parser.add_argument("--output", default="generated_task.sh", help="Output path for generated script.")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"), help="Model name.")
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"), help="API base URL.")
    parser.add_argument("--output-dir", default="agent_outputs", help="Output artifact directory.")
    parser.add_argument("--data-context", type=Path, default=Path("dummy_db.json"), help="Optional data context file.")
    parser.add_argument("--no-data-context", action="store_true", help="Disable data context.")
    parser.add_argument("--prompt-config", type=Path, help="Optional prompt override JSON.")
    parser.add_argument("--preset", type=Path, help="Optional preset JSON for defaults.")
    parser.add_argument("--json-mode", action="store_true", help="Require JSON for planning agents.")
    parser.add_argument("--strict-json-all", action="store_true", help="Require JSON for all agents where applicable.")
    parser.add_argument("--stream", action="store_true", help="Use streaming mode.")
    parser.add_argument("--max-retries", type=int, default=3, help="API retry count.")
    parser.add_argument("--timeout", type=int, default=60, help="Request timeout seconds.")
    parser.add_argument("--print-only", action="store_true", help="Print final script only.")
    parser.add_argument("--demo", action="store_true", help="Run demo task.")
    parser.add_argument("--parallel", action="store_true", help="Parallelize planner+requirements.")
    parser.add_argument("--self-test", action="store_true", help="Run API connectivity check.")
    parser.add_argument("--interactive", action="store_true", help="Prompt for task interactively.")
    parser.add_argument("--cache-file", type=Path, default=Path(".agent_cache.json"), help="Response cache file.")
    parser.add_argument("--no-cache", action="store_true", help="Disable cache.")
    parser.add_argument("--memory-db", type=Path, default=Path("agent_memory.sqlite"), help="SQLite memory DB path.")
    parser.add_argument("--memory-limit", type=int, default=25, help="How many recent items to inject as memory context.")
    parser.add_argument("--novelty-threshold", type=float, default=0.55, help="Jaccard similarity threshold for duplicate detection.")
    parser.add_argument("--max-regenerations", type=int, default=3, help="Regeneration attempts if memory flags duplicate content.")
    parser.add_argument("--no-memory", action="store_true", help="Disable persistent memory checks and storage.")
    args = parser.parse_args()

    preset = load_preset(args.preset)
    task = args.task or preset.get("task")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is required to run the agent flow.")

    if args.self_test:
        run_self_test(args.model, args.base_url, api_key, args.timeout)
        print("Self-test passed.")
        return

    if args.interactive and not task:
        task = input("Describe the task for the agents: ").strip()
    if args.demo:
        task = "Create a script that backs up logs and compresses archives."
    if not task:
        raise SystemExit("Provide --task or use --demo.")

    overrides = load_prompt_overrides(args.prompt_config)
    global AGENTS
    AGENTS = apply_prompt_overrides(overrides)
    global AGENT_INDEX
    AGENT_INDEX = {a.name: a for a in AGENTS}

    data_context = "" if args.no_data_context else load_data_context(args.data_context)
    cache = {} if args.no_cache else load_cache(args.cache_file)

    memory_context = ""
    if not args.no_memory:
        init_memory_db(args.memory_db)
        memory_context = build_memory_context(args.memory_db, args.memory_limit)

    combined_data_context = "\n\n".join(part for part in [data_context, memory_context] if part)

    attempt = 0
    similarity_matches: list[str] = []
    while True:
        attempt += 1
        results = build_output(
            task=task,
            model=args.model,
            base_url=args.base_url,
            api_key=api_key,
            data_context=combined_data_context,
            json_mode=args.json_mode,
            strict_json_all=args.strict_json_all,
            stream=args.stream,
            max_retries=args.max_retries,
            timeout=args.timeout,
            parallel=args.parallel,
            cache=cache,
            use_cache=not args.no_cache,
        )
        if args.no_memory:
            break
        too_similar, similarity_matches = is_too_similar(args.memory_db, results["reviewer"], args.novelty_threshold)
        if (not too_similar) or attempt >= args.max_regenerations:
            break
        combined_data_context = "\n\n".join(
            part for part in [combined_data_context, f"Avoid these similar past titles: {', '.join(similarity_matches)}"] if part
        )

    validator = json.loads(results["validator"])
    if not validator.get("approved", False):
        print("Validator did not approve output. Issues:")
        for issue in validator.get("issues", []):
            print(f"- {issue}")
    if similarity_matches:
        print("Memory similarity matches found:")
        for title in similarity_matches:
            print(f"- {title}")

    output_path = Path(args.output)
    persist_outputs(Path(args.output_dir), results, json_mode=args.json_mode)
    if not args.no_cache:
        save_cache(args.cache_file, cache)

    if args.print_only:
        print(results["reviewer"])
    else:
        output_path.write_text(results["reviewer"] + "\n", encoding="utf-8")
        output_path.chmod(0o755)

    if not args.no_memory:
        store_memory(args.memory_db, task, results["reviewer"])

    print("Agent flow complete. Outputs:")
    for agent in AGENTS:
        print(f"\n[{agent.name.upper()}]")
        print(results[agent.name])
    print(f"\nScript written to: {output_path}")


if __name__ == "__main__":
    main()
