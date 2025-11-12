---
layout: default
title: Home
---

# Bennett Financials Blog
Welcome to the Bennett Financials blog — expert insights on CFO strategy, tax planning, and financial structure.

## Recent Articles
{% assign pages_list = site.pages | sort: 'title' %}
{% for page in pages_list %}
  {% if page.title and page.name != "index.md" and page.name != "_config.yml" %}
- [{{ page.title }}]({{ page.url | relative_url }})
  {% endif %}
{% endfor %}

