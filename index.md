---
layout: default
title: Home
---

# Bennett Financials Blog
Welcome to the Bennett Financials blog — expert insights on CFO strategy, tax planning, and financial structure.

## Recent Posts
{% for post in site.pages %}
- [{{ post.title }}]({{ post.url | relative_url }})
{% endfor %}
