---
name: map
description: Show the PageRank-weighted structural overview of this codebase
arguments:
  - name: focus
    description: File to focus on (optional — shows the dependency neighborhood)
    required: false
---

Call the `get_repo_map` MCP tool to get a structural overview of this codebase.

If the user provided a focus file argument, use it:
```
get_repo_map(focus_file: "<focus argument>", max_tokens: 2000)
```

If no focus argument, get the default overview:
```
get_repo_map(max_tokens: 2000)
```

Present the output directly — it's already formatted as a readable structural overview with PageRank scores, callers/callees, and function signatures organized by importance tier.
