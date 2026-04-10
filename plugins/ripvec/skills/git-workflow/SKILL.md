---
name: git-workflow
description: >-
  Use when the user mentions "git pull", "git push", "push my changes",
  "pull latest", "commit and push", "merge", or any git operation in a
  repo that has a .ripvec/ directory with a repo-local search index.
  Also use when git pull fails due to dirty .ripvec/ files, or when
  the user asks about committing search index changes.
---

# Git Workflow with ripvec Repo-Local Index

When a repo has `.ripvec/cache/` committed to git, reindexing creates and
deletes object files that dirty the working tree. This skill handles git
operations correctly in that context.

## Before Pull

`pull.autoStash` is the git-native solution. When enabled, git automatically
stashes dirty files (including `.ripvec/`) before pull and pops them after.

Check if it's configured:

```bash
git config --local pull.autoStash
```

If not set, recommend:

```bash
git config --local pull.autoStash true
```

If pull fails with "dirty working tree" errors mentioning `.ripvec/`:

```bash
git stash push -u -m "ripvec-cache" -- .ripvec/
git pull
git stash pop
```

## Committing Index Changes

When pushing, check if `.ripvec/` has uncommitted changes and commit them
separately from feature work:

```bash
git add .ripvec/
git commit -m "ripvec: update search index" -- .ripvec/
```

Use a pathspec-restricted commit (`-- .ripvec/`) to avoid touching other
staged files.

## After Merge Conflicts

- **manifest.json conflicts:** `.ripvec/.gitattributes` sets `merge=ours`,
  so manifest conflicts auto-resolve. The next reindex reconciles from the
  filesystem.
- **Object conflicts:** Delete `.ripvec/cache/objects/` and reindex. Objects
  are content-addressed and will be regenerated.

## Key Points

- Never manually edit files in `.ripvec/cache/`
- Object files are binary (marked `binary` in `.gitattributes`)
- `config.toml` and `.gitattributes` in `.ripvec/` are managed by ripvec
- The `auto_stash` field in `config.toml` tracks whether the user was asked
  about `pull.autoStash`
