---
name: feature
description: Manage the current feature workflow — report status, load a spec, start implementation, review, test, explain, or complete a feature. Use when the user asks for feature status or progress, or to load/start/review/test/explain/complete a feature or fix, or refers to the feature lifecycle tracked in context/current-feature.md.
---

# Feature Workflow

Manages the full lifecycle of a feature from spec to merge. State is tracked in
`context/current-feature.md` — read that file at the start of every action.

## Working File

`context/current-feature.md` has these sections:

- `# Current Feature` — H1 heading, includes the feature name when a feature is active
- `## Status` — Not Started | In Progress | Complete. `complete` resets this to "Not Started" and
  records the feature under History, so "Complete" is only ever a transient value
- `## Goals` — bullet points of what success looks like
- `## Notes` — additional context, constraints, or details from the spec
- `## History` — the previously completed feature: the one finished immediately before whatever is
  currently loaded. Exactly one entry. `complete` moves any prior entry to
  `context/feature-history.md` (the archive, oldest to newest) before writing the new one, so this
  section never accumulates

## Spec Files

Feature specs live in `context/features/{nn-name}.md` (fixes in `context/fixes/`), numbered in
dependency order. Each carries a status line directly under its H1:

```markdown
# Bays CRUD

**Status:** Not Started
```

Only two values are used: `Not Started`, and `Complete — merged YYYY-MM-DD (<sha>)` which
`complete` writes. There is deliberately no "In Progress" here — only one feature is ever active,
and `context/current-feature.md` tracks that, so a second mutable copy would only drift.

Outstanding work is therefore `grep -L "Status:.*Complete" context/features/*.md`. Completed specs
stay in place rather than moving to an archive folder: they are still referenced constantly
(every CRUD feature builds on `02-domain-schema`, and 15 features extend `04-seed-harness`), and
each spec's `Depends on:` line names siblings by filename.

## Actions

The user names one action. Determine which from their request.

| Action     | Description                                               |
| ---------- | --------------------------------------------------------- |
| `status`   | Report where things stand — read-only, changes nothing    |
| `load`     | Load a feature spec or inline description                 |
| `start`    | Begin implementation, create branch                       |
| `review`   | Check goals met, code quality                             |
| `test`     | Write or verify tests for the feature's new logic          |
| `explain`  | Document what changed and why                             |
| `complete` | Commit, push, merge, reset                                |

Before executing, read the matching instruction file for the full steps:
`.claude/skills/feature/actions/{action}.md`

If the user did not specify an action, list the options above and ask which they want.

## Constraints

- Read `context/ai-interaction.md` at the start of any action that runs commands or touches git.
  It is the source of truth for workflow, branch naming, commit format, and dev commands.
  `context/coding-standards.md` is the source of truth for code conventions. Do not hardcode
  stack-specific commands or assumptions here.
- Never commit, merge, push, or delete a branch without explicit permission, and never before the
  build passes. This overrides any looser wording in the action files.
