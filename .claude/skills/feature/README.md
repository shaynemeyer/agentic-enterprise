# /feature Skill

Spec-driven feature development workflow. Each feature gets a written spec before code is written, keeping Claude focused on defined goals and making reviews predictable.

---

## Setup

**Required the first time this skill is installed in a project.** Run once to create the `context/` directory structure and seed Claude's context files. No other commands will work until this is done.

```
/feature setup
```

Creates:

| File | Purpose |
|------|---------|
| `context/project-overview.md` | Auto-generated summary of the project's stack and structure |
| `context/coding-standards.md` | Code conventions derived from analyzing the actual codebase |
| `context/ai-interaction.md` | Rules for how Claude should behave in this project |
| `context/testing-instructions.md` | When and how to write tests (derived from your test setup) |
| `context/current-feature.md` | Tracks the active feature |
| `context/README.md` | Explains the context directory to teammates |
| `context/features/` | Directory where feature spec files are stored |

Also adds a `## Context Files` block to your root `CLAUDE.md` so Claude loads this context automatically on every conversation.

---

## Workflow

### Check status _(optional)_

```
/feature status
```

Read-only. Reports the active feature and its status, progress across all specs, git state
(branch, uncommitted/unpushed changes), gate state (lint/tests/migrations), and what to do next.

### 1. Create a spec _(optional)_

```
/feature <name>
/feature <name> <description>
```

Scaffolds a spec file at `context/features/<name>.md`. If a description is provided, Claude pre-populates the spec sections from it. Without a description, sections are left as empty placeholders for you to fill in.

The spec is **not** activated — run `/feature load <name>` when you're ready to start work.

### 2. Load a spec _(required)_

```
/feature load <name> [description]
```

Reads the spec and populates `context/current-feature.md` with goals and notes. Sets status to `Not Started`.

- If the spec file exists, it is loaded directly.
- If it doesn't exist, a description is required to create it first.
- Omit `<name>` to derive one automatically from the description.

### 3. Start implementation _(required)_

```
/feature start
```

Sets status to `In Progress`, creates and checks out a feature branch, then implements the goals from `current-feature.md` one by one.

To specify the branch name:

```
/feature start my-branch-name
```

### 4. Review progress _(optional)_

```
/feature review
```

Audits all changes against the goals in `current-feature.md`. Reports on goals met, missing, code quality issues, scope creep, and test coverage. Returns a verdict: ready to complete or needs changes.

### 5. Explain changes _(optional)_

```
/feature explain
```

Lists every file created or modified, with a 1–2 sentence explanation of what changed and why. Ends with a summary of how the pieces fit together — useful for writing a PR description or onboarding a reviewer.

### 6. Write tests _(optional)_

```
/feature test
```

Reviews what was implemented and writes tests following the project's testing conventions from `context/testing-instructions.md`.

### 7. Complete the feature _(required)_

```
/feature complete
```

Stages and commits all changes, resets `current-feature.md` for the next feature, and appends a summary to the history log. Prompts you to open a Merge Request.

---

## Command Reference

`<required>` `[optional]`

| Command | What it does |
|---------|-------------|
| `/feature setup` | **Run first.** Initialize context files in the project |
| `/feature status` | _(optional)_ Report active feature, progress, git and gate state |
| `/feature <name> [description]` | _(optional)_ Create a spec (optionally pre-populated from description) |
| `/feature load <name> [description]` | **Required.** Load a spec, creating it from description if it doesn't exist. Omit `<name>` to derive one from the description. |
| `/feature start [branch]` | **Required.** Branch and implement the active feature |
| `/feature review` | _(optional)_ Audit changes against goals |
| `/feature explain` | _(optional)_ Summarize files changed and how they connect |
| `/feature test` | _(optional)_ Write tests for the implemented feature |
| `/feature complete` | **Required.** Commit, reset, and prompt for MR |
| `/feature help` | _(optional)_ Show all commands |
