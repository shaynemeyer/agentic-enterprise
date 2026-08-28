# Status Action

Read-only. Never modifies a file, never touches git state, never asks for permission.

1. Read `context/current-feature.md` — the H1 and `## Status`, plus the `## History` entry (the
   previously completed feature).
2. Count progress across specs:
   - complete: `grep -l "Status:.*Complete" context/features/*.md`
   - outstanding: `grep -L "Status:.*Complete" context/features/*.md`
   - the next feature is the lowest-numbered outstanding spec whose dependencies are all complete
3. Report git state: current branch, uncommitted change count, commits not pushed to origin.
4. Report gate state using the commands in `context/ai-interaction.md`. Run them only if they are
   fast; if a gate is slow or needs a service that is not running, say it was not checked rather
   than reporting a guess.
5. Note anything blocked or awaiting a decision — a dirty tree, an unmerged branch, a spec loaded
   but not started, gates failing.

## Output Format

Lead with the answer, then the detail. Keep it to a handful of lines — this action is used to
orient quickly, so a long report defeats it.

```
Current:   <feature name and status, or "nothing loaded">
Progress:  <n> of <total> complete — next up <nn-name>
Git:       <branch>, <clean | n uncommitted>, <n unpushed>
Gates:     <lint / tests / migrations, or "not checked (reason)">
Previous:  <feature from History>
```

Follow with a line on what to do next, and anything waiting on the user. If everything is clean
and nothing is loaded, say so plainly instead of padding the report.
