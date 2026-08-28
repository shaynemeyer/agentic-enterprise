# Load Action

1. Interpret what the user supplied after "load":
   - Looks like a filename (single word, no spaces): look for `context/features/{name}.md` OR
     `context/fixes/{name}.md`
   - Multiple words: treat as an inline feature description and generate goals from it
   - Nothing supplied: stop and report that `load` requires a spec filename or feature description

2. Update `context/current-feature.md`:
   - Update the H1 heading to include the feature name (e.g. `# Current Feature: Add Navbar`)
   - Write goals as bullet points under `## Goals`
   - Write any additional notes/context under `## Notes`
   - Set Status to "Not Started"

3. Confirm the spec loaded and show the feature summary.

If the spec's `**Status:**` line already reads `Complete`, say so and confirm before proceeding —
loading a finished feature is usually a mistake.
