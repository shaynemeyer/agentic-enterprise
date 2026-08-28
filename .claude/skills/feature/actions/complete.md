# Complete Action

Every git step below requires explicit user permission before running, and the build must pass
first. Confirm before starting.

1. Verify the project's build and tests pass, using the commands in `context/ai-interaction.md`.
2. Stage the changes and commit, following the commit message convention in
   `context/ai-interaction.md`.
3. Switch to the base branch documented in `context/ai-interaction.md` and merge the feature
   branch (no push yet).
4. Ask before deleting the local feature branch, then delete it.
5. Mark the spec complete: set the `**Status:**` line under the H1 of the feature's file in
   `context/features/` (or `context/fixes/`) to
   `Complete — merged YYYY-MM-DD (<merge commit sha>)`.
6. Roll the history forward, in this order — the move must happen before the new entry is written,
   or the archive ends up out of order:
   - If `## History` in `context/current-feature.md` already holds an entry, append it to the END
     of `context/feature-history.md` (the archive, oldest to newest)
   - Then reset `context/current-feature.md`:
     - Change the H1 back to `# Current Feature`
     - Set Status back to "Not Started"
     - Clear the Goals and Notes sections (keep placeholder comments)
     - Write this feature's summary as the **only** entry under `## History`
7. Commit the reset: `chore: reset current-feature.md after completing [feature]`
8. Push the base branch to origin ONCE (a single push with all changes).
9. If the feature branch was previously pushed, delete it from origin.
