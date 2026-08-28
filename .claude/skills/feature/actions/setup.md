# Setup Action

This will setup the supporting files for this skill to work.

Create the following in the root of the target project:

1. `context/features` - This will be where all feature files will be stored

2. `context/ai-interaction.md` - Copy the content from `@.claude/skills/feature/templates/ai-interaction.md`

3. `context/project-overview.md` - Use `@.claude/skills/feature/templates/project-overview.md` as a base template but analyze the target project and update the file to reflect the current state of the project.

4. `context/README.md` - Copy the content from `@.claude/skills/feature/templates/README.md`

5. `context/current-feature.md` - Copy the content from `@.claude/skills/feature/templates/current-feature.md`

6. `context/testing-instructions.md` - Analyze the project's test setup (test runner, test files, conventions) and write concise instructions covering: when to write tests, what to test, and how to run them. Only meaningful tests — no unnecessary coverage.

7. `context/coding-standards.md` - Analyze the actual code in the project (do not assume a stack)
   to derive conventions already in use, covering what's applicable:
   - Language/type conventions (strictness, naming, module style)
   - Framework-specific patterns (e.g. component structure, module/DI wiring, routing conventions)
   - File/folder organization and naming
   - Error handling and validation patterns actually used
   - Linting/formatting setup already configured (reference the config, don't restate its rules)
   - Any conventions that differ between sub-projects (e.g. separate frontend/backend apps) —
     document each separately rather than forcing one shared set of rules
   Only document patterns with real evidence in the code — do not invent conventions the project
   doesn't follow yet. If the codebase is too new or thin to have established patterns in an area,
   say so rather than filling it with generic best practices.

8. Add the following section to the root `CLAUDE.md` (create the file if it does not exist):

    ## Context Files
    Read the following to get the full context of the project:

    @context/project-overview.md
    @context/coding-standards.md
    @context/ai-interaction.md
    @context/testing-instructions.md
    @context/current-feature.md

9. **Confirm** which files were created.
