# Test Action

1. Read `context/current-feature.md` to understand what was implemented.
2. Read `context/ai-interaction.md` for the project's test and build commands, and
   `context/coding-standards.md` for testing conventions. Use those — do not assume a framework
   or command.
3. Identify what this feature added or modified that carries real logic: request handlers,
   business rules, data access, pure utility functions.
4. Check whether tests already exist for those.
5. For untested code with testable logic, write tests that follow the existing test layout and
   fixtures already in the repo:
   - Cover the happy path and the error cases that matter (auth, not-found, validation).
   - Test pure logic and boundaries. Skip UI components, thin framework wrappers, and
     generated code.
   - Do not write tests just to write them. Use your best judgement.
6. Run the project's test and build commands as documented in `context/ai-interaction.md` and
   confirm they pass.
7. Report what was tested and any gaps in coverage.
