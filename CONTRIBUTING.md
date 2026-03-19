# Contributing to Q²

Thank you for your interest in contributing! Please read the guidelines below before
opening issues or submitting pull requests.

---

## Issues first — always

**Every pull request must reference at least one open issue.**

The workflow is:

1. **Find or create an issue** that describes the bug, feature, or improvement.
2. **Discuss the approach** in the issue comments before writing code — this avoids
   duplicate effort and ensures alignment with the project direction.
3. **Open a PR** that references the issue using one of the keywords below.

> If you open a PR without referencing an open issue, an automated check will fail
> and the PR cannot be merged until the link is added.

### Referencing issues in your PR

Add a line to the PR description using one of the **closing keywords** (automatically
closes the issue when the PR is merged):

```
Closes #<issue-number>
Fixes #<issue-number>
Resolves #<issue-number>
```

Or use a **non-closing reference** when the issue should remain open after merge
(e.g. the PR only partially addresses it):

```
Addresses #<issue-number>
References #<issue-number>
Related to #<issue-number>
Part of #<issue-number>
See #<issue-number>
```

---

## Development workflow

```bash
# Install dependencies
bun install

# Run the linter and type-checker
bun run check

# Run the test suite
bun run test

# Start the dev server with file watching
bun run dev
```

Please make sure `bun run check` and `bun run test` both pass before requesting a
review.

---

## Commit messages

Use short, imperative-mood subject lines (e.g. `fix: correct edge case in q2Encode`).
Scope prefixes (`feat:`, `fix:`, `chore:`, `docs:`, `test:`) are encouraged.

---

## Code style

- TypeScript strict mode is enabled; `@typescript-eslint/no-explicit-any` is an error.
- Follow the patterns already present in `src/` for new files.
- Tests live in `test/` and use [Vitest](https://vitest.dev/).
