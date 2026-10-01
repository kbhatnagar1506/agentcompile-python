# Contributing

Thanks for helping. Issues and pull requests are welcome.

## Setup

```
pip install -e ".[dev]" ruff
pytest
ruff check src tests && ruff format --check src tests
```

## Pull requests

- One change per pull request, with a test.
- Commit messages follow [Conventional Commits](https://www.conventionalcommits.org): `fix: ...`, `feat: ...`, `docs: ...`, `chore: ...`. They decide the next version and the changelog.
- CI must pass on Python 3.9 and 3.13 before a pull request can merge.
- The SDK must stay safe in someone's live path: it never raises into the caller's agent, never edits the caller's prompts, and fails open.

## Releases

Maintainers merge the release pull request that release-please keeps open; that tags the release and publishes it to PyPI.
