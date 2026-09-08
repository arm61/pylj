## Contributing

Contributions to pylj are welcome. Fork the repository and open a pull request.

If you want to find a good issue to get you in the door as a contributor, check out the issues marked as [good first issue](https://github.com/arm61/pylj/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) in the GitHub issue tracker.

#### Notes

- Work on a branch, and install the development dependencies with `pip install -e ".[dev]"`.
- Before offering a pull request, run `ruff check pylj`, `mypy pylj` and `pytest`, and make sure they all pass.
- Use British spelling and Google-style docstrings in your code.
- The documentation is built with `pip install -e ".[docs]"` and then `make html` in the `docs` directory.
- To discuss an idea before implementing it, open an issue on GitHub.
- If you would like to offer a pull request, we will try our best to assess and merge them as appropriate in a timely manner.

## Releasing

1. Set `__version__` in `pylj/__init__.py` and `version` and `date-released` in `CITATION.cff`.
2. In `CHANGELOG.md`, rename the `Unreleased` heading to the version and date, and add an empty `Unreleased` above it.
3. Merge, tag the merge commit with the bare version (`2.0.0`), and publish a GitHub release from the tag; tick "pre-release" for a beta.
4. The release workflow builds the package, checks the version matches the tag, and publishes to PyPI by trusted publishing.
