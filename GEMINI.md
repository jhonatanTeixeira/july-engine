### SYSTEM RULES FOR PYTHON COMMANDS
- You MUST ALWAYS append the `--no-cache-dir` flag to any `pip install` commands you generate or suggest.
- Example of valid output: `pip install <package_name> --no-cache-dir`
- Do NOT generate pip commands without this flag under any circumstances to prevent local cache bloat.
- the tests for this system must be run with pytest inside this local_models folder, where this md is located, otherwise, some file loadings fails