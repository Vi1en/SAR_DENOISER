# Contributing

Thanks for helping improve this project.

## Before you open a PR

1. **Setup** — See **[docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)** for environment install, training, evaluation, and checks.
2. **CLI** — Prefer the unified entrypoint when it fits your workflow:

   ```bash
   python sar.py train --config configs/train/smoke.yaml
   python sar.py eval --no-run-log
   python sar.py verify
   python sar.py api -- --host 127.0.0.1 --port 8000
   python sar.py streamlit
   ```

3. **Tests & lint** (aligns with CI):

   ```bash
   pip install -r requirements-dev.txt
   pytest -q
   ruff check algos/evaluation.py sar.py tests inference api workers evaluators scripts
   ```

4. **Changelog** — For user-visible or structural changes, add an entry to **[fixes/updates.md](fixes/updates.md)** (newest first), matching the existing section layout.

## Pull request process

Same flow as **[README.md](README.md) → Contributing**: fork, branch, commit, push, open a PR with a short description of what changed and why.

## Questions

Use your repository’s **Issues** tab (or course forum) so others can find answers later.
