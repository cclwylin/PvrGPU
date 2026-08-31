# GitHub Preparation Checklist

Before pushing this workspace to GitHub:

- [ ] Confirm repository visibility: private first is recommended while the Gallium driver is still early.
- [ ] Choose a license, or intentionally keep no license.
- [ ] Keep `config/local.env` local only.
- [ ] Keep captures and generated reports outside the repository.
- [ ] Configure and build the native targets:

  ```bash
  cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo
  cmake --build build
  ```

- [ ] Run the source-tree guard directly:

  ```bash
  python3 tests/check_source_tree.py --root .
  ```

- [ ] Run the registered native unit, smoke, and Python regression gates:

  ```bash
  ctest --test-dir build --output-on-failure
  ```

- [ ] Run Python tests:

  ```bash
  PYTHONPATH=tools PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -v
  ```

- [ ] If committing from a fresh clone, configure local paths by copying:

  ```bash
  cp config/local.env.example config/local.env
  ```

- [ ] Push only after checking staged files:

  ```bash
  git status --short
  git diff --cached --stat
  ```
