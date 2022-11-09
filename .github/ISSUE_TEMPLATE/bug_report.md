---
name: Bug report
about: Create a report to help us improve
title: ''
labels: ''
assignees: ''

---

# Bug report

**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior, for example:
1. `git clone ...`
2. `git checkout ...`
3. `cd s3prl; pip install -e ".[all]"`
4. `cd s3prl; python3 run_downstream.py ...`

**Expected behavior**
A clear and concise description of what you expected to happen.

**Screenshots**
If applicable, add screenshots to help explain your problem.

**Environment**
List your **OS, Python, and PyTorch, and S3PRL** versions

- If you install S3PRL from PyPi (with `pip install s3prl`):
    - we expect the version to be like `v0.x.x`
- If you install S3PRL in the editable mode (with `pip install -e ./` after cloning S3PRL repo):
    - we expect the version to be a **commit hash** like `683a6043` (this is the first 8 characters for the full 40 characters hash, which is usually long enough to be unique)
