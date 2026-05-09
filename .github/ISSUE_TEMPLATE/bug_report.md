---
name: Bug report
about: Something isn't working as expected
title: '[bug] '
labels: bug
assignees: ''
---

**Describe the bug**
A clear, concise description of what went wrong.

**Minimal reproduction**
```python
from baar import BAARRouter

router = BAARRouter(budget=0.10)
# smallest code that reproduces the issue
```

**Expected behavior**
What you expected to happen.

**Actual behavior**
What actually happened — include the full traceback if there is one.

**Environment**
- `baar-core` version: (run `pip show baar-core`)
- `litellm` version: (run `pip show litellm`)
- Python version:
- OS:
- Model(s) used (e.g. `gpt-4o-mini`, `claude-3-haiku`):
- Store type: MemoryBudgetStore / FileBudgetStore / SQLiteBudgetStore
- sync / async / streaming:
