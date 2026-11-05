# arXiv submission notes

**Upload:** `arxiv_submission.tar.gz` (LaTeX source with the precompiled bibliography
`main.bbl`; tested to compile with pdflatex and no BibTeX, 0 errors and 0 warnings).

**Title:** How Often Will My Agent Overspend? Certified Run-Level Budget Decisions for LLM Agents

**Authors:** Robat Das

**Primary category:** cs.LG (Machine Learning). **Cross-list:** stat.ML, cs.AI

**Comments:** 16 pages, 2 figures, 6 tables. Code, data and the Baar-Core library:
https://github.com/orvi2014/Baar-Core (commit 5dc764699a99436c85dd110b3aadb1b5c1130807, release v0.12.1)

**License (suggested):** CC BY 4.0

**Abstract (1796 characters; arXiv limit 1,920; math in $...$ is rendered by arXiv):**

A budget enforcer for an LLM agent must decide before every call whether it is affordable, without knowing how long the response will be. Reserving the expected output gives no control over how often a run overspends; reserving the maximum output never overspends but can make capable models unaffordable. We certify the guarantee in between, that a whole run exceeds its budget with probability at most $\epsilon$ (with confidence $1-\delta$), by testing a family of reservation policies in a fixed sequence with exact binomial $p$-values (Learn-then-Test). It needs only i.i.d. runs, and for admit/refuse controllers one batch of at least $\ln(1/\delta)/\epsilon$ logged runs serves every policy. Per-call risk levels do not translate into run-level risk, and per-step union bounds forfeit most of the capability a budget can buy. In simulation, 320,000 calibrations scored against precisely known risks selected an unsafe policy in at most 5.3% of cases in any setting (95% CI 4.4--6.4%), consistent with $\delta = 5%$. On 2,100 real gpt-4o-mini and gpt-4o runs replayed under the shipped controller, no setting showed a significant violation; gains over certified expected-cost reservation are small at $\epsilon = 0.05$ and large at tight tolerances. On 3,000 long-horizon coding-agent trajectories no policy overspends, but runs that cannot finish within budget absorb up to 61% of spend. For such agents we certify a second decision, stopping a run early: with at most 5% of all runs stopped although they would have solved their task, the certified rule saved 11--23% of spend, while an uncertified threshold was unsafe at tight budgets. The price of one maximum-length output relative to the budget predicts which decision matters. Both methods ship in the open-source library Baar-Core.

## Before you submit
- First-time submitters to cs.LG usually need an endorsement; arXiv shows how when you start.
- Competing interests: the paper says "The author maintains Baar-Core". A commented-out line in
  `main.tex` (search "noburn") adds the commercial interest; uncomment it for venues that require it.
- After arXiv assigns an ID, TMLR submission goes through OpenReview (https://openreview.net/group?id=TMLR)
  using the same PDF; TMLR requires an anonymised version (remove author block and the GitHub link,
  or use their template).
