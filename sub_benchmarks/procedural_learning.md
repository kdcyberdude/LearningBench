# ProcBench: Skill Acquisition Across Trials
**11 multi-round tasks: does practice improve performance, and does what was learned transfer to new instances?**

Every task runs as a repeated-episode loop: the model acts inside a hidden environment, receives feedback, and repeats across five practice episodes. After practice, it faces held-out test instances with no further hints. The composite score is:

**Score = `0.30 × transfer + 0.25 × asymptote + 0.25 × trajectory + 0.20 × consistency`**

- **Transfer** — accuracy on held-out instances after practice ends (no feedback)
- **Asymptote** — mean score on the final half of practice episodes (peak skill reached)
- **Trajectory** — OLS slope of per-episode scores over practice (positive = improving)
- **Consistency** — weighted mean of practice scores, later episodes weighted more

| Task | What it tests |
|---|---|
| adaptive-sort-rule | Hidden multi-key sort; swap-hint feedback across practice lists; cold lists at test |
| boolean-circuit | Hidden fault in a Boolean network; probe outputs; transfer on a new net |
| dialect-morphology | Hidden composed string transforms; probe `transform`; transfer on unseen words |
| grammar-induction | Hidden string language; membership queries + regex submit; held-out grammar test(s) |
| lights-out-variant | Variant Lights Out; toggles + feedback; transfer on new boards |
| nim-variant | Hidden Nim-style game rule; legal moves + outcomes; transfer games |
| opponent-strategy | Hidden scoring rule per episode; play to target; transfer episodes |
| packet-filter | Hidden 2-condition firewall; packet probes; transfer rules |
| sql-reverse-engineering | Hidden `WHERE` over a fixed employee table; query vs submit; transfer clauses |
| state-machine-password | Hidden DFA; accept/reject-at-step feedback; transfer DFAs |
| voting-protocol | Hidden aggregation schema per episode; outcome feedback; transfer schemas |

---

## What this benchmark reveals

Across 14 models and 11 tasks, practice performance and transfer are not reliably correlated — and the gap between them is the core finding. On `dialect-morphology`, all 14 models reach a mean practice asymptote of 0.821 but score 0.000 on every transfer test, because the practice environment allows querying the answer directly without ever inducing the underlying rule; standard evaluation would record only the high practice score and miss the failure entirely. The reverse also holds: on `lights-out-variant`, models achieve a mean transfer score of 0.733 despite a practice asymptote of only 0.183, showing that the procedure was understood but not consistently executed under the within-episode efficiency penalty. On `boolean-circuit` and `nim-variant`, both practice and transfer are near zero across all 14 models — five feedback-rich episodes are not enough to bootstrap the required compositional reasoning in any current model, frontier or otherwise. The benchmark's contribution is measuring all three of these behaviours under the same protocol: it distinguishes models that exploit the practice environment from those that genuinely acquire a transferable procedure, and it identifies tasks where the procedural capability does not yet exist in current LLMs at all.

**Evaluation cost and an open question.** Each task runs as a genuine multi-turn conversation — the model acts, waits for environment feedback, then acts again, for up to 25 turns per episode across 5 practice episodes and 4 transfer tests. A single task evaluated across 14 models consumes roughly 840–1,800 API calls, which is why the sub-benchmark covers 11 tasks rather than a larger set. This cost also means ablation studies — varying episode count, turn budget, or feedback granularity — remain largely undone. There is also an open methodological question the current results cannot resolve: most LLMs are trained on data where multi-turn interaction is relatively sparse compared to single-pass question answering, which may explain part of the flat learning trajectory observed. Whether the failure is a fundamental limitation of current architectures or an artifact of how models are trained is a question this benchmark surfaces but does not answer.

https://www.kaggle.com/benchmarks/kdcyberdude/procedurallearningbench