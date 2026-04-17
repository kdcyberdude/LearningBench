# ObsBench: Hidden Process Inference in LLMs
**30 novel tasks — can the model infer a hidden machine, cipher, or rule from demonstrations alone, with no interaction and no feedback?**

This benchmark measures **process inference from demonstrations alone** — no interaction, no feedback, no correction. Every task presents a sequence of input→output demonstrations of an invented machine, cipher, or rule. Models must infer the hidden process from those demonstrations and then correctly apply it to novel inputs. 


| Task | What it tests |
|---|---|
| affine-transform-chain | Sequential affine maps (y = ax + b mod n) chained; infer full chain from I/O pairs |
| agglutinative-morphology | Agglutinative suffix stacking rule; infer from word-paradigm examples |
| arithmetic-entropy-coding | Arithmetic encoding with a hidden probability model; infer model from encoded/decoded pairs |
| auction-mechanism-second-price | Second-price auction variant with hidden tie-breaking rule; infer from bid/outcome logs |
| codon-table-translation | Alien genetic code: infer the full codon→amino-acid table from sequence pairs |
| feistel-cipher-round | Feistel round function; infer from plaintext-ciphertext block pairs |
| finite-state-transducer | FST inference: state transitions and output symbols from I/O tape pairs |
| flow-network-capacity | Max-flow with hidden edge capacities; infer capacities from source/sink flow observations |
| hidden-attribute-rule | Multi-attribute classification; infer conjunction/disjunction predicate from labeled examples |
| hidden-damping-physics | Damped oscillator: infer the damping constant from position-time samples |
| hidden-matrix-fill | Matrix completion with a hidden algebraic pattern; infer rule from partial fills |
| hidden-modal-logic-kripke | Kripke frame inference: infer accessibility relation from modal formula truth values |
| hidden-modal-logic-kripke2 | Kripke inference, second variant: different frame structure and operator mix |
| hidden-priority-order | Non-standard priority queue: infer the hidden scoring function from enqueue/dequeue logs |
| hidden-token-filter | Token classification: infer the filter predicate from accept/reject examples |
| lattice-meet-join | Abstract lattice: infer the partial order and meet/join operations from examples |
| lfu-cache-eviction | LFU variant with hidden tie-breaking policy; infer eviction rule from access/eviction logs |
| linear-feedback-shift-register | LFSR: infer feedback tap positions from output bit sequence |
| mealy-machine-output | Mealy machine: output depends on both state and input; infer from I/O traces |
| phonological-alternation-harmony | Vowel harmony alternation rule; infer from surface-form pairs |
| pipeline-hazard-stall-counting | CPU pipeline with hidden hazard conditions; infer stall rules from instruction sequences |
| pushdown-automaton-inference | PDA: infer stack alphabet and transition rules from accept/reject strings |
| regex-intersection-membership | Three hidden regex constraints; infer membership rules from positive/negative examples |
| ring-operations-hidden-carry | Ring arithmetic with a hidden carry propagation rule; infer from expression/value pairs |
| shapley-values-cooperative-game | Cooperative game with hidden characteristic function; infer from coalition/value pairs |
| sigil-naming | Invented symbol-to-name system with compositional rules; infer from examples |
| singleton-gate-local-max | Boolean gate with hidden local-max semantics; infer from input/output pairs |
| syntax-tree-rewrite | Tree grammar rewrite rules; infer production rules from derivation examples |
| titration-curve-diprotic | Diprotic acid titration: infer pKa values from pH-volume curve segments |
| two-counter-machine | Minsky two-counter machine: infer opcode semantics from register-state execution traces |

Grading is programmatic: the reference output is produced by the same generator that created the demonstration examples. No LLM-as-judge, no approximate matching.

---

## Domain coverage

The 30 tasks span 5 broad families, ensuring no single representational prior dominates:

| Domain | Tasks | Examples |
|---|---|---|
| Formal automata & languages | 7 | FST, PDA, Mealy machine, LFSR, two-counter machine, syntax tree rewriting, regex intersection |
| Mathematics & abstract structures | 8 | Affine chains, ring arithmetic, lattice operations, matrix fill, arithmetic entropy coding, flow network, modal logic ×2 |
| Computer systems & circuits | 5 | CPU pipeline hazards, LFU cache eviction policy, priority queue scoring, Feistel cipher rounds, singleton gate logic |
| Language, symbol & code systems | 6 | Agglutinative morphology, phonological harmony, sigil naming, token filter, attribute rule, codon table |
| Physical & economic models | 4 | Damped oscillator, diprotic titration, second-price auction mechanism, Shapley values |

The span is deliberate. Observational learning is domain-agnostic: the same cognitive act — inferring a hidden process from demonstrations — applies whether the process is a formal automaton, a physical law, or an economic mechanism. A model that can only do one category is not exhibiting observational learning; it is applying domain priors.

---

## What this benchmark reveals

When a model gets all demonstrations at once with no ability to ask questions or receive corrections, the task reduces to one thing: construct the hidden process, verify it against every example, and commit. Whether that process is a finite-state machine, a physical damping law, or an auction tie-breaking rule does not matter. The cognitive act is the same, and most models cannot do it reliably.

The strongest single variable is extended reasoning. Qwen Thinking scores 0.668; the same model in instruct mode scores 0.241 — a +0.43 gap. Models that use deliberate step-by-step reasoning consistently dominate. The bottleneck is not recognizing a surface pattern but mentally simulating the process against all demonstrations before answering.

**Recognition shortcuts inflate scores on familiar structures.** `pushdown-automaton-inference` and `regex-intersection-membership` are solved by nearly every model (≥0.75). `shapley-values-cooperative-game`, `feistel-cipher-round`, and `lattice-meet-join` score 0.000 for 11 of 14 models. The difference is recognisability — PDA and regex are well-documented structures; models apply a known algorithm rather than reading the demonstrations. Tasks requiring genuine induction from the examples alone expose the real gap.

**Tasks designed with deceptive early evidence systematically score lower.** The benchmark includes tasks where the first few demonstrations are consistent with a simpler incorrect model — the correct pattern only becomes distinguishable across the full set. `hidden-damping-physics` is the clearest instance: the first 3 data points fit a simple linear model; the true damping constant requires fitting the longer sequence. Most models score 0.000–0.250 on it. The claim is not limited to this task: the pattern holds across structurally similar tasks where early commitment is possible.



