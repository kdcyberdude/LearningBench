Comprehensive LearningBench Task Catalog
Associative Learning (17 tasks, excluding 3 numeric-ID legacy files)
All tasks are single-turn (one prompt, structured multi-question response). The model is given training observations and must answer questions about novel combinations.

blocking-effect: Tests causal blocking/confounding in a 20-trial allergy log. 8 questions spanning causal insufficiency (UNKNOWN for confounded substances), confirmed solo-established effects, compositional prediction, and meta-epistemic counting (how many compounds are conclusively classified). Distinct: classical blocking paradigm with 4 structurally different confounding reasons.
latent-cross-binding: Tests hidden two-dimensional binding rule (letter-count parity + vowel count must both match for association). 6 YES/NO queries on novel word pairs. Distinct: conjunction of two latent structural properties on invented words.
latent-set-binding: Tests directed tripartite set membership. 9 tokens partitioned into 3 ordered tiers; pairs VALID iff tier(X) < tier(Y). 7 queries, including one token (Zhovek) never seen in training. Distinct: hierarchical ordering with a fully held-out token requiring process-of-elimination.
latent-set-variant: Tests three-way set binding. 9 words into 3 groups; a triple is VALID iff one word from each group. 7 queries from 11 training triples. Distinct: 3-way combinatorial grouping rather than pairwise.
sensory-preconditioning: Four novel infix operators (QREL, VARN, TYLX, WIRM) on digits 0-9 must be induced from ~128 examples. 8 queries test direct use, composition, inverse, and commutativity. Distinct: multi-operator induction with modular arithmetic.
serial-chain-reconstruction: Dual-phase glyph-to-integer bijection + hidden binary operation ((a*b + a) mod 9) + 1. 12 coupled symbolic equations to solve. Distinct: simultaneous operation inference and symbol grounding.
overexpectation: Tests associative overexpectation from numerical conditioning logs. Two labs with solo → compound → retest phases. 8 questions (integers + YES/NO) testing whether Phase-2 attribution preserves Phase-1 ratios (NO -- a consistency trap). Distinct: quantitative sub-additivity reasoning.
xor-attribute-binding: Three hidden binary properties (vowel-parity, letter-parity, ends-in-vowel) with a mixed XOR/XNOR conjunction rule for BOND. 6 queries including one with ambiguous 'y' (UNKNOWN). Distinct: 3D feature conjunction with asymmetric interaction.
inhibitory-summation: Six signals with different hidden roles (strong/weak excitor, partial/full inhibitor, modulator, neutral) in a 3-outcome system (FLARE/DAMP/INERT). 8 novel-combination queries from 18 observations. Distinct: quantitative inhibitory algebra with hierarchical modulation.
counterfactual-sequence-rewrite: Hidden cumulative state machine rewriting rule. K/J are control tokens (increment/decrement mod 4 register); non-special tokens are output as P^state(T) where P is a 4-cycle. 500 training examples, 8 test sequences testing non-local state tracking. Distinct: cumulative stateful transformation with control-token mechanism.
learned-irrelevance: Tests the learned-irrelevance effect from session-by-session conditioning rate data across 4 experimental groups with different pre-exposure correlations. Must infer acquisition ordering and project to a novel Group E. Distinct: learning-theoretic paradigm with numerical CR rate trajectories.
occasion-setting: Dual-path hierarchical occasion setting with 7 modulators/pulses and two deliberate co-occurrence traps. 14 probes testing OPEN/CLOSED outcomes. Distinct: rich Boolean logic with pulse-specific gating and statistical traps.
retrospective-revaluation: 6 players across 12 matches under a "differential credit" rule (balance-dependent credit allocation). 7 questions including retroactive queries, credit allocation, and counterfactual revaluation. Distinct: multi-step ledger tracking with counterfactual reasoning.
second-order-extinction: Four-phase cross-interference extinction with partial reinforcement, blocking, and second-order chains (KREL->MOVEN->THASP). 10 YES/NO/UNKNOWN probes. Distinct: multi-phase conditioning with extinction/reintroduction and epistemic uncertainty.
spurious-hue-true-edge: Spurious 75%-reliable correlate (tint) vs. true XOR rule over two feature partitions. 4 test items all have tint pointing wrong. Distinct: explicit spurious correlation resistance + configural XOR rule.
temporal-pairing-kmp: Gated dual-signal binding with 6 signal tokens and 3 gate types (^=first, *=second, ==max-rank). 500 examples, 6 queries including order traps. Distinct: positional-vs-rank selection gates.
temporal-pairing-tnr: Temporal binding with multiplicative-vs-additive delay and two-dimensional identity switch triggered by a '*' modifier. 500 examples, 4 test sequences. Distinct: dual formula switch (arithmetic + identity) triggered by context token.
Concept Learning (18 tasks)
All tasks use a multi-turn active-learning protocol: the model requests labeled examples, then answers exam questions. Scored on accuracy x sample efficiency.

consonant-clusters: Two-component word transformation: odd-length consonant clusters reversed; vowels shifted cyclically by preceding cluster length. Distinct: interacting phonological rules on string structure.
digit-cipher: Positional affine cipher on integers: letter = ALPHA[(d3 + i7) % 26]. Same digit maps differently by position. Distinct: modular arithmetic with positional dependence.
grid-transform: Fixed spatial transformation on 5x5 grids: transpose then horizontal flip. Distinct: compositional spatial operations.
nested-logic: Hidden logical rule over 5 binary features: YES iff (A==B) AND sum(C,D,E) is odd. Distinct: conjunctive structure with equality + parity.
state-machine: Hidden state machine: start at 0; X=+1, Y=-1, Z=reset to 0; YES iff final state > 0. Distinct: stateful sequential processing with reset.
semantic-override: Structural rule masked by semantic distractors: YES iff word contains adjacent identical letters. Uses real English words (animals, objects). Distinct: semantic-vs-structural conflict.
violation-counter: Count violations in integer sequences: valid iff value > previous OR value == 1. Distinct: sequential rule with exception clause.
relational-pairs: Hidden mathematical relationship: VALID iff b == (a*a) mod 7. Distinct: modular quadratic residue.
triple-parity: YES iff digit_sum(value) mod 4 == index mod 4. Distinct: dual modular computation on value and position.
disjunctive-noise: Disjunctive rule (weight > height OR color_code % 3 == 0) with material as red herring and 2 structurally noisy labels. Distinct: noisy disjunctive multi-attribute classification.
dual-recurrence: Coupled recurrence sequences mod 17: A(n) = (A(n-1)+B(n-1)) % 17, B(n) = (A(n-2)*B(n-2)) % 17. Distinct: interleaved coupled recurrence.
encoded-triple: Modular arithmetic over 5 nonsense symbols: C = (A+B) mod 5 with symbol-to-value mapping unknown. Distinct: joint symbol grounding + modular addition.
interleave-reverse: String transformation: split by index parity, reverse odd-index stream, interleave back. Distinct: split-reverse-merge permutation.
layered-transform: Three-step string transformation: remove vowels -> shift i-th consonant by +i -> reverse. Distinct: ordered multi-step pipeline with positional shift.
modular-subsequence: Count indices where seq[i] mod 3 == i mod 3. Distinct: dual modular counting.
positional-encode: Keep letters iff (vowel at even index) OR (consonant at odd index). Distinct: type-position conjunction filter.
positional-mapping: YES iff 1-based index of the single uppercase letter equals the trailing integer. Distinct: cross-modal letter-position to number relationship.
vowel-rotation: Cyclic right-shift of the vowel subsequence within a word, reinserting at original positions. Distinct: subsequence-specific cyclic permutation.
Language Learning (26 tasks)
All tasks use multi-turn active-learning with invented/conlang languages. The model requests examples and must produce correct surface forms or interpretations.

dimval-metathesis: Morphophonology with suffix vowel harmony, syncope, metathesis, and nasal hardening in fixed order. Distinct: 4-rule ordered phonological pipeline.
drafnelt-switch-reference: Switch-reference markers (-ven same-subject, -dru different-subject) + logophoric pronoun binding asymmetry. Distinct: discourse-level reference tracking.
dralven-tone-sandhi: Tone sandhi rules in an invented tonal language. Distinct: context-dependent tone changes.
drelkovak-harmony: Dual simultaneous harmony: vowel harmony (back/front) + pharyngeal consonant harmony. Distinct: two independent harmony systems interacting.
drelvak-reduplication: Three semantically contrastive reduplication types (distributive, attenuative, intensifying) with different surface forms + nasal interactions. Distinct: form-meaning pairing in reduplication.
grelkan-suppletion: Context-sensitive verbal suppletion across 3 verb classes with different stem-alternation patterns (2-way, 3-way, 5-way splits). Distinct: unpredictable suppletive paradigms.
gwelthar-mirative-evidential-tone: 3-way tonal interaction: lexical tone, evidential suffix tone, and mirative floating H tone. Distinct: suprasegmental tone stacking.
kelstran-tone: Lexical tones encoding grammatical mood (declarative, interrogative, imperative, subjunctive) + aspect-triggered tone displacement. Distinct: grammatical function encoded in tone.
kophar-quantity: Mass/count noun syntax with measure words + base-6 numeral system. Distinct: noun classification + non-decimal numerals.
mixed-radix-number (KROMATH): Compositional verb morphology: PAST geminates, PASS prefixes, INFER appends. Distinct: simple morphological composition.
norkvash-scalar: Scalar implicature, entailment, and exhaustivity with quantifiers (all/most/some/few/none). Distinct: pragmatic interpretation.
pelvan-agreement: Multi-probe verbal agreement: 5-dimensional grid (person x number x gender x animacy x case) with split agreement and animacy override. Distinct: highest-dimensional agreement system.
prentova-allomorphy-wugtest: Opaque allomorphy with 3 declension classes (regular, ablaut, suppletive) + determiner agreement. Wug-test paradigm filling. Distinct: class-conditioned morphological alternation.
skelth-allomorph: 5-way suffix allomorphy conditioned by BOTH phonological AND morphological environments + V-initial mutation. Distinct: dual conditioning of allomorphy.
sklonveth-root-pattern: Semitic-style root-and-pattern (trilateral consonant roots woven into 8 CV-skeleton patterns) + negation triggers prosodic inversion. Distinct: templatic morphology.
skolvren-polysynthetic: 6-slot polysynthetic verb template with noun incorporation and classifiers. Distinct: complex agglutination with incorporation.
skovar-deletion: Deletion + epenthesis interaction for cluster repair (final CC, initial CCC) with glottal stop insertion. Distinct: interacting repair rules.
strelkov-ergative: Split ergativity (PAST=ergative/absolutive, PRESENT=nominative/accusative) + antipassive voice interaction. Distinct: case alignment conditioned by tense + voice.
strevoklan-neg: Negation morphology in an invented language. Distinct: scope of negation marking.
telvari-evidentiality: 6 evidential categories (direct-visual, direct-nonvisual, inferred-result, inferred-reasoning, reported-hearsay, reported-reading) + compound evidential chaining. Distinct: inference chain evidentials.
threlkav-scope-ergativity: Quantifier scope x 3-way split ergativity (PAST/PRESENT/FUTURE different alignments) where zero-marked cases = wide scope, overt-marked = narrow scope. Distinct: scope-case interaction.
trenval-bleeding: All 4 phonological rule ordering relationships: feeding, bleeding, counterfeeding, counterbleeding. Distinct: explicit rule interaction typology.
trevkovan-gradation: Finnish-style consonant gradation (9 pairs) + morphological trigger grid (6 cases alternate strong/weak). Distinct: systematic stem alternation.
vrelthan-rule-interaction: 3 rules in fixed order (nasal place assimilation, obstruent voicing, nasal deletion) creating opaque surface forms. Distinct: opacity from rule ordering.
vrendel-templatic: Circumfix + prosodic template + allomorphy across 3 verb classes. Distinct: non-concatenative morphology with prosodic constraints.
wukal-tones: Tone spreading + OCP (Obligatory Contour Principle) + nasal blocking. Distinct: tonal processes and their precedence.
Observational Learning (30 tasks)
All tasks are single-turn: the model observes demonstration input-output pairs and must answer 4 test questions.

affine-transform-chain: Chain of 3 affine transforms (rotation, scale, translation); predict output points. Distinct: geometric composition.
agglutinative-morphology: 5-slot agglutinative language with vowel harmony + direct evidential fission. Distinct: morphological agglutination with harmony.
arithmetic-entropy-coding: Infer 5 hidden symbol probabilities from bit-length observations; ceiling of sum of log2. Distinct: information-theoretic inference.
auction-mechanism-second-price: Sealed-bid auction with hidden reserve price R, buyer's premium B%, seller's discount S%. Distinct: mechanism design inference.
codon-table-translation: Alien 3-nucleotide codon system with Latin-square structure; infer amino acid mapping from partial observations. Distinct: biological code cracking.
feistel-cipher-round: 2-round Feistel cipher with hidden affine round function; decrypt test ciphertexts. Distinct: cryptographic reversal.
finite-state-transducer: Symbol-based meaning flips using paired * and @ modifiers; even/odd flip counting. Distinct: compositional semantic flipping.
flow-network-capacity: Path-graph edges with tag priorities determining max flow selection. Distinct: hidden priority ordering in networks.
hidden-attribute-rule: Objects with 5 symbolic attributes labeled PASS/FAIL by a stateful rule with internal state evolution. Distinct: stateful classification.
hidden-damping-physics: Damped oscillator Aexp(-gammat)cos(omegat+phi); predict future amplitudes. Distinct: continuous physics model inference.
hidden-matrix-fill: Ternary cellular automaton (27-entry lookup table) applied to grids; fill missing rows. Distinct: CA rule table inference.
hidden-modal-logic-kripke: Reconstruct atom valuation + accessibility relation in 5-world Kripke frame from nested modal formulas. Distinct: modal logic model theory.
hidden-modal-logic-kripke2: Same as above but different instance. Distinct: second Kripke frame instance.
hidden-priority-order: Word lists sorted by hidden 3-key comparator (vowel count, last character, word length). Distinct: multi-key sort inference.
hidden-token-filter: 2-condition word filter (odd length AND no repeated letters) with deceptive Phase 1. Distinct: conjunctive filtering with deception.
lattice-meet-join: Infer C2xC4 product lattice from meet/join observations with permuted labels. Distinct: abstract algebraic structure.
lfu-cache-eviction: LFU-LRU cache eviction policy; distinguish from LRU and FIFO on test sequences. Distinct: systems/architecture policy inference.
linear-feedback-shift-register: 8-bit LFSR with 3 hidden tap positions; predict next bits. Distinct: hardware/crypto sequence prediction.
mealy-machine-output: 3-state Mealy transducer inference from 6 I/O demos; predict output on 4 held-out strings. Distinct: automata theory.
phonological-alternation-harmony: Progressive consonant voicing harmony in CC clusters; early demos show no clusters. Distinct: phonological assimilation with progressive reveal.
pipeline-hazard-stall-counting: Infer 3 hidden constants (boot overhead, tier costs, contention penalty) from execution traces. Distinct: computer architecture cycle counting.
pushdown-automaton-inference: Infer CFL a^n b^(n+k) c^k from accept/reject examples (initially looks like a^n b^n). Distinct: formal language inference.
regex-intersection-membership: Infer 3 independent structural constraints on {a,b,c} strings from membership observations. Distinct: language intersection inference.
ring-operations-hidden-carry: Ring with hidden carry K: a+b = (a+b+K) mod M; early demos hide the carry. Distinct: abstract algebra with deceptive start.
shapley-values-cooperative-game: Novel "inverse-solo-weighted" allocation rule for 3-player games (NOT Shapley/Nash). Distinct: game theory -- completely novel rule.
sigil-naming: 3-component naming rule for tile sequences (prefix from first P-tile size, body from center tile, suffix from F-count). Distinct: multi-dimensional naming with deceptive initial demos.
singleton-gate-local-max: Singleton-gate partitioning + intra-block local-maxima selection. Distinct: sequence partitioning + selection.
syntax-tree-rewrite: Deterministic rewrite on bracketed binary trees. Distinct: tree transformation.
titration-curve-diprotic: Four-regime piecewise cyclic function with disjoint Unicode symbol alphabets per regime. Distinct: regime segmentation + modular cycling.
two-counter-machine: Infer semantics of 7 opaque opcodes from execution traces of a two-counter machine. Distinct: instruction set reverse engineering.
Procedural Learning (11 tasks)
All tasks use multi-turn trial-and-error with feedback loops: 5 practice instances (learning phase) then final test instances. Scored on learning efficiency + final test accuracy.

adaptive-sort-rule: Hidden 3-key sorting rule (odd-last-digit first, then descending digit-sum, then value). Swap-position hints after wrong attempts. Distinct: multi-key sort with hint feedback.
boolean-circuit: Hidden 4-gate Boolean circuits (AND/OR/NOT, 3 inputs). Probe input triples to deduce 8-entry truth table. Distinct: combinatorial circuit reverse-engineering.
dialect-morphology: Hidden 2-rule phonological transformations in a fixed order. Probe words to observe dialect outputs. Distinct: compositional rule isolation from combined outputs.
grammar-induction: Hidden CFGs (palindromes over {a,b} or a^n b^n). Test membership with strings, submit regex. Distinct: formal language identification from membership queries.
lights-out-variant: Hidden L-shaped toggle mask on 4x4 grid. Discover mask from feedback on 5 practice boards. Distinct: spatial toggle-pattern discovery.
nim-variant: 3-cache Crystal Claim games with hidden charge limit J in {2,3,4}. Learn J from opponent behavior across 5 games. Distinct: game-theoretic parameter inference.
opponent-strategy: Identify hidden behavioral rules in Iterated Prisoner's Dilemma (5 novel automata). Deliberate probing to discover rules and best-respond. Distinct: opponent modeling in repeated games.
packet-filter: Reverse-engineer hidden 2-attribute AND firewall rule from BLOCKED/ALLOWED outcomes. Distinct: network security rule inference.
sql-reverse-engineering: Reverse-engineer hidden SQL WHERE clause by querying candidates and comparing matched row IDs. Distinct: database query reverse-engineering.
state-machine-password: Find 8-symbol sequences accepted by hidden DFAs over {A,B,C,D}; rejection-step feedback. Distinct: DFA exploration with prefix-based search.
voting-protocol: Identify which of 4 voting rules (Plurality, Borda, IRV, Approval) governs elections by designing discriminating ballot configurations. Distinct: social choice mechanism identification.
Reinforcement Learning (30 tasks)
All tasks are multi-turn interactive with a step budget. The model takes actions and receives environment feedback (often noisy/obfuscated). Scored on solve-within-budget efficiency.

affine-cipher-word: Random monoalphabetic substitution; recover plaintext using graded cipher-consistent position telemetry. Distinct: codebreaking with partial feedback.
arithmetic-next: Hidden integer; BIT_OVERLAP (popcount of XOR) feedback. Distinct: bitwise distance oracle.
base7-decode: Alternating-position glyph substitution over base-3 digit strings with unknown endianness; MAP probes return mismatch mass. Distinct: encoding inference with endianness.
battleship-1d: Hidden 4-5 slot rigid object on 20-slot line with delayed sensor feedback. Distinct: 1D search with echo delay.
battleship-two-ships: Two disjoint length-2 craft on 24-slot line; 2-slot salvos return aggregate hits without per-beam attribution. Distinct: multi-target search with aggregate feedback.
bitstring-hamming: Hidden 14-bit binary tape (lambda/rho glyphs); DIVERGENCE count feedback with dither. Distinct: binary search with obfuscated naming.
chebyshev-point: Hidden 2D point; weighted L-infinity SHELL distance with hidden axis weights. Distinct: anisotropic 2D search.
coin-balance: 12 coins, one heavier; 3-vs-3 weighings with hidden drift bias {-1,0,+1} and 8% jam rate. Distinct: classic puzzle with hidden systematic bias.
collatz-length: Modified Collatz map with hidden odd perturbation delta; HIGHER/LOWER feedback on step-count guesses. Distinct: iterated dynamics with hidden parameter.
crt-unique: Hidden integer in [0,210); modular remainder queries with primes 7,11,13 + 15% noise. Distinct: CRT reconstruction under noise.
cubic-eval: Hidden integer cubic polynomial; VAL x queries (one may be corrupted); compute P(7). Distinct: polynomial interpolation with outlier.
digitwise-l1: Hidden 4-digit code; PRESSURE = weighted L1 with hidden per-position weights 1-4. Distinct: weighted digit-by-digit search.
divisor-count: Hidden integer n; GCD oracle queries; guess divisor count tau(n). Distinct: number-theoretic probing.
fib-like-next: Non-Fibonacci recurrence mod 97 with unknown coefficients; QUERY k reveals S[k]; guess S[6]. Wrong guesses get only parity feedback. Distinct: recurrence coefficient inference.
graph-shortest-path: Hidden weighted 5-node complete graph; noisy EDGE queries (20% +/-1); submit shortest path 0->4. Distinct: graph exploration under noisy oracle.
gray-hamming: Hidden 12-bit integer; Hamming distance after hidden REFLECT XOR-mix + 9% dither. Distinct: search through unknown bit-space transformation.
hanoi-two: 3-disk Hanoi with hidden goal peg AND hidden forbidden disk-peg placement rule. Distinct: planning puzzle with hidden constraints.
manhattan-point: Hidden 2D point; noisy tier of non-Euclidean score (L1 + weighted Linf). Distinct: metric inference + localization.
mastermind-aggregate: Hidden 6-symbol code; only aggregate COUPLING_SUM (lock+drift combined) with no split. Distinct: code-breaking with minimal feedback.
perm-footrule: Hidden permutation; FLOW cost (footrule) feedback with sensor dither. Distinct: permutation search.
product-hotcold: Hidden factors (a,b); L1 distance in factor space; product never stated. Distinct: factorization via distance feedback.
quadratic-root: Piecewise linear hidden function with unique zero; integer probes. Distinct: root finding with non-smooth function.
recurrence-second-order: Hidden order-2 linear recurrence mod M with noisy oracle (15% flip). Distinct: recurrence identification under noise.
rule90-step: Hidden 1D cellular automaton (Wolfram rule 30/90/110/150); TEST probes then predict challenge string's next step. Distinct: CA rule identification.
shift-cipher: Hidden affine index map on 31-symbol roster: cipher = (A*plain + B) mod 31. PROBE queries; PHASE_REJECT/PHASE_LOCK feedback. Distinct: affine cipher in non-standard alphabet.
sudoku-2x2: 4x4 Latin square (symbols 1-4) with partial clues; unique completion. Synthetic band feedback. Distinct: constraint satisfaction with obfuscated feedback.
sum-product-xy: Hidden (x,y); each turn reveals random linear functional ax+by with 12% noise. Distinct: system-of-equations from noisy random projections.
verbal-bandit: 7-arm nonstationary bandit with verbal payoff tiers; explore then commit. Distinct: classic explore-exploit with natural language feedback.
wordle-micro: Hidden 6-symbol pattern; ANCHOR/SWAY/VOID telemetry with hidden per-slot weights. Distinct: Mastermind with weighted/obfuscated feedback.
xor-subset-hamming: 4 source registers; environment XORs hidden subset into 10-bit target; Hamming distance feedback with dither. Distinct: subset identification via bitwise distance.
Summary counts: Associative (17) + Concept (18) + Language (26) + Observational (30) + Procedural (11) + Reinforcement (30) = 132 tasks total.