# Design decisions (and how to defend them)

Each section states the decision, why, the alternative rejected, and the question an
interviewer is likely to ask.

## 1. Problem framing

**Decision.** Two independent outputs per request: (a) which of 16 assistant skills should
handle it, or abstain; (b) whether executing it would have a side effect that needs user
confirmation.

**Why separate.** Routing errors are recoverable (the user rephrases); a missed side effect
is not (an email is sent, a light switches off, money is spent). Coupling them - e.g. deriving
"side effect" from the predicted intent - means every routing error is also a safety error.
A separate head with its own loss and its own threshold lets the safety operating point
(>= 99% recall) be chosen independently of routing accuracy.

*Q: Why not let the LLM/agent decide?* An on-device router runs before any LLM call, in
milliseconds, on every request. It decides where the request goes and whether to ask first.

## 2. Data

**MASSIVE 1.1 (Amazon, CC BY 4.0)**, en-US / hi-IN / bn-BD. Each utterance id is one English
request professionally localised into each language, and the train/dev/test partition is
per id - so translations of the same request can never straddle a split. Verified in code
(`data/build.py::validate`, `tests/test_text_and_data.py`).

**Leakage control beyond ids.** Different ids sometimes share identical short utterances
("stop", "pause"). The build normalises text (NFC, Indic digits -> ASCII, casefold,
punctuation) and drops any dev/test row whose normalised text already exists in an earlier
split - 1,526 evaluation rows removed. Without this, test accuracy is inflated by memorised
strings.

**Held-out domains as OOD.** `cooking` and `takeaway` are removed from train and dev
entirely; their test rows measure abstention. Food was chosen because it overlaps lexically
with in-domain skills (qa, lists, recommendation, transport booking) - an easy OOD set
(e.g. random English sentences) would overstate abstention quality. No OOD data is used for
any tuning: the abstention threshold is set from in-domain dev confidence only.

**Side-effect labels** come from a reviewed mapping of MASSIVE's human intent labels
(`configs/side_effects.yaml`) - deterministic and auditable, not model-generated.

**Code-mixing, three kinds, never conflated:**
| slice | source | role |
|---|---|---|
| `human_codemix` | native rows where >= 2 of 3 MASSIVE judges said "target language + English" | real, human-judged code-mixing |
| `romanized` | rule-based transliteration (`data/translit.py`) | synthetic robustness probe + augmentation |
| `codemix`, `codemix_romanized` | slot-aligned code-switching (`data/codemix.py`) | synthetic robustness probe + augmentation |

*Q: Isn't synthetic data cheating?* Synthetic rows are never the headline test set. Headline
numbers are on human-translated test rows. Synthetic test rows are derived from test ids only
(split inherited), so augmentation never sees a test request in any form.

*Q: How do you generate code-mixed text without a model?* MASSIVE annotates slots in every
language and the translations are parallel, so the i-th `time` slot in Hindi and the i-th
`time` slot in English are the same entity. Swapping values yields "friday ko subah nine am
mujhe jaga do" with the label unchanged by construction.

*Q: Known weaknesses of the romaniser?* Medial schwa is kept (Bengali "kemana", users type
"kemon"); spelling is deterministic whereas users vary ("mujhe"/"muje"). Mitigations: the
TF-IDF model uses character n-grams; the transformer uses subwords. Real romanised user data
is the obvious next step.

## 3. Models

| | what | why it is in the comparison |
|---|---|---|
| A | multilingual keyword rules | what teams ship before data exists; also reused as the always-on lexical safety guard |
| B | TF-IDF (char 2-5 + word 1-2 grams) + logistic regression, C tuned on dev | strong, cheap classical baseline; any neural model must beat it to justify itself |
| C-frozen | XLM-R encoder frozen, MLP head on cached mean-pooled features | low-compute baseline; isolates what fine-tuning adds |
| C-finetune | XLM-R fine-tuned, embeddings frozen, all 12 layers trained | the main model |

**Tokenisation pitfall avoided (B).** Python's `\w` does not match Devanagari/Bengali vowel
signs (Unicode Mc/Mn), so the common `token_pattern=r"\w+"` shreds "मुझे". B tokenises on
whitespace after normalisation; there is a unit test demonstrating the pitfall.

**Why XLM-R base.** Covers Hindi and Bengali, and its CC-100 pretraining corpus includes
romanised Hindi and Bengali text. MuRIL (Indic-specific, also trained on transliterations) is
the natural ablation; mDistilBERT is smaller but its Wikipedia-only pretraining has little
romanised text.

**Architecture.** SentencePiece tokenizer -> encoder -> masked mean pooling -> dropout -> two
linear heads (16-way softmax; 1-logit sigmoid). Mean pooling instead of `<s>`: XLM-R was not
pretrained with a sentence-level objective on the first token, and mean pooling is more stable
for short inputs.

**Fitting in 8 GB (Apple M3).**
- Token-embedding matrix frozen: it is 192M of 278M parameters (69%). Freezing it removes its
  gradients and AdamW moments (~2.3 GB) while every attention/FFN layer still trains. With a
  250k vocabulary, most rows receive almost no gradient on 32k examples anyway.
- Checkpoints store only trainable tensors (343 MB vs 1.1 GB) plus the pinned base-model
  revision; loading asserts that every missing key is a frozen parameter.
- Padding bucketed to multiples of 16. Fully dynamic lengths made the MPS kernel-graph cache
  grow past 5 GB and training thrashed swap (~1 example/s); bucketing holds ~3.2 GB.
- Micro-batch 16 with 2-step gradient accumulation (effective batch 32).

**Optimisation.** AdamW, encoder LR 3e-5 and head LR 1e-3 (heads start from random init,
encoder from pretrained weights), no weight decay on biases/LayerNorm, linear warm-up over 6%
of steps then linear decay, gradient-norm clipping at 1.0, early stopping on dev macro-F1,
best epoch checkpointed.

**Class imbalance.** Scenario frequencies differ ~6x. Cross-entropy uses inverse-sqrt-frequency
class weights (full inverse frequency over-corrects and costs precision on large classes);
side-effect BCE uses `pos_weight = sqrt(neg/pos)`. Model selection uses macro-F1, so small
classes count equally.

**Reproducibility.** Fixed seeds (Python, NumPy, torch, DataLoader generator),
`use_deterministic_algorithms(warn_only=True)` - some MPS kernels have no deterministic
implementation, so runs are close but not bit-identical on MPS.

## 4. Evaluation protocol

- Fit on train; tune everything (C, early stopping, temperature, thresholds) on dev; score test once.
- Calibration: temperature scaling (one scalar fit by LBFGS on dev NLL, in log-space so T > 0);
  ECE with 15 equal-width bins, before and after. Temperature never changes the argmax.
- Abstention: threshold tau = 5th percentile of dev max-probability (answer 95% of in-domain
  requests); reported: coverage, selective accuracy, OOD abstention at tau, OOD AUROC, FPR@95TPR.
- Side effect: threshold = largest value with >= 99% dev recall. Reported for the head alone
  and for the gate actually used at serving time (lexical guard OR model).
- Cross-lingual consistency: share of source requests whose English/Hindi/Bengali (or
  native/romanised) forms get the same prediction - a property per-language accuracy hides.
- Latency: CPU, batch 1, 4 threads, p50/p95 over 200 sampled test requests after warm-up;
  fp32 vs dynamic int8 quantisation (all `nn.Linear` layers), with the int8 accuracy cost.

## 5. Safety design

The rule: **ML may add friction, never remove it.**
1. Lexical guard (deterministic, multilingual) - can only require confirmation.
2. Model side-effect head - can only require confirmation; if the model is unavailable the
   gate fails closed (requires confirmation).
3. Tool policy (`serving/gate.py::may_execute`) - the authoritative check. A tool that mutates
   state executes only with explicit user confirmation. It takes no model output as input,
   so no prediction can authorise anything.

Tested as a monotonicity property in `tests/test_serving.py`.

*Q: If the gate requires 99% recall, why not 100%?* Layer 3 is the hard guarantee; layers 1-2
decide when to ask early. Pushing the model threshold to 100% dev recall makes it confirm most
read-only requests too - see the precision column in `reports/results.md`.

## 6. Serving

`Router.route(text)` -> `RouteDecision` (scenario or abstain, calibrated confidence, side-effect
probability, requires_confirmation + reasons, model version `name@sha256[:12]`, fallback flag,
latency). Inference exceptions fall back to the keyword router and the gate fails closed. Logs
are structured JSON and contain a hash, length and script of the request - not the text.
FastAPI exposes `/route`, `/healthz`, `/metrics` (Prometheus text format).

## 7. What I would do next

- Native-speaker review of the challenge set; collect real romanised user requests.
- MuRIL vs XLM-R ablation; distil the fine-tuned model into a small student for on-device use.
- Energy-based or Mahalanobis OOD scores instead of max-softmax.
- Multi-seed runs with confidence intervals (bootstrap over test requests).
