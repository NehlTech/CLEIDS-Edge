# CLEIDS-Edge — Project Brief (reference for every Claude Code / Colab session)

This file is the single source of truth for the experimental pipeline behind the MPhil thesis:
**"An Edge-Enabled Hybrid Deep Learning Framework for Real-Time Intrusion Detection in
Resource-Constrained IoT Networks of TVET Institutions in Ghana: A Case Study of Suntreso
Technical Institute."**

Paste this file (or its relevant section) at the top of any Claude Code session working on this
project, so the session has full context without needing prior chat history.

---

## 1. System overview

- **System name:** CLEIDS-Edge (CNN-LSTM Edge Intrusion Detection System)
- **Core model:** Hybrid CNN + LSTM classifier — CNN layers extract spatial traffic features,
  LSTM layers learn temporal attack sequences.
- **Edge adaptation:** Post-training INT8 dynamic quantization + magnitude pruning, to fit
  within the memory/latency envelope of low-cost IoT edge hardware.
- **Framework:** TensorFlow / Keras (chosen for edge conversion path via TFLite).
- **Training compute:** Google Colab Pro, GPU runtime (L4).
- **Latency/throughput benchmarking compute:** CPU-only, single-thread — must NOT be measured
  on GPU. This is the number that supports the "edge-deployable" claim.
- **GitHub repo:** `https://github.com/NehlTech/CLEIDS-Edge.git`

## 2. Datasets (benchmark-only, no institutional data collection)

| Dataset | Role |
|---|---|
| NSL-KDD | Classical baseline / cross-dataset generalization check |
| CICIDS2017 | Modern attack diversity (DoS, DDoS, brute force, port scan, botnet) |
| IoT-23 | IoT-specific malware / botnet traffic |
| UNSW-NB15 / TON_IoT | Network anomaly detection, additional IoT context |

All datasets are public. Acquisition scripts must pull from official sources (Canadian Institute
for Cybersecurity site for CICIDS2017/NSL-KDD, Stratosphere Lab for IoT-23, UNSW site for
UNSW-NB15/TON_IoT). Never use unofficial mirrors.

**Provenance decision (2026-07-24):** the official CIC source for NSL-KDD (`unb.ca/cic/datasets/nsl.html`
dataset download) and CICIDS2017 (`cicresearch.ca` direct zip) are both dead/unreachable as of this
date (confirmed via direct fetch — soft-404s and "no longer available" responses). Both are sourced
via Kaggle mirrors instead (`hassan06/nslkdd`; `shadman1028/cicids2017-official-flow-feature-csv-files`),
logged as `kaggle-mirror` rather than `official-direct` in every manifest, per the project's
no-fabrication rule. IoT-23 (Stratosphere) and UNSW-NB15/TON_IoT (UNSW SharePoint, gated behind
interactive login) are also Kaggle-mirrored for the same reason. This is a documented deviation, not
a silent substitution.

**Split policy (Notebook 01, binding for Chapter 3 methodology):** NSL-KDD and UNSW-NB15 retain
their official published train/test partitions (confirmed by exact test-split row-count match:
NSL-KDD 22,544, UNSW-NB15 82,332), with validation carved via a stratified 90/10 split from each
dataset's official *training* file only. CICIDS2017 and TON_IoT have no canonical official
partition, so both use a uniform stratified 70/15/15 split with `random_state=42`. All four are
saved to disk exactly once (`data/processed/<dataset>/{train,val,test}.npz`) so every model in
Notebooks 03/04 trains and evaluates on identical data per dataset.

**SMOTE policy:** applied to the training split only, per dataset, with method chosen by scale:
full SMOTE-to-majority where memory allows (NSL-KDD, UNSW-NB15, TON_IoT); a 50,000-row cap for
CICIDS2017 (majority class ~2.27M rows makes full match infeasible on 7.8GB RAM); and, for
NSL-KDD specifically, classes with fewer than 6 samples (`spy`, `perl`, `phf`) are excluded from
SMOTE entirely and kept at their natural rarity, since a single global `k_neighbors` sized off
one 2-sample class would force near-duplicate interpolation across every oversampled class, not
just the rare ones.

**IoT-23 (2026-07-25): reservoir sampling + capped SMOTE, both explicit deviations from a
straightforward port of the other datasets' pipeline.**
- *Scale problem*: the full IoT-23 dataset is ~325.3M rows (real count, confirmed by streaming
  all 23 `conn.log.labeled` capture files in a single pass) / ~47.06GB uncompressed (real,
  tarfile-member-size sum, not estimated) — ~80x the combined training data of the other four
  datasets. Extracting it required more disk than this machine had free (hit 100% full / 15MB
  free mid-extraction). Resolved by streaming the compressed `tar.gz` directly via Python's
  `tarfile` module (no extraction to disk) and taking a **reservoir sample (Algorithm R,
  seed=42)** of 2,000,000 rows, uniformly across the entire stream with no bias toward any
  capture file or point in time. This is a documented ~162x subsample, not a claim of using the
  complete dataset — stated explicitly in the notebook and `data/dataset_manifest.json` rather
  than implied.
- *Zeek log quirk (verified, not assumed)*: the trailing `tunnel_parents`/`label`/`detailed-label`
  fields are whitespace-separated from each other, not tab-separated like the other 21 standard
  Zeek `conn.log` fields — confirmed by direct byte-level inspection of the raw stream before
  writing the parser.
- *Label handling*: `label` casing normalized (`benign`→`Benign`). `detailed-label`'s `'-'`
  placeholder (meaning "no attack subtype", i.e. benign) is mapped to `Benign` *before* the
  shared `clean_frame` cleaning step, since that step's blanket `'-'`→NaN replacement would
  otherwise silently convert legitimate benign rows to a missing multiclass label.
- *Ultra-rare compound-label merge (explicit decision)*: 3 of the 10 raw `detailed-label` values
  are Stratosphere "multi-hit" compound tags (a flow matching more than one detection heuristic
  at once) with too few rows to survive a stratified split at all —
  `C&C-PartOfAHorizontalPortScan`=5, `C&C-HeartBeat-Attack`=4, `C&C-HeartBeat-FileDownload`=1.
  Merged into their base category (`C&C`, `C&C-HeartBeat`, `C&C-HeartBeat` respectively) rather
  than dropped or force-placed train-only, per explicit decision — a standard simplification in
  published IoT-23 work. Final multiclass taxonomy: 7 classes (`Benign`, `PartOfAHorizontalPortScan`,
  `Okiru`, `DDoS`, `C&C-HeartBeat`, `C&C`, `Attack`).
- *`history` field / Zeek-boolean collision (real bug caught and fixed)*: IoT-23's `history`
  field (Zeek TCP-flag-history string) can legitimately take the literal value `'F'` (3 rows in
  the sample) — colliding with the shared `clean_frame`'s `'T'`/`'F'`→`1`/`0` boolean conversion
  (correct for TON_IoT's genuine boolean columns), which silently produced a mixed int/str column
  and crashed `OneHotEncoder`. Fixed by skipping that conversion step for IoT-23 specifically
  (safe here since the dataset's only genuine boolean columns, `local_orig`/`local_resp`, are
  100% `'-'`/degenerate in this sample and are dropped entirely, not encoded).
- *SMOTE degeneracy caught and fixed (same category of issue as the NSL-KDD fix above, larger
  magnitude)*: full SMOTE-to-majority completed without OOM (IoT-23 has far fewer feature columns
  than CICIDS2017), but was rejected on inspection — it would have oversampled `Attack` from 40
  real training rows to 920,512 (a 23,013x ratio, the most extreme in this project) and nearly
  tripled the training set size (6.44M rows) versus every other dataset here. Switched to the
  same `smote_capped(cap=50,000)` compromise already used for CICIDS2017: only the 3 classes
  below the cap (`Attack`=40, `C&C`=100, `C&C-HeartBeat`=140) are raised to 50,000 (1,250x for
  `Attack`); the other 4 classes are already above the cap and untouched. Final train size:
  1,549,670 rows — comparable in scale to the other four datasets.
- *Split*: 70/15/15 stratified, `random_state=42` (no official IoT-23 split exists), same policy
  as CICIDS2017/TON_IoT.
- *Dropped columns*: `ts`/`uid`/`id.orig_h`/`id.orig_p`/`id.resp_h`/`id.resp_p` (per-flow
  identifiers, same non-generalizability rationale as TON_IoT's dropped columns);
  `tunnel_parents` (94.2% `'-'`, real values are opaque per-tunnel identifier strings, not a
  fixed vocabulary); `local_orig`/`local_resp` (100% `'-'`, degenerate); `scenario` (added by the
  sampling script itself to record which of the 23 capture files a row came from — dropped to
  avoid the model learning a shortcut association between capture-file identity and label rather
  than genuine traffic features).

**Mirror verification requirement:** every dataset's row/column counts and class distribution must be
checked against authoritative published documentation (the dataset's own research page, or the
original paper) before use, with an explicit PASS/MISMATCH verdict recorded in the relevant notebook.
A real mismatch was caught this way: the `mrwellsdavid/unsw-nb15` Kaggle mirror has
`UNSW_NB15_training-set.csv` (82,332 rows) and `UNSW_NB15_testing-set.csv` (175,341 rows) swapped
relative to official UNSW documentation (training=175,341, testing=82,332) — confirmed by matching
the smaller file's exact per-class distribution (Normal=37,000, Generic=18,871, Exploits=11,132, ...)
against the officially-documented testing-set counts. Notebook 01 uses the files by their *actual*
row-count/class-distribution identity, not their on-disk filename.

## 2b. Binary evaluation protocol (formalized 2026-07-26, applies to CLEIDS-Edge and all 8 baselines)

**Threshold calibration** (validation-set max-F1 search over 99 candidate cutoffs) is applied to
every binary evaluation, unconditionally, for every model on every dataset.

**FPR>0.20 auto-retry rule**: if a model's default-threshold (0.5) FPR exceeds 0.20 on a given
dataset, that specific model/dataset combination is automatically retrained with extended patience
(`patience=10` instead of 5) before final numbers are reported — a programmatic check, not a
case-by-case judgment call.

**Provenance of the 0.20 cutoff, stated honestly**: for CLEIDS-Edge's own Notebook 03 results, this
was NOT decided in advance — UNSW-NB15/TON_IoT/IoT-23 were retried after seeing all 5 datasets'
results together and noticing which looked bad (IoT-23 tripped the model's own majority-baseline
warning), not by applying a stated numeric rule as each result arrived. 0.20 is formalized here
*after the fact* because it's the threshold that cleanly reproduces that actual selection (the real
FPRs split cleanly: NSL-KDD 0.078 / CICIDS2017 0.0018 vs. UNSW-NB15 0.421 / TON_IoT 0.481 / IoT-23
0.981). For Notebook 04 (baselines) onward, this is applied *prospectively* — decided before any
model trains, checked programmatically, no exceptions — which is what makes it a real protocol
rather than a post-hoc description.

**What the retry does and doesn't fix, per real evidence (not assumed)**: retrying doesn't uniformly
help. UNSW-NB15 was genuine undertraining (patience=10 measurably improved FPR 0.42->0.35).
TON_IoT and IoT-23 were threshold-miscalibration, not undertraining — more epochs changed their
decisions not at all (TON_IoT: bit-identical at 8 vs 12 epochs; IoT-23: bit-identical at 10 vs 41
epochs) — but threshold tuning fixed TON_IoT well (0.48->0.16) and IoT-23 partially (0.98->0.71,
capped by its own low AUC=0.75, a genuine feature-separability limit, not a fixable training issue).

## 2c. IoT-23 binary — real, verified failure mode (not resolved, honestly reported)

Confirmed via the real trained model + real test data (not derived): confusion matrix
`[[TN=539, FP=27912], [FN=28, TP=271521]]` on the real 28,451-benign/271,549-attack test split
(9.48%/90.52%, matching the full sample's proportions exactly, as expected from stratified
splitting). The model predicts "Attack" for 99.8% of all test rows regardless of true label.
AUC=0.75 (vs. 0.90+ for every other dataset) indicates the underlying ranking ability itself is
weak, not just the decision threshold — plausibly because IoT-23's coarse feature set (8 numeric
flow stats + 4 categorical columns) doesn't cleanly separate `Benign` from `Okiru` (a Mirai-variant
botnet that itself performs heavy scanning, so it may resemble `PartOfAHorizontalPortScan` at this
feature granularity) — stated as a plausible hypothesis, not a proven mechanism.

## 3. Baseline models (for head-to-head comparison in Chapter 4)

1. Random Forest (classical ML floor)
2. Support Vector Machine (classical ML floor)
3. Standalone CNN
4. Standalone LSTM
5. Nazir et al. (2024) hybrid CNN-LSTM architecture — *Ain Shams Engineering Journal*, 15, 102777
6. Altaie & Hoomod (2024) hybrid lightweight CNN+LSTM (Raspberry Pi-targeted) — *Eng. Technol.
   Appl. Sci. Res.*, 14, 16740–16743
7. Wang et al. (2023) "DL-BiLSTM" — IPCA + dynamic quantization lightweight IDS
8. Misrak & Melaku (2025) lightweight IDS with dynamic quantization — *Discover Internet of
   Things*, 5, 97

Re-implement each baseline as faithfully as published hyperparameters allow; where a detail is
missing, use a reasonable default and note the assumption in the notebook markdown cell.

**Fidelity, checked against real sources (2026-07-26), not assumed**:
- **Altaie & Hoomod (2024)**: fully open access (ETASR) — read directly from the source PDF.
  Architecture and hyperparameters (Table I: epochs=30, batch_size=32, lr=0.001, dropout=0.3) in
  `src/models.py::build_altaie_hoomod2024` are genuinely verified, not assumed.
- **Wang et al. (2023) "DL-BiLSTM"**: open access (PeerJ), but the paper itself tunes hyperparameters
  per-dataset via Optuna rather than publishing fixed values — `build_wang2023_dlbilstm`'s exact
  layer sizes are a documented reasonable default consistent with its stated DNN+BiLSTM topology.
- **Nazir et al. (2024)** and **Misrak & Melaku (2025)**: paywalled (ScienceDirect, Springer) — real
  full-text access attempts failed (403 / auth redirect). Only abstract-level detail is verifiable;
  `build_nazir2024_hybrid`/`build_misrak_melaku2025` use documented reasonable defaults, not sourced
  architectures. Misrak & Melaku's "DNN-BiLSTMQ" naming indicates it extends Wang (2023)'s design, so
  the same topology is reused for it (no independently-verified difference could be confirmed).

## 4. Evaluation metrics

- Detection: Accuracy, Precision, Recall, F1-score, False Positive Rate (FPR), AUC-ROC
- Efficiency: Inference latency (ms/sample, CPU-only), throughput (samples/sec), model size (MB),
  peak memory footprint
- Robustness: performance drop after quantization/pruning (accuracy delta vs compression ratio)

**Figures (Notebook 03, per training run — dataset x binary/multiclass), all 300 DPI:**
1. Confusion matrix — raw counts AND row-normalized (%), side by side (normalized is what makes
   an imbalanced-class matrix actually readable).
2. Training curves — loss and accuracy/F1 per epoch, train vs val.
3. ROC curve — binary: single curve + AUC. Multiclass: one-vs-rest per class + macro-average.
4. Precision-Recall curve — same one-vs-rest treatment; more informative than ROC on these
   imbalanced datasets (a rare-attack ROC can look deceptively good while precision collapses).
5. Per-class precision/recall/F1 bar chart (multiclass only) — visual companion to the
   classification-report table.
6. t-SNE and PCA projection of the trained model's penultimate-layer embeddings on the test set,
   colored by true class — side by side, one figure per run.

**Figure (Notebook 03, once across all datasets, own results only):** cross-dataset summary bar
chart — accuracy/F1/AUC grouped by dataset and binary vs multiclass, built from `main_results.json`.

Baseline-vs-CLEIDS-Edge comparison charts and latency/model-size plots stay in Notebooks 04 and 06
respectively, where that scope already lives (see §6) — not duplicated in Notebook 03.

## 5. Contribution mapping (what each result must support)

| Contribution | Evidence required |
|---|---|
| 1. Compressed hybrid CNN-LSTM meets edge latency/memory budget with minimal accuracy loss | Table: accuracy vs model size vs latency, pre/post compression |
| 2. First empirical evaluation grounded in a TVET-institution device/threat profile | Table: baseline comparison on all 5 datasets |
| 3. Reproducible benchmark against recent published lightweight/hybrid IDS models | Table: CLEIDS-Edge vs baselines 1–8, all metrics |

## 6. Repository structure

```
repo-root/
  notebooks/
    00_Setup_and_Data.ipynb
    01_Preprocessing_and_Feature_Engineering.ipynb
    02_CLEIDS_Edge_Architecture.ipynb
    03_Training_CLEIDS_Edge.ipynb
    04_Baselines.ipynb
    05_Quantization_and_Pruning.ipynb
    06_Edge_Latency_Benchmark_CPU.ipynb
    07_Final_Results_and_Figures.ipynb
  data/            (gitignored — raw datasets, restored from Drive each session)
  models/          (saved model checkpoints, gitignored if large — else Git LFS)
  results/
    main_results.json
    baseline_results.json
    compression_results.json
    all_paper_numbers.json
  figures/         (all PNGs at 300 DPI, referenced in thesis Chapter 4)
```

## 7. Rules every notebook must follow

1. Cell 1: GitHub clone/pull + auth (token from Colab secrets).
2. Cell 2: Google Drive mount + restore any required files from Drive.
3. Cell 3: Package installation + explicit hardware check (`print(tf.config.list_physical_devices())`
   for GPU notebooks; force CPU for Notebook 06 via `tf.config.set_visible_devices([], 'GPU')`).
4. Final cells: save all outputs to Drive AND push to GitHub.
5. Last cell: print a clean summary of every metric produced, so numbers can be copied straight
   into the thesis without re-deriving them.
6. No invented numbers, ever — if a cell fails, report the failure, don't fabricate output.

## 8. Results Gate

Chapters 4, 5, and the Abstract of the thesis will NOT be written until all of the following are
confirmed complete and their outputs shared back:

- [x] Notebook 00 — repo/data setup confirmed
- [x] Notebook 01 — preprocessing complete, feature set finalized (5 datasets, including IoT-23's
      reservoir-sampled subset)
- [x] Notebook 02 — CLEIDS-Edge architecture defined and sanity-checked
- [x] Notebook 03 — CLEIDS-Edge trained on all 5 datasets (binary + multiclass), `main_results.json`
      and `tuned_threshold_results.json` saved, all figures/checkpoints pushed (2026-07-25)
- [ ] Notebook 04 — all 8 baselines trained, `baseline_results.json` saved
- [ ] Notebook 05 — quantization/pruning applied, `compression_results.json` saved
- [ ] Notebook 06 — CPU-only latency/throughput benchmark complete
- [ ] Notebook 07 — `all_paper_numbers.json` printed, all figures exported to `figures/`
