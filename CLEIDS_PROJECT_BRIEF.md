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

**Mirror verification requirement:** every dataset's row/column counts and class distribution must be
checked against authoritative published documentation (the dataset's own research page, or the
original paper) before use, with an explicit PASS/MISMATCH verdict recorded in the relevant notebook.
A real mismatch was caught this way: the `mrwellsdavid/unsw-nb15` Kaggle mirror has
`UNSW_NB15_training-set.csv` (82,332 rows) and `UNSW_NB15_testing-set.csv` (175,341 rows) swapped
relative to official UNSW documentation (training=175,341, testing=82,332) — confirmed by matching
the smaller file's exact per-class distribution (Normal=37,000, Generic=18,871, Exploits=11,132, ...)
against the officially-documented testing-set counts. Notebook 01 uses the files by their *actual*
row-count/class-distribution identity, not their on-disk filename.

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

## 4. Evaluation metrics

- Detection: Accuracy, Precision, Recall, F1-score, False Positive Rate (FPR), AUC-ROC
- Efficiency: Inference latency (ms/sample, CPU-only), throughput (samples/sec), model size (MB),
  peak memory footprint
- Robustness: performance drop after quantization/pruning (accuracy delta vs compression ratio)

## 5. Contribution mapping (what each result must support)

| Contribution | Evidence required |
|---|---|
| 1. Compressed hybrid CNN-LSTM meets edge latency/memory budget with minimal accuracy loss | Table: accuracy vs model size vs latency, pre/post compression |
| 2. First empirical evaluation grounded in a TVET-institution device/threat profile | Table: baseline comparison on all 4 datasets |
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

- [ ] Notebook 00 — repo/data setup confirmed
- [ ] Notebook 01 — preprocessing complete, feature set finalized
- [ ] Notebook 02 — CLEIDS-Edge architecture defined and sanity-checked
- [ ] Notebook 03 — CLEIDS-Edge trained, `main_results.json` saved
- [ ] Notebook 04 — all 8 baselines trained, `baseline_results.json` saved
- [ ] Notebook 05 — quantization/pruning applied, `compression_results.json` saved
- [ ] Notebook 06 — CPU-only latency/throughput benchmark complete
- [ ] Notebook 07 — `all_paper_numbers.json` printed, all figures exported to `figures/`
