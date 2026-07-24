PASTE EVERYTHING BELOW THIS LINE INTO CLAUDE CODE (Antigravity)
================================================================

I'm building `notebooks/00_Setup_and_Data.ipynb` for the CLEIDS-Edge project (an MPhil thesis
experiment pipeline). Read `CLEIDS_PROJECT_BRIEF.md` in this repo root first for full context
(system overview, dataset list, repo structure, notebook rules). If that file isn't in the repo
yet, create it at the repo root using the content I'm about to give you, then proceed.

Connect to a Google Colab runtime for this session (GPU not required for this notebook — CPU
runtime is fine for setup/data acquisition). Use whichever Colab connection method is configured
(MCP server or `colab` CLI).

Build `00_Setup_and_Data.ipynb` to do the following, in order:

1. **Repo setup**
   - Clone/pull the GitHub repo `https://github.com/NehlTech/CLEIDS-Edge.git` (auth token from
     Colab secrets, key name `GITHUB_TOKEN`).
   - Create the folder structure exactly as specified in `CLEIDS_PROJECT_BRIEF.md` §6, if not
     already present: `notebooks/`, `data/`, `models/`, `results/`, `figures/`.
   - Add a `.gitignore` covering `data/` and any large model checkpoint formats.

2. **Google Drive mount**
   - Mount Drive, create a `CLEIDS_Edge/` folder there if it doesn't exist, and use it as the
     persistence layer for raw datasets and checkpoints across sessions (Colab sessions don't
     persist local disk).

3. **Environment check**
   - Print Python version, TensorFlow version, and confirm GPU visibility with
     `tf.config.list_physical_devices('GPU')` — for this setup notebook we expect this to run
     fine on CPU, so don't fail if no GPU is attached.
   - `pip install` (into the Colab environment, not system-wide) any of: tensorflow, scikit-learn,
     pandas, numpy, imbalanced-learn (for SMOTE), matplotlib, seaborn — only what's missing.

4. **Dataset acquisition**
   - For each of the four datasets (NSL-KDD, CICIDS2017, IoT-23, UNSW-NB15/TON_IoT), write a
     acquisition cell that downloads from the **official** source (not third-party mirrors):
     - NSL-KDD: University of New Brunswick / Canadian Institute for Cybersecurity page
     - CICIDS2017: Canadian Institute for Cybersecurity page
     - IoT-23: Stratosphere Laboratory (CTU) page
     - UNSW-NB15 / TON_IoT: UNSW Canberra Cyber Range page
   - If a dataset requires manual download (e.g. a form/agreement gate), don't try to scrape
     around it — instead, print clear instructions for me to download it manually and place it
     in a specific `data/raw/<dataset_name>/` path, then have the notebook check for that path
     and proceed once present.
   - Save all raw files to the Drive folder (`CLEIDS_Edge/data/raw/`), and symlink/copy into the
     local `data/raw/` at the start of every future session.

5. **Sanity checks**
   - For each dataset once downloaded, print: number of rows, number of columns, label/class
     distribution (attack vs benign, and per-attack-type if multi-class), and flag any obvious
     issues (huge class imbalance, missing values, duplicate rows) — just report these, don't fix
     them yet (that's Notebook 01).

6. **Final cells**
   - Save a small `data/dataset_manifest.json` summarizing what was downloaded, file paths, row
     counts, and class distributions.
   - Push the notebook and manifest to GitHub.
   - Print a clean final summary block listing: which datasets are ready, which need manual
     download, and the exact next command/action needed from me if anything is blocked.

Constraints:
- No fabricated or placeholder data — if a dataset can't be acquired automatically, say so
  explicitly rather than substituting a synthetic stand-in.
- Keep all file paths consistent with `CLEIDS_PROJECT_BRIEF.md` §6 exactly, since later notebooks
  will assume this structure.
- At the end, give me the exact printed summary output so I can pass it back for review before
  we move to Notebook 01.
