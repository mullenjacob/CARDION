> Note: this spec dates from the original plan, in which UBFC-rPPG was the primary dataset. The final study uses **MCD-rPPG** (36/12/12 subject-disjoint split; see `data/README.md` and the paper); UBFC-rPPG and PURE loaders remain but were not used for any reported result.

# Spec 03 — datasets (`src/datasets.py`, `src/record.py`)

- `Clip(name, subject, fs, ppg(T), rgb(T,3), face(T,S,S,3)|None, face_mask|None, meta)`.
- Loaders: `load_ubfc_clip(subject, cfg)` (DATASET_2: `subjectN/vid.avi` + `ground_truth.txt`: line1 PPG, line2 HR, line3 time), `load_pure_clip`, `load_scamps_clip` (**PURE/SCAMPS loaders not yet exercised on real data**), `make_synthetic_clip(id, cfg)` (known pulse, shading, texture, head motion, flicker, sensor noise; skin-tone dependent pulse amplitude), `load_clip("ubfc:subject1" | "synthetic:3" | ...)`.
- Splits are by **subject**: `subject_split(subjects, fracs, seed)`; `build_manifest(dataset, cfg)` records exact IDs, sha256 + size of files, and missing files -> `data/splits/<dataset>_seed<seed>.json`.
- CLI: `python -m src.datasets manifest --dataset ubfc|synthetic|pure`.
- `src/record.py`: records author clips (protocol in its docstring) to `data/raw/author/<subject>_<stamp>/{vid.avi,timestamps.csv,meta.json}`; `--source synthetic` dry-runs without a camera.
