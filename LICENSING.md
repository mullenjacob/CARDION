# Licensing

Different parts of this repository are under different licences. Third-party material keeps its own licence and is **not** relicensed here.

| material | where | licence | notes |
|---|---|---|---|
| Original source code: research code, application, scripts, tests, launchers, configuration | `src/`, `scripts/`, `tests/`, `configs/`, `*.bat`, `requirements*.txt` | **MIT** (`LICENSE`) | Copyright (c) 2026 Jacob Mullen |
| Manuscript, figures and tables made by the author, documentation text | `paper/`, `docs/`, `specs/`, `README.md`, `REPRODUCE.md` | **CC BY 4.0** (`LICENSES/CC-BY-4.0.txt`) | Attribute "Jacob Mullen". Matches the licence recommended for the arXiv submission (the arXiv licence is chosen separately and irrevocably at submission). |
| Saved experimental outputs, split manifests, trained checkpoints | `out/`, `data/splits/`, `models/*.pt`, `models/models.json` | **CC BY 4.0**, with the attribution below | They are derived from the CC BY 4.0 dataset MCD-rPPG (per-window heart-rate estimates and reference heart rates computed from it; weights trained on it). CC BY 4.0 permits this if attribution and a statement that the material was adapted are kept. Use of the weights is additionally subject to the terms of the training data. |
| Software dependencies (PyTorch, snnTorch, MediaPipe, OpenCV, NumPy, SciPy, Matplotlib, ...) | installed from PyPI by the launcher | their own licences | not redistributed |

## Not included, with the reason

| material | status |
|---|---|
| MCD-rPPG videos / PPG | not redistributed. CC BY 4.0 (licence tag of the official page `kyegorov/mcd_rppg`, read through the Hugging Face API on 2026-10-25; the tag may change, so re-check before relying on it). Cite Egorov et al., ACM MM 2025. Videos show real people's faces; do not redistribute them or attempt re-identification. |
| SCAMPS example videos | not redistributed (see upstream terms). |
| `face_landmarker.task` (MediaPipe) | not redistributed: the licence terms of the bundled model files could not be confirmed from official sources. It is downloaded by the application from Google's model URL on first use. |
| rPPG-Toolbox code | not redistributed. Upstream uses the *Responsible AI Source Code License v1.1* (use restrictions that must flow down to redistributors). `scripts/fetch_third_party.py` downloads a pinned subset on request for the correctness-anchor check only. |
| v2e | MIT upstream; not redistributed (downloaded at a pinned commit on request). |
| Application screenshots, development logs, recordings, sessions, validation data | not included. |

## Attribution text for derived material

> Contains results, split manifests and model weights derived from the MCD-rPPG dataset (K. Egorov et al., "Gaze into the Heart: A Multi-View Video Dataset for rPPG and Health Biomarkers Estimation", ACM Multimedia 2025, doi:10.1145/3746027.3758255; CC BY 4.0 as stated on the dataset page). The material was adapted: reference heart rates were computed from the PPG signal, and networks were trained on emulated events. Changes were made by Jacob Mullen.

## Licence choices left to the author

The defaults above are recommendations. The author decides (a) whether the derived outputs and weights should be CC BY 4.0 (default) or MIT, (b) whether the paper on arXiv should be CC BY 4.0 (default) or another arXiv-offered option, and (c) whether documentation should be CC BY 4.0 (default) or MIT. Nothing here is legal advice.
