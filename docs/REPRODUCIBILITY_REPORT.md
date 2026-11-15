# Test and reproducibility report (final audit, 2026-10-25)

Everything below was **run during this audit** unless it says "previously saved". Environment: Windows 11, Python 3.11.17, PyTorch 2.14.1 (CPU), snnTorch 1.0.0, MediaPipe 1.1.1, 12th-generation Intel Core i7.

## Automated tests
| check | command | result |
|---|---|---|
| Full test suite, working project | `python -m pytest -q` | **98 passed, 1 skipped** in 171 s (the skip is the launcher test that builds a throw-away environment and needs `CARDION_SLOW_TESTS=1`) |
| Application self-test through the real launcher | `Run CARDION Self-Test.bat` | **18/18 PASS** (models load, UI served, demo estimate 73.8 vs known 72.0 BPM, video analysis 84.4 vs 84.0 BPM, recording/export, camera paths with a recorded face, positioning guidance, validation-mode session, launcher files) |
| Clean-environment run of the *public* tree | new virtual environment from `requirements-app.txt` + pytest; the staged repository copied to a fresh folder; `python -m pytest -q` | **82 passed, 17 skipped, 0 failed** (a second run after the model had been fetched: 83 passed, 16 skipped). Skips: 13 tests need a recorded face video, 1 needs the MCD-rPPG data, 1 needs the recorded clips, 1 builds a venv (slow), and (before the model is fetched) 1 doctor test. One launcher test originally failed in this environment because the MediaPipe model is deliberately not redistributed; the test now skips until the model has been downloaded. |
| Model download from a clean checkout | `ensure_model()` in the clean environment | downloaded `face_landmarker.task` (3,758,596 bytes); SHA-256 identical to the copy used in all experiments |
| Imports / model loading | covered by the tests above and by `scripts/doctor.py` in the project | pass |

## Statistics, claims and manuscript
| check | result |
|---|---|
| Claim registry (`python scripts/check_claims.py`), project | 191 claims used in the paper, 191 resolve to an existing artifact, 186 values found verbatim in their JSON artifact (one equals a stored count/mean; four point to non-JSON files) |
| Claim registry, public tree in the clean environment | same 191/191; **0** claims point to missing artifacts (467 stale development claims were dropped from the release copy) |
| Independent recomputation (`python scripts/paper_analysis.py`) from stored window-level outputs | subject MAEs, bootstrap CIs, Wilcoxon and Holm p-values and energy proxies of the validation and test tables reproduced: p-values exact; MAE max difference 0.003 BPM, CI bounds 0.005 BPM (stored rounding); energy proxy exact; frozen gate rule re-implemented independently: 486 of 1198 windows shown, MAE 2.372 BPM |
| Window-count reconciliation (`scripts/reconcile_windows.py`) | exact: validation 1202 -> 1239 (+37, recording `8772_before` at 18.4 fps), test 1197 -> 1198 (+1, recording `5223_before`); Table II now uses the evaluation definition |
| Test-split history (`scripts/test_history_audit.py`) | frozen files byte-identical to the freeze commit; window-level estimates identical in the two test runs (696 method-clip pairs); see `docs/TEST_SPLIT_HISTORY.md` |
| Bibliography (`scripts/verify_refs.py`) | 42 of 44 entries confirmed automatically against Crossref/arXiv; 2 confirmed manually (`docs/REFERENCES_VERIFIED.md`) |
| Manuscript build (`scripts/build_paper.ps1`) | 11 pages, no undefined references or errors, 0 overfull boxes; PDF metadata author "Jacob Mullen" |
| arXiv package | compiled from a clean directory (3 pdfLaTeX passes with the supplied `.bbl`): 11 pages, text **identical** to the project PDF; all file names meet arXiv's character rules. Compiled with MiKTeX, not arXiv's TeX Live - preview on arXiv before submitting. |
| Release archives (`scripts/verify_release.py`, run on the finished zips) | no byte-order marks; `.zenodo.json` strict JSON; `CITATION.cff` validates against the CFF 1.2.0 schema (cffconvert); author Jacob Mullen with no affiliation; versions agree; README equals the release template; no scratch/private/third-party files; Python files compile; Markdown links resolve; arXiv archive: root layout, file names, every figure/table present, `main.tex` identical to the repository copy, compiles with no undefined references and the same text as `paper/main.pdf` |
| Public-tree commands in a fresh environment (`pip install -r requirements.txt` from the extracted zip) | `pytest` 82 passed / 17 skipped; `check_claims.py` 191/191; `paper_analysis.py` runs; `reconcile_windows.py` and `test_history_audit.py` exit with an explanation when the data or development history is absent; `build_paper.py` regenerates `main.tex` and reports that no LaTeX is installed |
| Release scan | no local paths, user names, e-mail addresses, secret-like strings or files over 25 MB in the staged public tree; no broken relative Markdown links |

## Previously saved results versus re-run work
* **Previously saved, not re-run now:** the experiments themselves (data download, face tracking, emulation, training of 18+ networks, validation and test evaluation, diagnostics, noise and compression studies) - their outputs are the files in `out/`. A full re-run takes about 8-12 h and a 7 GB download; `REPRODUCE.md` lists the steps.
* **New in this audit:** all recomputations, reconciliations, history checks, bibliography checks and the test/packaging runs in the tables above.

## Not done / not verified
* No clean-room build of the Windows environment through `Launch CARDION.bat` in this audit (it downloads about 1.5 GB; done in earlier sessions); the launcher files are unchanged since.
* No re-run of the training or of the full evaluation from raw video; no run on macOS or Linux.
* No real-camera test with a person present; no comparison with a reference monitor (validation mode exists but contains no data).
* GitHub and Zenodo flows (repository creation, release, DOI minting, `.zenodo.json` validation) were not exercised; nothing was published.
* arXiv's own TeX Live compilation was not available.
