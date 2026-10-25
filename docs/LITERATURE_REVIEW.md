# Literature review and positioning (verified 2026-10-25)

Everything below was checked against primary pages (arXiv abstracts/HTML, Crossref records, publisher or CVF pages); numbers are copied from them, are recorded in
[`paper/literature.json`](../paper/literature.json) and registered as `lit.*` claims. Reference metadata and links: [`REFERENCES_VERIFIED.md`](REFERENCES_VERIFIED.md).
Preprints are marked and are not peer reviewed. **Published numbers use different datasets, subjects, cameras, windows, references and exclusion rules, so none of the comparisons
below is a ranking.**

## How the search was done, and its limits

* Tools available in this session: a general web-search engine (results pointing to arXiv, Crossref, CVF Open Access, PubMed Central, ScienceDirect, Springer/Nature, IEEE and ACM
  landing pages), page fetches of arXiv/Crossref/CVF/Zenodo pages, and the Crossref and arXiv APIs.
* **Not queried directly:** Google Scholar, IEEE Xplore, ACM Digital Library and PubMed search interfaces. Some paywalled papers (Nature npj Biosensing, HAL pages blocked by bot
  protection) were seen at abstract/snippet level only and no result is attributed to them beyond what was read.
* Queries covered: rPPG (classical, deep, reviews, datasets), event cameras and physiology, SNNs in rPPG, simulated events (ESIM, v2e, video-to-events, sim-to-real), compression,
  signal-quality/availability, skin type, wearables as references, SNN-versus-ANN energy.
* "No prior work found" below means *not found in these searches*; absence in a web search is weak evidence.

## Closest prior work

| Work | Status | Sensor / data | Method | Reported result (own protocol) | Limits stated or visible | Relation to this paper |
|---|---|---|---|---|---|---|
| Jagtap, Saripalli, Lemley, Shariff, Smeaton 2023, *Heart Rate Detection Using an Event Camera* ([arXiv:2309.11891](https://arxiv.org/abs/2309.11891)) | preprint | real event camera (Prophesee EVK4), wrist; 25 participants, 46 recordings | unsupervised: per-tile periodogram of events (5x5 tiles, 1/50 s bins), fused | MAE 1.478 (rest), 1.706 BPM (elevated); pulse found in 40 of 46 recordings | flicker, tremor (3 of 6 failures), lighting; watch reference delay | real events, wrist, unsupervised, small; ours is face, simulated, learned + classical |
| Moustafa, Lemley, Corcoran 2025, *Contactless Cardiac Pulse Monitoring Using Event Cameras* ([arXiv:2505.09529](https://arxiv.org/abs/2505.09529)) | preprint | real event camera + RGB (Brio 30 fps); **proprietary**, 64 seated subjects (10 for validation/test) | supervised TS-CAN-style CNN on 2-D event frames | MAE 2.33 (events 30 fps) vs 1.89 (RGB 30 fps); 1.58 (events 120 fps); RMSE 3.32 / 2.92 / 2.13 | proprietary data; stationary subjects; one representation; no de-noising | closest in question; real events, supervised. Direction at 30 fps (events slightly worse than RGB) matches ours but settings differ completely |
| Feng et al. 2026, *EMPD* ([arXiv:2603.26699](https://arxiv.org/abs/2603.26699); Zenodo doi:10.5281/zenodo.18765701, CC BY 4.0) | preprint | real event camera (EVK4) + RGB + oximeter; **660 nm laser at the radial artery** (wrist); 83 subjects, 193 records, 40-110 BPM | dataset paper | none on the abstract page | 9 archive parts, first part 46.4 GB | public real-event data exist now, but wrist/laser, not faces; a sim-to-real test on it would answer a different question |
| Feng et al. 2026, *Fusion-E2Pulse* ([arXiv:2606.15597](https://arxiv.org/abs/2606.15597)) | preprint, listed as accepted to MICCAI 2026 | event + RGB (dataset not named on the abstract page) | event-RGB fusion network for waveform reconstruction | HR MAE 0.78 BPM, waveform correlation 0.89 | not assessed (abstract only) | fusion rather than simulation; supports that events carry waveform detail |
| Liu et al. 2025, *Spiking-PhysFormer* ([doi:10.1016/j.neunet.2025.107128](https://doi.org/10.1016/j.neunet.2025.107128)) | peer reviewed | RGB frames; PURE, UBFC-rPPG, UBFC-Phys, MMPD | hybrid ANN patch embedding + spiking transformer + ANN head | "comparable" to PhysFormer; total modelled power -10.1 % in the published version (-12.4 % in the arXiv preprint), spiking transformer block 12.2x lower in both | how power was computed was not visible on the pages read; the 12.2x applies to the transformer block only | SNNs in rPPG exist (on frames). Ours: SNN on emulated events, matched ANN control, measured event density |
| Liu et al. 2023, *rPPG-Toolbox* ([arXiv:2210.00716](https://arxiv.org/abs/2210.00716)) | preprint of NeurIPS D&B paper | UBFC-rPPG, PURE | unsupervised and supervised baselines | POS MAE 4.08 (UBFC) / 3.67 (PURE); CHROM 4.06 / 5.77; green 19.73 / 10.09 | video-level HR, different protocol | reference implementations; our POS/CHROM match them on identical traces |
| McDuff, Blackford, Estepp 2017 ([doi:10.1109/FG.2017.17](https://doi.org/10.1109/FG.2017.17)) | peer reviewed | 25 participants | x264/x265 at several quality levels | systematic effect of compression on iPPG (effect sizes not read) | - | compression harms rPPG; supports our diagnostic hypothesis |
| Rapczynski, Werner, Al-Hamadi 2019 ([doi:10.1109/TBME.2019.2904326](https://doi.org/10.1109/TBME.2019.2904326)); Yu et al. 2019 ([doi:10.1109/ICCV.2019.00024](https://doi.org/10.1109/ICCV.2019.00024)) | peer reviewed | - | encoding effects; enhancement network for highly compressed video | compression degrades rPPG (details not read) | - | same |
| Gehrig et al. 2020 ([doi:10.1109/CVPR42600.2020.00364](https://doi.org/10.1109/CVPR42600.2020.00364)); Hu et al. 2021 v2e ([doi:10.1109/CVPRW53098.2021.00144](https://doi.org/10.1109/CVPRW53098.2021.00144)); Rebecq et al. 2018 ESIM; Stoffregen et al. 2020 | peer reviewed (Stoffregen: ECCV 2020 per arXiv listing) | video / images | video-to-event conversion; sim-to-real gap | synthetic events transfer for recognition/segmentation; v2e adds mismatch, bandwidth, noise | tasks are not physiological | ours tests whether a sub-1 % signal survives emulation |
| Dampfhoffer et al. 2023 ([doi:10.1109/TETCI.2022.3214509](https://doi.org/10.1109/TETCI.2022.3214509)); Lemaire et al. 2023 ([doi:10.1007/978-3-031-30105-6_48](https://doi.org/10.1007/978-3-031-30105-6_48)) | peer reviewed | - | hardware-aware and analytical SNN energy models | SNNs compete only at very high sparsity (0.15-1.38 spikes/synapse/inference) | - | our operation-count proxy is the hardware-agnostic kind; our streams are dense |
| Nowara, McDuff, Veeraraghavan 2020 ([doi:10.1109/CVPRW50498.2020.00150](https://doi.org/10.1109/CVPRW50498.2020.00150)) | peer reviewed | 73 people, >400 videos (meta-analysis) | - | average accuracy drops significantly for the darkest skin type | - | we have no skin-type annotation; stated as a limitation |
| Elgendi et al. 2024 ([doi:10.1038/s44328-024-00002-1](https://doi.org/10.1038/s44328-024-00002-1)) | peer reviewed | public rPPG datasets | single SNR-based quality index with a recommended threshold | not read (paywalled) | - | quality gating is established; we quantify what it does to reported accuracy |
| Shcherbina et al. 2017; Bent et al. 2020 | peer reviewed | wrist wearables | accuracy studies | Apple Watch lowest error of seven devices (60 volunteers), but error larger with darker skin tone, higher BMI and walking | - | informal watch comparisons are not ground truth |

## Is CARDION novel? A critical assessment

**Already in the literature (not claimed as new):** event cameras detect the pulse (wrist, faces, 2023-2026); SNNs have been applied to rPPG (on frames); events can be emulated from
video and used for training in other tasks; compression harms rPPG; quality indices exist for rPPG; SNN energy claims are contested; a real public event dataset (EMPD) now exists.

**Not found in these searches (candidate contributions, stated cautiously):**

1. A controlled comparison of RGB estimators, classical event estimators, a matched ANN and an SNN on events *emulated from real compressed webcam video*, under one
   subject-disjoint protocol with a protected test split, validation-only selection, three seeds, subject-level statistics and effect sizes.
2. A quantification of the **selection effect** of a reliability gate on reported accuracy, with per-subject availability, a same-window single-method baseline and an exploratory
   accuracy-availability curve (single-method gate at least as good as the consensus on validation windows).
3. An operation-count analysis using the *measured event density of real compressed video*, showing the SNN's count advantage disappears at fine thresholds.
4. A claim registry plus independent recomputation of the headline statistics.

**Strongest defensible contribution:** a rigorous, honestly framed negative-and-diagnostic result - *emulating events from compressed webcam video is not a free route to event-camera
physiological data, and an SNN is not worthwhile on such streams* - together with the methodological lesson that accuracy of a gated system must be reported with availability and
against a same-window baseline. It is **not** a new architecture, not evidence about real event hardware, and not a clinical validation. The "first" claims in the project charter
(first spiking representation, first synthetic-event pulse estimator) are *not* supported and are not made in the paper.

## Where leading work is ahead of this project

* Real event hardware with ECG reference (Moustafa et al.), laser-assisted real-event data at scale (EMPD), event-RGB fusion with sub-1 BPM MAE (Fusion-E2Pulse, abstract level).
* Supervised video models (PhysFormer, EfficientPhys, Spiking-PhysFormer) trained on large public datasets with cross-dataset testing; we train a small 18 k-parameter network on 36 subjects.
* Hardware-aware energy modelling; ours is an operation-count proxy.

## Other facts checked

* MCD-rPPG ([doi:10.1145/3746027.3758255](https://doi.org/10.1145/3746027.3758255)): 600 subjects, 3600 recordings, CC BY 4.0 (licence tag of the official page `kyegorov/mcd_rppg`, read through the Hugging Face API on 2026-10-25; the page text itself could
  not be opened).
