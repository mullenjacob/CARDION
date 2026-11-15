# Critical review of the final manuscript (written as a demanding reviewer would; updated in the final audit)

Scope: `paper/main.pdf` (11 pages including appendix). Qualitative judgements only; no scores.

## Summary
Asks whether events emulated from compressed webcam video preserve the pulse for classical, ANN and spiking estimators, whether an SNN helps, and how much a reliability gate flatters reported accuracy. Findings: the emulated-event estimators built here are clearly less accurate than RGB POS; the SNN shows no accuracy advantage over a matched ANN and uses more synaptic operations at fine thresholds; the gate's accuracy comes mostly from selecting easy windows.

## Strengths
* A narrow question and an honest negative result; claims are phrased as observed accuracy differences, with explanations labelled as hypotheses and no information-theoretic claim.
* Subject-disjoint splits, validation-only selection, three seeds, subject-level statistics, paired effect sizes with CIs rather than only p-values; non-significance is not read as equivalence.
* The protected test split's history is stated plainly (frozen protocol, a first scoring with a summary-table bug, a second scoring with identical window-level estimates, later descriptive analyses) and backed by hashes and timestamps (`docs/TEST_SPLIT_HISTORY.md`).
* Independent recomputation of the headline statistics from stored window-level outputs (p-values and Holm values exact; MAE/CI within 0.005 BPM; energy proxy exact; gate coverage/MAE reproduced by a separate implementation); every number in the text is registered with its source; literature numbers too; window counts reconciled exactly (`scripts/reconcile_windows.py`).
* Availability reported with accuracy; same-window single-method baselines; per-subject coverage (3-72 %).
* Clear separation between the application demonstration and the controlled study; the displayed number is attributed to the RGB consensus, not to the SNN.

## Weaknesses and likely reviewer objections
1. **No real events.** The sim-to-real gap is untested; the central conclusion applies to *these emulated events* only. Needs paired real-event and RGB recordings of faces (new data and approvals).
2. **One dataset, 12 test subjects, compressed video.** Wide intervals; no external validation (UBFC-rPPG could not be downloaded; PURE needs a request); compression cannot be separated from emulator fidelity on MCD-rPPG.
3. **Explanations are hypotheses**, supported by simulation (compression, flicker, noise), not demonstrated on the real data.
4. **Small learned baselines** (18 k parameters, 36 training subjects, 24 short epochs, three seeds). Reviewers will say the networks had little chance against supervised video models; the paper restricts its claim accordingly.
5. **Energy numbers are a proxy** (45 nm constants, no memory traffic); the SNN-versus-ANN operation comparison depends on it and on event density.
6. **Exploratory gate analysis** (single-method gate better on validation windows) was done after the rule was frozen; it is a hypothesis, listed first among future experiments, and the deployed rule was not changed.
7. **Test-set exposure:** scored twice and re-read for descriptive analyses; documented; no decision changed, but the split is no longer strictly unseen.
8. **Reference:** finger-PPG spectral peak with a validity rule applied to all methods; the rule removes hard reference windows for every method and could interact with ranking.
9. **Statistics at n = 12:** conservative Holm over 24 comparisons; percentile bootstrap intervals approximate; window-pooled agreement statistics are descriptive.
10. **Literature search** was web-based and not systematic; some papers read at abstract level only; no direct IEEE Xplore/ACM DL/PubMed queries.
11. **AI assistance** is substantial and disclosed; venue rules may restrict AI-written text.

## Assessment by criterion
Originality: modest and defensible when stated as a controlled negative result plus an availability/selection analysis; no new architecture; no "first" claims. Scientific accuracy: audited against saved outputs; no known unsupported number. Literature: adequate for positioning. Rigor and statistics: good for the scale, limited by n = 12 and one dataset. Baselines: strong RGB baselines, small learned baselines. Reproducibility: high (code, manifests, claim registry, recomputation scripts, clean-environment test run); a full re-run of the experiments takes 8-12 h and a 7 GB download.

## Essential blockers versus optional improvements
*Essential before posting a preprint (none is a scientific blocker):* an arXiv endorsement (personal, for a first submission); the author's own verification of the content, references and AI-disclosure wording; real repository and archive links in the availability statement. (The dataset licence was confirmed on the official page on 2026-10-25.)
*Optional / future:* the experiments in the paper's final section (real-event recordings, uncompressed capture, pre-specified single-method gate confirmation, external datasets, hardware energy, larger or temporal-coding SNNs).
