# Model card: CARDION SNN and ANN checkpoints

| | |
|---|---|
| Files | `snn.pt`, `ann.pt`, `models.json` |
| Author | Jacob Mullen (code and training); trained on data derived from MCD-rPPG (CC BY 4.0 according to the dataset pages; see `LICENSING.md`) |
| Architecture | Conv 5x5/2 (8 ch) - N - Conv 3x3/2 (16 ch) - N - AvgPool 4x4 - FC 256-64 - N - FC 64-1 - leaky integrator; N = leaky integrate-and-fire neuron (beta 0.9, fast-sigmoid surrogate gradient) in the SNN, ReLU in the ANN; 18,289 parameters each |
| Input | signed event frames, 3 colour channels x 32 x 32, 33.3 ms bins, emulated from face video at contrast threshold 0.05 (these checkpoints), soft-clipped with tanh(x/4) |
| Output | a pulse waveform; heart rate is picked from 10 s windows (sub-harmonic summation) |
| Training data | 36 subjects of MCD-rPPG (72 clips); epoch selected on 12 validation subjects; the 12 test subjects were not used |
| Intended use | research on emulated-event heart-rate estimation; shown *for comparison* in the CARDION application. They do **not** determine the number the application displays. |
| Not intended for | any medical or safety use; physical event cameras (never tested); other populations or lighting without validation |
| Performance | The paper evaluates three-seed *families*, not these single files. For threshold 0.05 the held-out test subject-level MAE is 12.8 BPM for the ANN family and 22.7 BPM for the SNN family (SNN best validation threshold 0.1: 18.9 BPM), versus 5.8 BPM for RGB POS. The exported checkpoints are the seed-1 threshold-0.05 checkpoints with the lowest validation error (`models.json`: SNN 15.0, ANN 10.9 BPM on validation); their individual test errors are not separately reported. |
| Limitations | one dataset, 36 training subjects, 24 short epochs, 3 seeds; compressed MPEG-4 video; no skin-type, motion or low-light evaluation; seed-to-seed variability is large for the SNN |
