# Spec 06 — spiking network (`src/snn_model.py`, `src/ann_model.py`, `src/train.py`, `src/efficiency.py`)

- `PulseNetBase` (Conv5x5/2 -> N -> Conv3x3/2 -> N -> AvgPool4 -> FC -> N -> FC -> leaky-integrator readout; input (B,T,3,32,32) signed event frames; output (B,T)). `SpikingPulseNet` (snnTorch `Leaky`, fast-sigmoid surrogate) and `AnnPulseNet` (ReLU) share every layer: 18,289 parameters each.
- `train.py`: events cached in `data/events`; windows cut within *training subjects only*; loss = MSE + (1 - Pearson) vs bandpassed z-scored PPG; best-val checkpoint `out/ckpt_<kind>_*.pt`, `learning_curve_*.json`. HR from the predicted waveform uses the same 10 s sliding FFT as the baselines. Currently supports `--dataset synthetic`; SCAMPS training is the next step.
- `efficiency.measure(model, x)`: params, dense MACs/FLOPs, SOPs (spikes x fan-out, upper bound), energy proxy (4.6 pJ/MAC, 0.9 pJ/AC), latency.
