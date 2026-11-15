"""Efficiency accounting: parameters, FLOPs/MACs, synaptic operations (SOPs), energy proxy, wall-clock latency.

Conventions (stated in the paper):
* ANN cost = dense MACs (one forward over the whole window); FLOPs = 2 x MACs.
* SNN cost = SOPs = sum over layers of (spikes emitted into the layer) x (fan-out per spike), an upper bound
  that ignores padding/stride truncation savings; the first layer's "spikes" are the input events.
* Energy proxy uses 45 nm figures (Horowitz 2014): 4.6 pJ per 32-bit MAC, 0.9 pJ per accumulate. It is an
  accounting proxy, not a measurement, and ignores memory traffic and neuron-state updates unless stated.
"""
from __future__ import annotations

import argparse
import json
import time

import torch

from .snn_model import PulseNetBase

E_MAC_PJ, E_AC_PJ = 4.6, 0.9


def count_params(model: torch.nn.Module) -> int:
    """Trainable parameter count."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def macs_per_step(model: PulseNetBase, h: int, w: int) -> int:
    """Dense MACs for one time step of the architecture."""
    c = model.cfg
    h1, w1 = (h + 1) // 2, (w + 1) // 2
    h2, w2 = (h1 + 1) // 2, (w1 + 1) // 2
    conv1 = h1 * w1 * c["c1"] * c["in_ch"] * 25
    conv2 = h2 * w2 * c["c2"] * c["c1"] * 9
    return conv1 + conv2 + c["c2"] * 16 * c["hidden"] + c["hidden"]


def sops_from_counts(model: PulseNetBase) -> float:
    """SOPs from spike counts recorded in the last forward pass (upper bound; see module docstring)."""
    c = model.cfg
    return (model.input_events * c["c1"] * (5 / 2) ** 2 + model.n1.spike_count * c["c2"] * (3 / 2) ** 2
            + model.n2.spike_count * c["hidden"] + model.n3.spike_count * 1)


def measure(model: PulseNetBase, x: torch.Tensor, n_repeat: int = 5) -> dict:
    """Run ``model`` on ``x`` (B,T,C,H,W) and report params, MACs/FLOPs, SOPs, energy proxy and latency."""
    model.eval()
    T, h, w = x.shape[1], x.shape[3], x.shape[4]
    with torch.no_grad():
        model(x)
        sops = sops_from_counts(model)
        times = []
        for _ in range(n_repeat):
            t0 = time.perf_counter()
            model(x)
            times.append(time.perf_counter() - t0)
    macs = macs_per_step(model, h, w) * T * x.shape[0]
    spiking = type(model).__name__.startswith("Spiking")
    acts = [n.spike_count / max(n.units * T * x.shape[0], 1) for n in model.neurons]
    out = {"model": type(model).__name__, "params": count_params(model), "steps": T, "batch": x.shape[0],
           "dense_macs": macs, "dense_flops": 2 * macs, "sops": sops if spiking else None,
           "activity_per_layer": acts, "input_events": model.input_events,
           "latency_ms_per_window": 1000 * float(sorted(times)[len(times) // 2]) / x.shape[0],
           "latency_ms_per_step": 1000 * float(sorted(times)[len(times) // 2]) / (T * x.shape[0])}
    out["energy_uj"] = (sops * E_AC_PJ if spiking else macs * E_MAC_PJ) * 1e-6
    return out


def main() -> None:
    """CLI: efficiency table for the SNN and ANN on random event tensors of a given density."""
    from .ann_model import AnnPulseNet
    from .snn_model import SpikingPulseNet
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--density", type=float, default=0.02, help="fraction of active input pixels per step")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    x = ((torch.rand(1, args.steps, 3, 32, 32) < args.density).float()
         - (torch.rand(1, args.steps, 3, 32, 32) < args.density).float())
    from .common import out_path, register_claim, save_json
    results = []
    for m in (SpikingPulseNet(), AnnPulseNet()):
        r = measure(m, x)
        results.append(r)
        print(json.dumps(r, indent=1))
    path = save_json(results, out_path({"out_dir": "out"}, "efficiency_random_input", "json"))
    cmd = f"python -m src.efficiency --steps {args.steps} --density {args.density} --seed {args.seed}"
    register_claim("model.params", results[0]["params"], path, cmd, note="SNN and ANN have identical parameter counts")
    register_claim("model.params_ann", results[1]["params"], path, cmd)


if __name__ == "__main__":
    main()
