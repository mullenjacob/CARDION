"""Spiking pulse regressor (snnTorch) and the shared architecture base used by the matched ANN control.

Input ``(B, T, 3, H, W)`` signed event frames (one frame per temporal bin, one channel per colour stream).
Output ``(B, T)`` pulse waveform. Architecture::

    Conv5x5/2 -> N -> Conv3x3/2 -> N -> AvgPool(4x4) -> FC -> N -> FC -> leaky-integrator readout

where ``N`` is a LIF neuron (spiking) or ReLU (ANN control). The readout is the same non-spiking leaky
integrator in both so the two models differ *only* in the hidden nonlinearity.
"""
from __future__ import annotations

import argparse
from typing import Callable

import snntorch as snn
import torch
from snntorch import surrogate
from torch import nn


class SpikingNeuron(nn.Module):
    """Stateful LIF layer wrapper that tracks emitted spike counts for SOP accounting."""

    def __init__(self, beta: float) -> None:
        super().__init__()
        self.lif = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid(slope=25))
        self.mem: torch.Tensor | None = None
        self.spike_count = 0.0
        self.units = 0

    def reset(self) -> None:
        """Clear membrane state and counters."""
        self.mem = None
        self.spike_count, self.units = 0.0, 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mem is None:
            self.mem = self.lif.init_leaky()
        spk, self.mem = self.lif(x, self.mem)
        self.spike_count += float(spk.detach().sum())
        self.units = spk[0].numel()
        return spk


class ReluNeuron(nn.Module):
    """Stateless ReLU stand-in with the same interface (counts non-zero activations)."""

    def __init__(self) -> None:
        super().__init__()
        self.spike_count = 0.0
        self.units = 0

    def reset(self) -> None:
        """Clear counters."""
        self.spike_count, self.units = 0.0, 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.relu(x)
        self.spike_count += float((y.detach() > 0).sum())
        self.units = y[0].numel()
        return y


class PulseNetBase(nn.Module):
    """Shared architecture; subclasses choose the neuron type via :meth:`make_neuron`."""

    def __init__(self, in_ch: int = 3, c1: int = 8, c2: int = 16, hidden: int = 64, readout_beta: float = 0.8,
                 input_scale: float | None = None) -> None:
        super().__init__()
        self.cfg = dict(in_ch=in_ch, c1=c1, c2=c2, hidden=hidden, readout_beta=readout_beta, input_scale=input_scale)
        self.input_scale = input_scale          # soft-clips event counts: tanh(x / scale); None = raw counts (older checkpoints)
        self.conv1 = nn.Conv2d(in_ch, c1, 5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(c1, c2, 3, stride=2, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(4)
        self.fc1 = nn.Linear(c2 * 16, hidden)
        self.fc2 = nn.Linear(hidden, 1)
        self.readout_beta = readout_beta
        self.n1, self.n2, self.n3 = self.make_neuron(), self.make_neuron(), self.make_neuron()
        self.neurons = [self.n1, self.n2, self.n3]
        self.input_events = 0.0

    def make_neuron(self) -> nn.Module:
        """Return a fresh hidden-layer nonlinearity."""
        raise NotImplementedError

    def reset_state(self, batch: int, device: str | torch.device = "cpu") -> None:
        """Clear neuron membranes, counters and the read-out integrator (start of a sequence)."""
        for n in self.neurons:
            n.reset()
        self.input_events = 0.0
        self._out_mem = torch.zeros(batch, device=device)

    def step(self, x_t: torch.Tensor) -> torch.Tensor:
        """One time step: x_t (B,C,H,W) -> (B,) read-out sample. Usable for streaming inference after :meth:`reset_state`."""
        self.input_events += float(x_t.detach().abs().sum())
        if self.input_scale:
            x_t = torch.tanh(x_t / self.input_scale)
        h = self.n1(self.conv1(x_t))
        h = self.n2(self.conv2(h))
        h = self.n3(self.fc1(self.pool(h).flatten(1)))
        self._out_mem = self.readout_beta * self._out_mem + self.fc2(h).squeeze(1)
        return self._out_mem

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B,T,C,H,W) -> (B,T) pulse prediction."""
        self.reset_state(x.shape[0], x.device)
        return torch.stack([self.step(x[:, t]) for t in range(x.shape[1])], dim=1)


class SpikingPulseNet(PulseNetBase):
    """LIF version of the pulse regressor."""

    def __init__(self, beta: float = 0.9, **kw) -> None:
        self.beta = beta
        super().__init__(**kw)
        self.cfg["beta"] = beta

    def make_neuron(self) -> nn.Module:
        return SpikingNeuron(self.beta)


def build_model(kind: str, cfg: dict) -> PulseNetBase:
    """Factory: ``kind`` in {snn, ann}; widths follow ``cfg['snn']``."""
    from .ann_model import AnnPulseNet
    s = cfg["snn"]
    kw = dict(hidden=s["hidden"], input_scale=s.get("input_scale"))
    return SpikingPulseNet(beta=s["beta"], **kw) if kind == "snn" else AnnPulseNet(**kw)


def main() -> None:
    """CLI: print the architecture and run a forward pass on random events."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=30)
    args = ap.parse_args()
    torch.manual_seed(0)
    m = SpikingPulseNet()
    x = (torch.rand(2, args.steps, 3, 32, 32) < 0.05).float() - (torch.rand(2, args.steps, 3, 32, 32) < 0.05).float()
    y = m(x)
    print("params:", sum(p.numel() for p in m.parameters()), "output:", tuple(y.shape),
          "spikes/layer:", [int(n.spike_count) for n in m.neurons])


if __name__ == "__main__":
    main()
