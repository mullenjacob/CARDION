"""Matched ANN control: the spiking architecture with ReLU in place of LIF (identical parameter count)."""
from __future__ import annotations

import argparse

import torch
from torch import nn

from .snn_model import PulseNetBase, ReluNeuron, SpikingPulseNet


class AnnPulseNet(PulseNetBase):
    """ReLU version of the pulse regressor; same layers, widths and readout as :class:`SpikingPulseNet`."""

    def make_neuron(self) -> nn.Module:
        return ReluNeuron()


def main() -> None:
    """CLI: verify that the SNN and ANN have identical parameter counts and output shapes."""
    argparse.ArgumentParser(description=__doc__).parse_args()
    a, s = AnnPulseNet(), SpikingPulseNet()
    x = torch.randn(1, 10, 3, 32, 32)
    pa, ps = sum(p.numel() for p in a.parameters()), sum(p.numel() for p in s.parameters())
    print(f"ANN params {pa}  SNN params {ps}  match={pa == ps}  out {tuple(a(x).shape)} {tuple(s(x).shape)}")


if __name__ == "__main__":
    main()
