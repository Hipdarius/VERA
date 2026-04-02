"""1D ResNet for spectrum-based mineral classification + ilmenite regression.

Architecture overview
---------------------
Input: (B, 1, 301)  — concatenated [spectrum(288) | LED(12) | LIF(1)]

  Stem       : conv(1→32, k=9)          + BN + ReLU + MaxPool(2)
  Stage 1    : 2 × BasicBlock1D(32→32)
  Stage 2    : 2 × BasicBlock1D(32→64,   stride=2)
  Stage 3    : 2 × BasicBlock1D(64→128,  stride=2)
  Stage 4    : 2 × BasicBlock1D(128→192, stride=2)
  Global avg pool -> dropout -> two heads (classification, sigmoid reg)

Target parameter count: ~670k — comfortably in the 500k–1M envelope
requested for the scaled-up brain. The architecture is deterministic
under :func:`regoscan.train.set_global_seed` (BatchNorm + Conv1d only;
no RNG-dependent ops in the forward pass).

The class name ``RegoscanCNN`` is preserved on purpose so older run
directories and tests keep working — what changes is the inside.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from regoscan.schema import N_CLASSES, N_FEATURES_TOTAL


def _set_torch_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class BasicBlock1D(nn.Module):
    """Classic two-layer residual block for 1D signals."""

    def __init__(self, c_in: int, c_out: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(
            c_in, c_out, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm1d(c_out)
        self.conv2 = nn.Conv1d(
            c_out, c_out, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm1d(c_out)
        if stride != 1 or c_in != c_out:
            self.downsample: nn.Module = nn.Sequential(
                nn.Conv1d(c_in, c_out, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(c_out),
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = out + identity
        return F.relu(out, inplace=True)


class RegoscanCNN(nn.Module):
    """1D ResNet with two heads.

    Forward signature (unchanged from the previous tiny CNN):
        x: (B, 1, 301) -> (logits (B, 5), ilm (B,))
    """

    def __init__(
        self,
        n_classes: int = N_CLASSES,
        *,
        channels: tuple[int, int, int, int] = (32, 64, 128, 192),
        blocks_per_stage: int = 2,
        dropout: float = 0.25,
        seed: int = 0,
    ) -> None:
        super().__init__()
        _set_torch_seed(seed)

        c1, c2, c3, c4 = channels
        self.stem = nn.Sequential(
            nn.Conv1d(1, c1, kernel_size=9, stride=1, padding=4, bias=False),
            nn.BatchNorm1d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
        )

        self.stage1 = self._make_stage(c1, c1, blocks_per_stage, stride=1)
        self.stage2 = self._make_stage(c1, c2, blocks_per_stage, stride=2)
        self.stage3 = self._make_stage(c2, c3, blocks_per_stage, stride=2)
        self.stage4 = self._make_stage(c3, c4, blocks_per_stage, stride=2)

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.head_cls = nn.Linear(c4, n_classes)
        self.head_reg = nn.Linear(c4, 1)

        # Kaiming init on conv layers for stable deep residual training.
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    @staticmethod
    def _make_stage(c_in: int, c_out: int, n_blocks: int, stride: int) -> nn.Sequential:
        layers: list[nn.Module] = [BasicBlock1D(c_in, c_out, stride=stride)]
        for _ in range(n_blocks - 1):
            layers.append(BasicBlock1D(c_out, c_out, stride=1))
        return nn.Sequential(*layers)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.global_pool(x).flatten(1)
        return self.dropout(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.features(x)
        logits = self.head_cls(h)
        ilm = torch.sigmoid(self.head_reg(h)).squeeze(-1)
        return logits, ilm


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def assert_input_size():
    """Sanity-check the architecture against the canonical feature vector."""
    assert N_FEATURES_TOTAL == 301, (
        f"CNN architecture assumes 301-feature input, got {N_FEATURES_TOTAL}"
    )


__all__ = ["RegoscanCNN", "BasicBlock1D", "count_params", "assert_input_size"]
