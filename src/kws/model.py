"""log-mel スペクトログラムを入力にとる小型 2D-CNN（AudioCNN）.

特徴量変換（log-mel）は data.py 側に置いたので，このモデルは
「(B, 1, n_mels, T') の log-mel を受け取って 35 クラスの logits を返す」
純粋な分類器に専念する．
"""

from __future__ import annotations

from torch import nn


class ConvBlock(nn.Module):
    """Conv2d(3x3) → BatchNorm → ReLU → MaxPool(2) の 1 ブロック."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.block(x)


class AudioCNN(nn.Module):
    """(B, 1, n_mels, T') → logits (B, n_classes).

    ConvBlock を 4 段重ねて特徴を抽出し，Global Average Pooling で
    時間・周波数方向を 1 点に潰してから全結合で分類する．
    GAP を使うことで入力の時間長 T' に依存せず固定次元のベクトルになる．
    """

    def __init__(self, n_classes: int = 35, base: int = 32):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1, base),
            ConvBlock(base, base * 2),
            ConvBlock(base * 2, base * 4),
            ConvBlock(base * 4, base * 4),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(base * 4, n_classes)

    def forward(self, x):
        x = self.features(x)            # (B, C, h, w)
        x = self.gap(x).flatten(1)      # (B, C)
        return self.classifier(x)       # (B, n_classes)
