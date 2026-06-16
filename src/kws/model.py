"""log-mel スペクトログラムを入力にとる小型 2D-CNN（AudioCNN）.

【第2回で穴埋めするファイル】
notebook 02 で動かしたコードをここに整理する．`# TODO` を埋めると動く．
詰まったら解答: `git checkout ans -- src/kws/model.py`

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
        # TODO(第2回): Conv2d(3x3, padding=1, bias=False) -> BatchNorm2d -> ReLU -> MaxPool2d(2)
        self.block = nn.Sequential(
            ...  # TODO
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
        # TODO(第2回): ConvBlock を 1->base, base->base*2, base*2->base*4, base*4->base*4 の4段
        self.features = nn.Sequential(
            ...  # TODO
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(base * 4, n_classes)

    def forward(self, x):
        x = self.features(x)            # (B, C, h, w)
        x = self.gap(x).flatten(1)      # (B, C)
        # TODO(第2回): 分類器に通して logits を返す
        raise NotImplementedError("forward を実装してね（notebook 02 参照）")
