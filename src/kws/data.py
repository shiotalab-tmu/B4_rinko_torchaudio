"""SPEECHCOMMANDS のデータ処理：Dataset・ラベル変換・collate_fn・log-mel 変換.

特徴量変換（log-mel）は「前処理側」に置く方針なので，ここ（data.py）で持つ．
collate_fn の中で「波形を 1 秒に揃えて stack → log-mel」まで行い，
モデルには (B, 1, n_mels, T') の log-mel スペクトログラムが流れる．
"""

from __future__ import annotations

import os

import torch
import torchaudio
import torchaudio.transforms as T
from torch import nn
from torch.utils.data import DataLoader

SAMPLE_RATE = 16000
NUM_SAMPLES = 16000  # 1 秒ぶん（16kHz × 1s）

# SPEECHCOMMANDS v0.02 の 35 クラス．
# 順序を固定しておくことで「ラベル → index」の対応が環境に依らず再現可能になる．
LABELS = [
    "backward", "bed", "bird", "cat", "dog", "down", "eight", "five", "follow", "forward",
    "four", "go", "happy", "house", "learn", "left", "marvin", "nine", "no", "off",
    "on", "one", "right", "seven", "sheila", "six", "stop", "three", "tree", "two",
    "up", "visual", "wow", "yes", "zero",
]
label_to_index = {label: i for i, label in enumerate(LABELS)}
index_to_label = {i: label for i, label in enumerate(LABELS)}


def make_logmel(sample_rate: int = SAMPLE_RATE, n_mels: int = 64) -> nn.Module:
    """波形 (B, T) → log-mel スペクトログラム (B, n_mels, T') を返す変換."""
    return nn.Sequential(
        T.MelSpectrogram(
            sample_rate=sample_rate, n_fft=400, hop_length=160, n_mels=n_mels
        ),
        T.AmplitudeToDB(),  # パワー → dB（"log" mel）
    )


def pad_or_trim(waveform: torch.Tensor, num_samples: int = NUM_SAMPLES) -> torch.Tensor:
    """1 次元波形を num_samples ちょうどに揃える（短ければ 0 パディング，長ければ切り詰め）.

    長さを揃える理由は「バッチ化」：ミニバッチを 1 つの (B, T) テンソルに
    stack するには，全サンプルの長さが同じである必要があるため．
    """
    n = waveform.shape[-1]
    if n < num_samples:
        waveform = torch.nn.functional.pad(waveform, (0, num_samples - n))
    elif n > num_samples:
        waveform = waveform[..., :num_samples]
    return waveform


def make_collate_fn(transform: nn.Module):
    """DataLoader 用の collate_fn を作る．

    各サンプル (waveform, sample_rate, label, ...) を
      1) 波形を 1 秒に揃える
      2) バッチで stack して (B, T)
      3) log-mel 変換して (B, 1, n_mels, T')
    に変換し，ラベルは index の LongTensor にして返す．
    """

    def collate(batch):
        waveforms, targets = [], []
        for waveform, _sr, label, *_ in batch:
            waveforms.append(pad_or_trim(waveform.squeeze(0)))
            targets.append(label_to_index[label])
        waveforms = torch.stack(waveforms)         # (B, T)
        feats = transform(waveforms).unsqueeze(1)  # (B, 1, n_mels, T')
        targets = torch.tensor(targets, dtype=torch.long)
        return feats, targets

    return collate


def get_dataset(root: str, subset: str):
    """SPEECHCOMMANDS の指定 subset（training/validation/testing）を返す（無ければ DL）."""
    # torchaudio は root 自体は自動作成しないため，先に用意しておく．
    os.makedirs(root, exist_ok=True)
    return torchaudio.datasets.SPEECHCOMMANDS(root=root, download=True, subset=subset)


def get_dataloaders(
    root: str = "data",
    batch_size: int = 256,
    n_mels: int = 64,
    num_workers: int = 4,
    sample_rate: int = SAMPLE_RATE,
) -> dict[str, DataLoader]:
    """train / val / test の DataLoader を作って dict で返す."""
    transform = make_logmel(sample_rate, n_mels)
    collate = make_collate_fn(transform)
    splits = {"train": "training", "val": "validation", "test": "testing"}
    loaders = {}
    for split, subset in splits.items():
        dataset = get_dataset(root, subset)
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            collate_fn=collate,
            pin_memory=True,
            drop_last=(split == "train"),
        )
    return loaders
