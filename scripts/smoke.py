"""環境チェック：import・version・CUDA・データの有無を確認する．
   データの DL はしない（download を踏ませない）．DL は手順5で明示的に行う．"""
from pathlib import Path
import torch
import torchaudio

print("torch       :", torch.__version__)
print("torchaudio  :", torchaudio.__version__)
print("CUDA avail  :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device      :", torch.cuda.get_device_name(0))

# データは「既に取得済みか」だけ確認する（自動 DL を踏ませない）
sc_dir = Path("data") / "SpeechCommands" / "speech_commands_v0.02"
if sc_dir.exists():
    print("SPEECHCOMMANDS: found  ->", sc_dir)
else:
    print("SPEECHCOMMANDS: NOT found. 手順5のデータ取得をまだ実行していません．")
