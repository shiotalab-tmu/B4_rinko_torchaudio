"""kws: SPEECHCOMMANDS を題材にした PyTorch KWS チュートリアルのパッケージ.

各回の notebook で穴埋めした処理を，このパッケージのモジュールに整理していく:
- data.py     : データ処理（Dataset / ラベル変換 / collate_fn / log-mel）
- model.py    : モデル（AudioCNN）
- train.py    : 学習ループ（CLI）
- evaluate.py : 評価（accuracy / confusion matrix）
- utils.py    : 汎用ユーティリティ（seed / device / checkpoint）
"""

__version__ = "0.1.0"
