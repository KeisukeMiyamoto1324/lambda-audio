# Whisper MLX ベンチマーク

`mlx-community/whisper-large-v3-turbo-4bit`を使う、Apple Silicon Mac向けの推論・CER評価・時間計測です。`whisper/`内で完結し、`src/`やLambdaAudioの`mlx/`内のコード、PyTorch、Transformersには依存しません。

## セットアップ

Python 3.12以降を使用し、プロジェクトのルートで実行してください。

```bash
python3 -m venv .venv-whisper
source .venv-whisper/bin/activate
pip3 install -r whisper/requirements.txt
pip3 install --no-deps -r whisper/backend-requirements.txt
```

バックエンドは`mlx-audio-plus==0.1.8`です。パッケージ全体の依存には音声合成などが含まれるため、`--no-deps`で導入し、Whisperの実行に必要なライブラリだけを`requirements.txt`で指定しています。同じ`mlx_audio`モジュールを提供する`mlx-audio`との混在を避けるため、専用の仮想環境を使用してください。

作業ディレクトリの`.env`に`HF_TOKEN`を設定してください。別の場所にある場合は`--env-file /path/to/.env`で指定できます。モデル・設定・tokenizerは、このトークンで指定リポジトリから取得します。

## 推論と時間計測

```bash
python -m whisper.benchmark \
  --input data/fleurs_ja_test.parquet \
  --batch-size 1 \
  --max-new-tokens 256 \
  --no-show-table
```

入力はWAV、または`audio.bytes`と`text`列を持つParquetです。入力順を維持し、16 kHz・monoに変換します。30秒を超える音声はエラーになります。バッチサイズは1のみ対応します。

日本語の文字起こしを指定し、greedy生成を使用します。最大生成トークン数は256（変更可能範囲1〜444）で、EOSでも停止します。timestamp生成・温度を変えた再試行・無音判定によるスキップ・前の音声の文脈引き継ぎは無効です。

`Elapsed time`は音声デコード・前処理・生成・CER集計を含みます。モデルの取得・読み込み・tokenizerの準備と、Parquetの読み込みは除外します。明示的な推論ウォームアップは行いません。MLXのメモリ再利用キャッシュ上限はLambdaAudioと同じ0です。

CERの正規化・除外条件と時間計測の区間はLambdaAudioのMLXベンチマークと同じです。モデル固有の前処理はそれぞれ使用するため、Whisperでは音声が30秒の窓にパディングされます。結果は各モデルの推論処理全体の速度として比較してください。

`whisper/`だけをコピーして利用する場合は、コピー先の親ディレクトリから実行し、入力と`.env`のパスを指定してください。

バックエンドの実装: https://github.com/DePasqualeOrg/mlx-audio-plus
