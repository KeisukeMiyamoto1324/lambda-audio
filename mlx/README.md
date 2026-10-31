# LambdaAudio MLX

Apple Silicon Mac向けの推論・CER評価・時間計測です。`src/`、PyTorch、Transformers、Triton、学習用ライブラリは使用しません。

## セットアップ

Python 3.12以降を使用してください。`mlx/`の親ディレクトリで実行します。

```bash
python3 -m venv .venv-mlx
source .venv-mlx/bin/activate
pip3 install -r mlx/requirements.txt
```

作業ディレクトリの`.env`に`HF_TOKEN`を設定してください。Hubへのアクセスには必ずこのトークンを使用します。別の場所にある場合は`--env-file /path/to/.env`で指定できます。

## 推論と時間計測

```bash
python -m mlx.benchmark \
  --checkpoint outputs/models/lambda-audio-v6/model-mlx-4bit.safetensors \
  --input data/fleurs_ja_test.parquet \
  --batch-size 1 \
  --max-new-tokens 256 \
  --no-show-table
```

`--checkpoint`を省略すると、`KeisukeMiyamoto/lambda-audio-v6`の`model-mlx-4bit.safetensors`をHugging Faceのキャッシュに取得します。設定とtokenizerは既定のencoder・decoderリポジトリから取得します。

入力はWAVファイル、または`audio.bytes`と`text`列を持つParquetです。入力順を維持し、音声を16 kHz・monoに変換します。30秒を超える音声はエラーになります。生成はgreedyで、EOSまたは指定トークン数で停止します。

`Elapsed time`は音声デコード・前処理・生成・CER集計を含み、モデル取得・読み込みとParquetの読み込みを除きます。明示的なウォームアップは行いません。CERの正規化・除外条件は従来のベンチマークと同じです。

`mlx/`だけを別のディレクトリへコピーしても利用できます。コピー先の親ディレクトリから実行し、入力・checkpoint・`.env`のパスを指定してください。`mlx/benchmark.md`の過去の実行コマンドは当時の構成を記録したものです。
