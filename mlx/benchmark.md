# MLX 推論ベンチマーク

Apple M1 Max / 64 GB、MLX 0.32.3。指定の650件全件、fp16 checkpoint、既定の最大256トークン。時刻は開始時刻（ローカル、UTCオフセット付き）。Elapsed time は標準ベンチマークの出力値。各batch sizeは同じbatch sizeの直前の有効な全件結果と比較します。部分計測は性能調査のみです。

2026-10-12 04:20以降、ユーザーの追加指定に従い、長さ順の並べ替えとencoderバッチ分割を完全に撤去しました。並べ替えを含む部分計測と `q6-compiled-chunk4-b16` は参考値であり、採用判断・最終結果には使用しません。全件測定のCLI引数は最初の添付の指定を維持（入力 `data/fleurs_ja_test.parquet`、最大生成256tokenは既定値）し、batch size 1/16以外は変更していません。

| 実行時刻 | 実験 | Batch size | CER | Elapsed time | 前回からの変更（300字以内） | 改善コミット |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-10-12T03:51:38-05:00 | baseline-b16 | 16 | 6.05% | 171.80 seconds | baseline-変更なし | — |
| 2026-10-12T03:54:47-05:00 | baseline-b1 | 1 | 6.06% | 313.22 seconds | baseline-変更なし | — |
| 2026-10-12T04:03:10-05:00 | fused-b16 | 16 | 6.05% | 124.49 seconds | 短い畳み込み、encoder depthwise＋Swoosh、masked group norm、相対位置softmaxをMetalで統合。EOS判定の同期を統合。重みはfp16のまま。 | `188ca12` |
| 2026-10-12T04:06:05-05:00 | fused-b1 | 1 | 6.05% | 148.37 seconds | batch size 16と同じ統合カーネルとEOS同期削減。重みはfp16のまま。 | `188ca12` |
| 2026-10-12T04:09:37-05:00 | 部分計測：encoder chunk=16 | 16 | 対象外（encoderのみ） | 1.101 seconds | 先頭16件を長さ順に並べ、16件ずつencode。3回の最短値。全件ベンチマークとは比較不可。 | — |
| 2026-10-12T04:09:41-05:00 | 部分計測：encoder chunk=8 | 16 | 対象外（encoderのみ） | 0.910 seconds | 先頭16件を長さ順に並べ、8件ずつencode。3回の最短値。全件ベンチマークとは比較不可。 | — |
| 2026-10-12T04:09:44-05:00 | 部分計測：encoder chunk=4 | 16 | 対象外（encoderのみ） | 0.778 seconds | 先頭16件を長さ順に並べ、4件ずつencode。3回の最短値。全件ベンチマークとは比較不可。 | — |
| 2026-10-12T04:09:46-05:00 | 部分計測：encoder chunk=2 | 16 | 対象外（encoderのみ） | 0.834 seconds | 先頭16件を長さ順に並べ、2件ずつencode。3回の最短値。全件ベンチマークとは比較不可。 | — |
| 2026-10-12T04:09:49-05:00 | 部分計測：encoder chunk=1 | 16 | 対象外（encoderのみ） | 0.896 seconds | 先頭16件を長さ順に並べ、1件ずつencode。3回の最短値。全件ベンチマークとは比較不可。 | — |
| 2026-10-12T04:14:25-05:00 | 部分計測：decoder 16bit | 1 | 対象外（合成入力） | 0.004613 seconds/token | encoderを除き、音声長300・50tokenを3回実行した最短値。線形層の重み精度を比較。全件ベンチマークとは比較不可。 | — |
| 2026-10-12T04:14:27-05:00 | 部分計測：decoder 16bit | 16 | 対象外（合成入力） | 0.017267 seconds/token | encoderを除き、音声長300・50tokenを3回実行した最短値。線形層の重み精度を比較。全件ベンチマークとは比較不可。 | — |
| 2026-10-12T04:14:31-05:00 | 部分計測：decoder 8bit | 1 | 対象外（合成入力） | 0.004293 seconds/token | encoderを除き、音声長300・50tokenを3回実行した最短値。線形層の重み精度を比較。全件ベンチマークとは比較不可。 | — |
| 2026-10-12T04:14:31-05:00 | 部分計測：decoder 8bit | 16 | 対象外（合成入力） | 0.012416 seconds/token | encoderを除き、音声長300・50tokenを3回実行した最短値。線形層の重み精度を比較。全件ベンチマークとは比較不可。 | — |
| 2026-10-12T04:14:35-05:00 | 部分計測：decoder 6bit | 1 | 対象外（合成入力） | 0.002794 seconds/token | encoderを除き、音声長300・50tokenを3回実行した最短値。線形層の重み精度を比較。全件ベンチマークとは比較不可。 | — |
| 2026-10-12T04:14:35-05:00 | 部分計測：decoder 6bit | 16 | 対象外（合成入力） | 0.012174 seconds/token | encoderを除き、音声長300・50tokenを3回実行した最短値。線形層の重み精度を比較。全件ベンチマークとは比較不可。 | — |
| 2026-10-12T04:14:39-05:00 | 部分計測：decoder 4bit | 1 | 対象外（合成入力） | 0.004100 seconds/token | encoderを除き、音声長300・50tokenを3回実行した最短値。線形層の重み精度を比較。全件ベンチマークとは比較不可。 | — |
| 2026-10-12T04:14:39-05:00 | 部分計測：decoder 4bit | 16 | 対象外（合成入力） | 0.013192 seconds/token | encoderを除き、音声長300・50tokenを3回実行した最短値。線形層の重み精度を比較。全件ベンチマークとは比較不可。 | — |
| 2026-10-12T04:16:05-05:00 | 参考・撤去済：q6-compiled-chunk4-b16 | 16 | 6.09% | 117.47 seconds | encoderを長さ順4件に分割後に元順へ復元。decoderの固定形状ブロックをcompileし、線形層をgroup 64の6bitに量子化。元checkpointは変更なし。 | — |
| 2026-10-12T04:18:58-05:00 | q6-compiled-b1 | 1 | 6.09% | 121.59 seconds | 6bit・固定形状compile版。batch size 1のため並べ替え・分割は実行されず、元の処理順を維持。 | `3a04088` |
| 2026-10-12T04:21:54-05:00 | q6-compiled-b16 | 16 | 6.08% | 118.52 seconds | 並べ替え・encoder分割を完全撤去。元のデータ順・batch 16のまま。6bit量子化と固定形状compile、統合Metalカーネルのみ。 | `3a04088` |
| 2026-10-12T04:27:02-05:00 | 不採用：q6-encoder-fused-b16 | 16 | 6.08% | 118.31 seconds | BiasNormをSIMD Metalカーネル化し、SwooshL・bypass・masked GLUを演算融合。データ順・batch・引数は不変。6bit量子化を維持。 | — |
| 2026-10-12T04:29:37-05:00 | 不採用：q6-encoder-fused-b1 | 1 | 6.08% | 131.14 seconds | batch size 16と同じBiasNorm専用カーネル・SwooshL/bypass/masked GLU融合。順序・引数は変更なし。 | — |
| 2026-10-12T04:32:55-05:00 | q4-compiled-b16 | 16 | 6.12% | 116.49 seconds | 遅くなったBiasNorm・活性化融合を撤去し3a04088の演算へ復元。decoder線形層の量子化だけ6bitから4bitへ変更。順序・引数は不変。 | `dcf99f9` |
| 2026-10-12T04:35:10-05:00 | q4-compiled-b1 | 1 | 6.12% | 106.79 seconds | 同じ4bit量子化版。その他のモデル実装、データ順、CLI引数は変更なし。 | `dcf99f9` |
| 2026-10-12T04:37:57-05:00 | 不採用：q8-compiled-b16 | 16 | 6.03% | 119.47 seconds | decoder線形層の量子化だけ4bitから8bitへ変更。group 64、演算、データ順、CLI引数は不変。 | — |
| 2026-10-12T04:40:27-05:00 | 不採用：q8-compiled-b1 | 1 | 6.03% | 124.33 seconds | 同じ8bit量子化版。その他のモデル実装、データ順、CLI引数は変更なし。 | — |
| 2026-10-12T04:44:29-05:00 | final-b16 | 16 | 6.12% | 118.63 seconds | 採用版dcf99f9（4bit）を再測定。8bit実験から4bitへ復元した以外の実装変更なし。指定650件・元順序・全引数固定。 | `dcf99f9` |
| 2026-10-12T04:46:54-05:00 | final-b1 | 1 | 6.12% | 109.52 seconds | 採用版dcf99f9（4bit）の再測定。実装変更なし。指定650件・元順序・全引数固定。 | `dcf99f9` |

## 最終結果

採用実装は `dcf99f98085db98c2e891dceafa7deff35220def`。統合Metalカーネル、decoder固定形状ブロックのcompile、decoder線形層の4bit量子化を使用します。量子化は読み込み時のメモリ上だけで行い、元checkpointは変更していません。encoder、埋め込み、cross-attentionの直接保持する射影重みはfp16です。

| Batch size | 変更前 | 最終再測定 | 時間短縮率 | 高速化倍率 | CER（変更前 → 最終） |
| --- | --- | --- | --- | --- | --- |
| 16 | 171.80秒 | 118.63秒 | 30.95% | 1.45倍 | 6.05% → 6.12% |
| 1 | 313.22秒 | 109.52秒 | 65.03% | 2.86倍 | 6.06% → 6.12% |

最終再測定はいずれも650/650件、Skipped 0、CER 7%未満です。同じ実装の初回測定値も表に残し、上表には最後の再測定値を使用しています。データ順、batch構成、生成上限、EOS停止条件、CLI引数は維持しています。追加のBiasNorm・活性化融合はbatch size 1で悪化したため撤去し、8bit量子化は両条件で4bitより遅かったため採用していません。

## 最終検証

`ruff check . --fix` と `ruff format .` は通過し、自動修正はありませんでした。Macで実行可能なpytestは199件通過しました。通常の `pytest` は既存のTriton未導入により2ファイルの収集で失敗するため、`tests/test_gradient_kernels.py` と `tests/test_training_kernels.py` を除外して検証しています。通常の `pyright` にも未変更の `src/modeling/kernels` 内のTriton importエラー13件が残っています。`mlx/model`、`mlx/inference`、`mlx/kernel`、`tests/test_mlx.py`、`tests/test_mlx_kernels.py` の型チェックはエラー0件です。

作業前後のSHA-256比較で、`src/` の全78ファイルと評価データの一致を確認しました。

| 対象 | SHA-256 |
| --- | --- |
| `data/fleurs_ja_test.parquet` | `1cab4feab0e681bddfbb11f7ea5164e9d8f45be08628d488fc19559dc8b52aa6` |
| `outputs/models/lambda-audio-v6/model-fp16.safetensors` | `ee1f2b7f538875a0befd80eb024e5cbc4fed208444fe6212260b0150ea4b5b04` |

## メモリ診断

| 実行時刻 | 実験 | Batch size | CER | Elapsed time | 前回からの変更（300字以内） | 改善コミット |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-10-12T05:15:33-05:00 | memory-diagnostic-b1 | 1 | 6.12% | 114.06 seconds | 保存済みmodel-mlx-4bitを使用。実装・順序・生成設定は不変。Python監視APIで50件ごとにMLXの使用中・キャッシュ容量を観測。標準タイマー終了後だけキャッシュ解放を実施。 | — |

650件完了時のMLX使用中メモリは0.499 GB、実行中の使用中メモリのピークは1.164 GB、再利用用キャッシュは35.346 GBでした（GBは10億bytes）。各音声の処理終了時の使用中メモリは約0.50 GBで安定しており、増加の主因はキャッシュです。標準の計時終了後、モデルを保持したまま `mx.clear_cache()` を実行すると、使用中メモリは0.499 GBのまま、キャッシュだけ0 bytesになりました。キャッシュ上限の既定値は別プロセスで65.284 GBと確認しました。推論実装の変更はありません。

詳細なメモリ推移は `outputs/diagnostics/mlx-memory-batch1.json` に保存しています。診断中に外部から `vmmap -summary` でも観測したため、この実行は通常の速度比較とは分けて扱います。

## メモリキャッシュ無効化

| 実行時刻 | 実験 | Batch size | CER | Elapsed time | 前回からの変更（300字以内） | 改善コミット |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-10-12T05:23:10-05:00 | no-cache-b1 | 1 | 6.12% | 118.37 seconds | モデル読み込み時にMLXのメモリ再利用キャッシュ上限を0に設定し、既存キャッシュを解放。保存済み4bitモデル、入力順、生成設定、CLI引数は維持。Python監視APIでメモリ容量を観測。 | `5952133` |
| 2026-10-12T05:25:29-05:00 | no-cache-b16 | 16 | 6.12% | 117.37 seconds | モデル読み込み時にMLXのメモリ再利用キャッシュ上限を0に設定し、既存キャッシュを解放。保存済み4bitモデル、入力順、生成設定、CLI引数は維持。Python監視APIでメモリ容量を観測。 | `5952133` |

両条件とも650/650件、Skipped 0、CER 6.12%。各バッチ終了時に確認した再利用用キャッシュはすべて0 bytesでした。MLX管理下の使用中メモリのピークはbatch size 1で1.103 GB、16で4.800 GBです。これはPythonや入力データ等を含むプロセス全体の使用量ではありません。無効化したのはメモリ領域の再利用キャッシュで、生成用KVキャッシュと部分コンパイルは維持しています。

前回のbatch size 1のメモリ診断114.06秒に対して、今回は118.37秒でした。batch size 16の前回最終再測定118.63秒に対して、今回は117.37秒でした。いずれも単発測定であり、実行時の変動や診断処理の差を含みます。キャッシュ無効化によって常に高速化するとは判断していません。詳細は `outputs/diagnostics/mlx-no-cache-batch1.json` と `outputs/diagnostics/mlx-no-cache-batch16.json` に保存しています。

変更後のRuffチェック・整形とMLX範囲の型チェックは通過しました。pytestは既存のTriton依存2ファイルを除いて204件通過。通常のpytestの収集エラー2件と、全体のpyrightのTriton importエラー13件は既存のものです。`src/` の変更はなく、評価データと保存済み4bit checkpointのSHA-256も一致しています。

## Whisper large-v3-turbo MLX 4bit

| 実行時刻 | 実験 | Batch size | CER | Elapsed time | 前回からの変更（300字以内） | 改善コミット |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-10-12T06:04:15-05:00 | whisper-large-v3-turbo-q4-b1 | 1 | 5.54% | 740.98 seconds | 比較モデルをMLX Whisper large-v3-turboの4bit・group 64へ変更。同じ650件・元順序・既存CER/Elapsed計測を使用。日本語greedy、最大256token、timestamp・再試行・無音スキップなし。メモリ再利用キャッシュ上限0。 | — |

650/650件完了、Skipped 0。モデルは `mlx-community/whisper-large-v3-turbo-q4`、revisionは `660c343bbf4e52ac257f0b7d952e5388e6f93bef`、MLXは0.32.3、mlx-whisperは0.4.3です。重みは4bit・group 64、演算dtypeはfloat16です。`mlx/benchmark.py` からモデルを登録し、既存の `src.benchmark.runner` を使用しました。`src/` と評価データに変更はありません。モデル読み込みは既存仕様どおりElapsed timeに含みません。

```bash
python -m mlx.benchmark \
  --model whisper-mlx \
  --input data/fleurs_ja_test.parquet \
  --batch-size 1 \
  --no-show-table
```

MLXのメモリ再利用キャッシュは観測した各時点で0 bytes、使用中メモリのピークは1.330 GBでした。詳細は `outputs/diagnostics/mlx-whisper-turbo-q4-batch1.json` に保存しています。

Ruffチェック・整形と追加ファイルの型チェックは通過しました。pytestは既存のTriton依存2ファイルを除いて208件通過しました。通常のpytestの収集エラー2件と、全体のpyrightのTriton importエラー13件は既存のものです。

---

