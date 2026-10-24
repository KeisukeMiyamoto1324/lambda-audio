mlx/ は lambda-audio の mlx　形式の実装です。あなたの仕事は mlx 形式で lambda-audio 向けのカスタムカーネルを作成し、 lambda-audio の文字起こし精度を保ちながら、推論時間を最小化することです。既存の実装に縛られることなく、柔軟に発想し、様々なアプローチを試し、限界を超えて下さい。

## 測定方法
文字起こし精度、推論時間は以下のベンチマークを使用して計測して下さい。

### batch size 16
python -m src.benchmark \
  --model lambda-audio-mlx \
  --checkpoint outputs/models/lambda-audio-v6/model-fp16.safetensors \
  --input data/fleurs_ja_test.parquet \
  --batch-size 16 \
  --no-show-table

(実行結果の例)
Inference ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 650/650 CER: 6.05% Skipped: 0 0:00:00
Model name: lambda-audio-mlx
CER: 6.05%
Elapsed time: 189.30 seconds

### batch size 1
python -m src.benchmark \
  --model lambda-audio-mlx \
  --checkpoint outputs/models/lambda-audio-v6/model-fp16.safetensors \
  --input data/fleurs_ja_test.parquet \
  --batch-size 1 \
  --no-show-table

(実行結果の例)
Inference ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 650/650 CER: 6.06% Skipped: 0 0:00:00
Model name: lambda-audio-mlx
CER: 6.06%
Elapsed time: 322.88 seconds

## 要件
- 上記コマンドでベンチマークを実行した時、CERが7%未満であること。
- batch size 1, 16 の場合で推論時間を最小化すること。
- カスタムカーネルは mlx/kernel に実装すること。

## 禁止事項
- src/ のコードは絶対に変更してはいけません。
- data/fleurs_ja_test.parquet を変更してはいけません。

## その他
- ベンチマークを実行した際は、毎度結果を mlx/benchmark.md に記録して下さい。記録してほしい項目は以下の通り。
  - ベンチマークを実行した時刻
  - CER
  - Elapsed time
  - 前回のベンチマーク実行から何を変更したか。(最大300字で概要を簡潔に説明)
  - 前回のベンチマーク結果から、要件を満たしながら推論時間が改善したならそこまでの変更を git に commit し、そのコミットハッシュ