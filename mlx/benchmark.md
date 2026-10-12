# MLX 推論ベンチマーク

Apple M1 Max / 64 GB、MLX 0.32.3。指定の650件全件、fp16 checkpoint、既定の最大256トークン。時刻は開始時刻（ローカル、UTCオフセット付き）。Elapsed time は標準ベンチマークの出力値。各batch sizeは同じbatch sizeの直前結果と比較します。

| 実行時刻 | 実験 | Batch size | CER | Elapsed time | 前回からの変更（300字以内） | 改善コミット |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-10-12T03:51:38-05:00 | baseline-b16 | 16 | 6.05% | 171.80 seconds | baseline-変更なし | — |
| 2026-10-12T03:54:47-05:00 | baseline-b1 | 1 | 6.06% | 313.22 seconds | baseline-変更なし | — |
| 2026-10-12T04:03:10-05:00 | fused-b16 | 16 | 6.05% | 124.49 seconds | 短い畳み込み、encoder depthwise＋Swoosh、masked group norm、相対位置softmaxをMetalで統合。EOS判定の同期を統合。重みはfp16のまま。 | — |
| 2026-10-12T04:06:05-05:00 | fused-b1 | 1 | 6.05% | 148.37 seconds | batch size 16と同じ統合カーネルとEOS同期削減。重みはfp16のまま。 | — |
