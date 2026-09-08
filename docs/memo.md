# LambdaAudio

## 参考資料
- https://huggingface.co/nvidia/parakeet-tdt_ctc-0.6b-ja
  - グローバルバッチは 19,200秒（320分、5時間20分）
  - ReazonSpeech v2.0 を合計約60.9周

## 現状の課題
- オーディオエンコーダが重要
- グローバルバッチサイズが足りない？（現状1400秒くらい)
