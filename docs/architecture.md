# LambdaAudio

## 概要
- Encoder-Decoder モデルの日本語特化ASRモデル

## データセット
- reazon-research/reazonspeech (https://huggingface.co/datasets/reazon-research/reazonspeech?utm_source=chatgpt.com)

## ベースモデル

### エンコーダ部分
- reazon-research/japanese-zipformer-base-k2 
  - https://huggingface.co/reazon-research/japanese-zipformer-base-k2
  - 出力・隠れ次元は768

### デコーダ部分
- LiquidAI/LFM2.5-230M 
  - https://huggingface.co/LiquidAI/LFM2.5-230M
  - 出力・隠れ次元は1024

## アダプター

Zipformer出力 [B, T, 768]
  ↓
LayerNorm(768)
  ↓
Conv1d(768 → 768, kernel=3, stride=2, padding=1)　(時間方向を1/2に圧縮)
  ↓
GELU
  ↓
Linear(768 → 1024)
  ↓
LayerNorm(1024)
  ↓
音声特徴 [B, ceil(T/2), 1024]