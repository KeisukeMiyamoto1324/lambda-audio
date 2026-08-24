# LambdaAudio

## 概要
- Encoder-Decoder モデルの日本語特化ASRモデル


## モデル入出力

### 入力
- 30秒以下の音声（形式はEncoderの仕様に準ずる）

### 出力
- 文字起こし後のテキスト


## データセット
- litagin/reazon-speech-v2-clone
  - https://huggingface.co/datasets/litagin/reazon-speech-v2-clone
  - 学習時にストリーミングで直接読み込む


## ベースモデル

### エンコーダ部分
- reazon-research/japanese-zipformer-base-k2 
  - https://huggingface.co/reazon-research/japanese-zipformer-base-k2
  - 出力・隠れ次元は768

### デコーダ部分
- LiquidAI/LFM2.5-230M-Base
  - https://huggingface.co/LiquidAI/LFM2.5-230M-Base
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


## Gated cross attention
- LFM2.5-230M-Baseにはfull_attention層が6個あるため、そのすべてに cross attention
- Self-AttentionとFFNの間でLFM側の隠れ状態と統合
- Scalar Gate
- LFMに合わせてnum_heads = 16, head_dim = 64


## 学習対象
1. LFM, encoderを固定し、その他のみを学習させる。安定してきたらLFM, encoderの重みも学習を開始する
2. Adapter＋Cross-Attention＋LFM2.5+Zipformer すべて


## Loss
- CE loss
