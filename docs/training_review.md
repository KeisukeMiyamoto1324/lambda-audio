# LambdaAudio 学習処理の調査レポート

調査日: 2026年9月19日

## 1. VRAMを増やさずに学習を高速化する候補

### 1-1. 勾配の正規化に伴うGPU待ち時間を減らす

**確認した事実:** 重みを更新する直前に、トークン数をGPUテンソルにしてから`.item()`でCPUへ取り出しています。その後、各パラメータの勾配をPythonのループで個別に乗算しています。

GPU上の値をCPUへ取り出す操作では、GPUの処理完了を待つ必要があります。また、小さな演算を個別に呼び出すことにもコストがあります。

**改善案:** すでにCPU側で保持しているトークン数から倍率を求め、トークン数のGPUテンソル化と`.item()`による取り出しを省きます。勾配への乗算は、複数のテンソルをまとめて処理する方式を検討します。

**確認事項:** 複数バッチの勾配をためてから重みを更新する「勾配蓄積」を使う場合も、蓄積したトークン数とLightning側の損失の平均化を考慮し、正規化後の勾配が従来と一致することを確認します。効果の大きさは、現在のGPU待ち時間とパラメータ数に依存します。

対象: [BaseLambdaAudioLightningModule.on_before_optimizer_step](../src/training/lightning.py)

参考: [PyTorch Performance Tuning Guide — CPUとGPUの不要な同期](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide)

## 2. 学習速度を落とさずにVRAMを減らす候補

### 2-1. 音声入力直後のGroupNormで巨大な中間テンソルを作らない

**確認した事実:** GroupNormは、特徴量の平均と分散を使って値を整える処理です。本実装ではpaddingを計算対象から除外するため、特徴量全体をFP32に変換し、平均との差、二乗、マスクとの乗算などを別々に計算しています。

音声入力直後は時間方向の長さが大きいため、ここで作る中間テンソルは非常に大きくなります。

既定の最初の畳み込みが512チャネル・stride 5で、Stage 1のpadding込み音声量が設定上限の2800秒に達した場合、特徴テンソル1個のFP32容量は概算で次の通りです。

```text
2800秒 × 16,000サンプル/秒 ÷ 5 × 512チャネル × 4バイト
  ≈ 18.35 GB
  ≈ 17.09 GiB
```

これは**テンソル1個の概算容量**であり、実測したピークVRAMではありません。実際のバッチ量、読み込んだモデル設定、演算中のテンソルの生存期間によってピークは変わります。

**改善案:** paddingを除外する現在の意味を保った融合カーネルを検討します。平均・分散の集計はFP32を維持し、全要素分のFP32中間テンソルの作成や書き戻しを減らします。

**確認事項:** paddingのある入力で従来の出力と一致することを確認します。paddingを含めて集計する通常のGroupNormへの単純置換では、計算の意味が変わります。手動の小分け処理だけでは速度が低下する可能性があるため、速度とメモリを同時に測る必要があります。

対象: [Fp32GroupNorm.forward](../src/modeling/zipformer/utils.py)、[ConvFeatureExtractionModel](../src/modeling/zipformer/wav2vec2_module.py)、[畳み込みの既定設定](../src/modeling/zipformer/configuration_zipformer.py)、[Stage 1のバッチ設定](../src/stage1/config.py)

### 2-2. 学習時の音声K/Vの結合コピーを省く

**確認した事実:** cross-attentionでは、音声の特徴からKeyとValue（K/V）という参照用の表現を作ります。現在は各層のK/Vをリストに作った後、`torch.stack`で全層分を1個のテンソルへ結合しています。

結合時には元のテンソルと結合先が一時的に共存し、コピーが発生します。

**改善案:** 学習時には各層のK/Vを個別に渡す、または各層で必要になった時点で計算する構成を検討します。生成時は同じ音声を繰り返し参照するため、現在のK/V再利用を維持する必要があります。

**確認事項:** K/Vの計算回数を増やさずに結合だけを省けることを確認します。勾配計算にもK/Vが必要なため、K/V全体の保存が不要になるわけではありません。主な削減対象は結合時のコピーと一時的な重複です。

対象: [AudioLfm2ForCausalLM.project_audio_key_values](../src/modeling/lfm2_decoder.py)、[GatedCrossAttention.project_key_value](../src/modeling/cross_attention.py)

### 2-3. 凍結した重みの一部をBF16で保持する

**確認した事実:** decoderは明示的にFP32で読み込まれます。学習にはBF16混合精度を使用していますが、混合精度は演算の精度を切り替える仕組みであり、FP32で保持する重みそのものを一律に半分の容量にする設定ではありません。

Stage 1ではdecoderの事前学習済み部分を更新しないため、そこにFP32の重みを保持する必要性を個別に検討できます。

**改善案:** 凍結したLinearやEmbeddingなどをBF16で保持し、FP32が必要な正規化処理などは維持します。対象の重み容量は、1要素4バイトから2バイトへ半減します。

**確認事項:** これは対象重みの容量についての効果であり、モデル全体のVRAMが半減するという意味ではありません。数値が丸められるため、損失とCERの比較が必要です。Stage 2で重みを更新する際の精度方針は別途検討し、学習対象の重みまで一律にBF16へ変換しないようにします。

対象: [load_model_components](../src/modeling/loading.py)、[create_trainer](../src/training/trainer.py)

参考: [PyTorch Automatic Mixed Precision](https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html)
