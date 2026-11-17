## Introduction (What is your topic in this paper? What are the problems to be solved? What can be the applications if your discussed solutions are successful?)



## Related work (What is the current status in the literature? What methods and ideas did the related papers propose? What are the advantages of your selected (or proposed) method?)

2022年4月に Alayrac らが発表した Flamingo [1] は、事前学習済みの画像 encoder と言語モデルを接続したマルチモーダルモデルである。Flamingo は、画像 encoder と言語モデルのパラメータを固定し、言語モデルの層の間に gated cross attention を追加することで、画像の情報を参照しながら文章を生成する。gate は画像の情報を言語モデルに加える量を制御し、初期値をゼロにすることで、学習開始時には元の言語モデルの出力を維持する。Flamingo は画像や動画を対象としたモデルであるが、事前学習済みの言語モデルに別モーダルの表現を与えるという考え方は、音声を扱うモデルにも応用できる。

同年12月に論文が公開された Whisper [2] は、OpenAI が開発した ASR モデルであり、計680,000 hours に上るオーディオコーパスで学習されている。Whisper はその高いマルチリンガル性能とノイズ耐性を特徴とし、特定のデータセットに追加学習を行わない zero-shot の設定でも高い認識性能を示した。Whisper は encoder-decoder 型のモデルであり、encoder、decoder ともに Transformer を使用し、二つを cross attention で接続することで音声文字起こしを実現している。この研究は、Transformer がASRタスクにおいても高い性能を達成できること、また、大規模で多様な音声データを使った学習が、未知の音声に対する認識性能の向上に有効であることを示した。

2023年10月に Yao らが発表した Zipformer [3] は、音声認識のための encoder であり、認識精度と計算効率の改善を目的としている。Zipformer は中間の encoder stack で音声特徴のフレームレートを下げることで、長い音声系列を処理する際の計算量を削減する。論文中の比較では、近年 Audio Encoder に広く用いられている Conformer アーキテクチャに対し、Zipformer は30秒の音声を処理する際の計算量を、下流の文字起こしタスクの精度を維持しながら約63.4%削減できることが示された。

2025年11月に Amini らが technical report を公開した LFM2 [4] は、Liquid AI が開発した言語モデルであり、デバイス上での効率的な推論を目的としている。LFM2 は gated short convolution と grouped query attention（GQA）を組み合わせた hybrid アーキテクチャである。short convolution は近接した token 間の情報を処理し、少数の attention block が離れた token 間の関係を扱う。また、GQA は複数の query head で key と value を共有することで、KV cache のメモリ使用量を抑える。論文では、同程度のパラメータ数を持つモデルと比較し、CPU 上での prefill と decode を最大約2倍高速化したと報告している。

本研究で構築する lambda-audio は、これらの研究を参考に、Zipformer を音声 encoder、LFM2 系の言語モデルを decoder として使用する。二つのモデルを adapter と gated cross attention で接続することで、音声特徴を参照しながら日本語の文字起こしを生成する。

## Methodology (Discuss in detail your implemented (or proposed) deep learning model, figures and equations can be helpful in the illustration)

本研究で構築する lambda-audio の全体構成を Figure 1 に示す。lambda-audio は、Zipformer を用いた音声 encoder、音声特徴を変換する audio adapter、そして LFM2 系の言語モデルを用いた decoder から構成される encoder-decoder 型の ASR モデルである。Figure 1 の左側は audio encoder が音声波形から特徴を抽出する処理、右側は decoder が音声特徴とそれまでの text token を使って次の token を予測する処理を示している。

音声 encoder は 日本語音声で事前学習済みの zipformer に対して、独自構築したオーディオコーパスで継続事前学習を施したモデルを使用する。encoder がまず16 kHz の音声波形を入力として受け取り、Conv1D feature extractor で時間方向の特徴を抽出する。この時点での音声特徴のフレーム間隔は 0.02 秒である。その後、出力に LayerNorm と linear projection を適用し、Zipformer の encoder stack に入力する。各 Zipformer stack は attention、convolution、feed-forward を組み合わせて音声特徴を処理する。Zipformer stack はモデル中間に行くほど、スタック内で時間方向の表現を圧縮し、1フレームあたりの表現を拡張することで、音声情報を十分保ちながら attention にかかる計算量を削減する。

Audio adapter は、音声 encoder と言語モデルの特徴次元を合わせるとともに、decoder が参照する音声系列の長さを削減する役割を持つ。Figure 1 の audio adapter に示すように、処理は LayerNorm、Conv1D、GELU、linear projection、LayerNorm の順に行う。Conv1D は stride を2とし、音声特徴の系列長を半分にする。これによって、音声特徴のフレーム間隔は 0.04秒になる。これはのちの decoder において、cross attention に必要な計算量を削減するためである。その後、linear projection で特徴次元を decoder の hidden dimension に変換する。

Decoder には LFM2.5-230M に対して日本語の文字起こしテキストで継続事前学習をしたモデルを使用する。Figure 1 の右側に示すように、gated cross attention は元からある full-attention block 内の self-attention と feed-forward の間に追加する。cross attention では text token から得た hidden state を query、audio adapter の出力を key と value として使用することで、文章を生成する各段階で音声特徴を参照する。cross attention の出力には学習可能な gate の tanh を掛け、residual connection によって元の hidden state に加える。gate はゼロで初期化し、学習開始時には事前学習済み言語モデルの出力を維持する。これにより、学習初期に未学習の cross attention の出力が加わることで、事前学習済み言語モデルの内部表現が乱れることを防ぐ。

![Figure 1: lambda-audio architecture](/Users/keisukemiyamoto/Project/LambdaAudio/docs/figures/lambda_audio_architecture.png)

Figure 1: lambda-audio のモデルアーキテクチャ。左側は音声 encoder と audio adapter、右側は gated cross attention を追加した LFM2 系 decoder を示す。

## Experiment (What are the datasets applied? What are the experiments conducted, to show the features or advantages of your implemented (or proposed) deep learning model? What are the results? Figures and tables can be helpful in the illustration)

本研究では、音声 encoder の継続事前学習、decoder の継続事前学習、そして二つのモデルを接続した音声文字起こしの学習という3段階で lambda-audio を構築した。

### 音声 encoder の継続事前学習

音声 encoder のベースモデルには、ReazonSpeech の日本語音声で事前学習された reazon-research/japanese-zipformer-base-k2 [5] を使用した。このモデルは最大長が15.625秒の音声で学習されているが [6]、本研究では30秒までの音声を扱うため、より長い音声系列への適応を目的として CPT を行った。

CPT には独自構築した Whisper Cluster Audio [8] を使用した。このデータセットは、Internet Archive や publicaly available podcast、radio などの音声から作成した約5,383時間の日本語音声で構築されている。データセット構築は以下の手順で実施した。

1. 全ての音声を 3から29.9秒の音声クリップに分割する。
2. 約200時間分の音声をを whisper large v3 turbo の encoder に通して音声表現ベクトルを抽出する。
3. 2で得られた音声表現ベクトルのうち、50%を使用してK-meansで500個のクラスタを作成する。各クラスタに0から499までのidを振る
4. 1で得られた全ての音声クリップをwhisper large v3 turbo の encoder に通して音声表現ベクトルを抽出し、そのベクトルが500個のどのクラスタに属するか、クラスタidを記録する。

encoderのcptでは音声特徴の一部を mask し、その位置の cluster ID を予測する手法を採用した。学習初期 encoderに新しく追加した cluster 予測 head のみを学習し、その後 encoder も更新した。学習後は cluster 予測 head を除き、下流タスクではencoder部分のみ [7] を使用した。

### Decoder の継続事前学習

Decoder のベースモデルには LiquidAI/LFM2.5-230M-Base [9] を使用した。CPTに使用したデータセット は日本語の文字起こしテキストで[11]、日本の公共放送団体であるNHKから公開されている NHK archive と、技術系 podcast の文字起こしから構築されている。学習データは約7,000万 token で2 epoch分の学習を行なった。

事前検証では、元の decoder を直接 encoder に接続して学習する場合と比べ、CPT 後に接続した場合は、同じ合計音声長の学習条件で CER が約半分となった。この原因として、encoder に接続する前に、日本語や文字起こしテキストのフォーマットを学習していたことが考えられる。これにより、接続後の学習では、音声とテキストの対応付けに集中できた可能性がある。

### Encoder と decoder を接続した学習

CPT 後の encoder と decoder を audio adapter と gated cross attention で接続し、音声と文字起こしのペアで学習した。使用した Lambda Audio Corpus [12] は、NHK archive、技術系 podcast、ReazonSpeech から構築された 約8,593時間のコーパスである。NHK archive と podcast の文字起こしには Whisper large-v3-turbo の出力を LLM で修正したテキストを使用し、ReazonSpeech には元のデータセットに付属する文字起こしを使用した。学習用、validation 用、test 用の比率は件数ベースで約99%、0.5%、0.5%とし、学習用の約233万件、約8,507時間をモデルの更新に使用した。

最初の4,000ステップは encoder と decoder の事前学習済みパラメータを凍結し、audio adapter と gated cross attention のみを学習した。この段階で音声特徴と言語モデルの表現を対応付け、その後すべてのパラメータを更新した。Optimizer は AdamW、最大 learning rate は5 × 10⁻⁵、weight decay は0.01とし、2,000ステップの warmup 後に cosine schedule で learning rate を減少させた。BF16 mixed precision を使用し、音声長の近いサンプルをまとめることで attention 計算時の padding を削減した。

Lambda-audio-v6 の学習中の loss を Figure 2 に示す。validation loss は最初の6,000ステップで0.5037まで大きく低下した。その後も緩やかに低下し、54,000ステップ時点では0.2783となった。Training loss も低下しており、接続部分のみの学習からモデル全体の学習へ移行した後も、validation loss の改善が続いた。

![Figure 2: lambda-audio-v6 training loss](/Users/keisukemiyamoto/Project/LambdaAudio/docs/plots/lambda_audio_v6_loss.png)

Figure 2: lambda-audio-v6 の training loss と validation loss。横軸は optimizer の更新回数、縦軸は正解 token あたりの cross entropy を示す。Training loss は生の値と100記録点の移動平均を表示し、破線は encoder と decoder の凍結を解除する4,000ステップを示す。

### 評価条件とベンチマーク結果

評価には ReazonSpeech、Common Voice 25 の日本語サブセット、FLEURS の日本語サブセット を使用した。比較対象は Whisper large-v3-turbo、Kotoba-Whisper v2.2、ReazonSpeech の japanese-wav2vec2-base-rs35kh、Parakeet tdt_ctc-0.6b-ja、Cohere Transcribe 03-2026 とした。Batch size は16とし、lambda-audio では greedy decoding を用い、最大生成長を256 token に設定した。CER は評価前に生成結果、正例ともに NFKC 正規化を適用し、空白と句読点を除去した。正解が空になるサンプルや、正規化後の正解文字の75%超が日本語以外のサンプルは CER の計算から除外した。モデルの実行には,
RTX PRO 6000 GPU 1台を使用した。

Table 1 に各モデルの CER と文字起こしにかかった合計時間を示す。

| モデル | ReazonSpeech | Common Voice 25 ja | FLEURS ja |
| --- | ---: | ---: | ---: |
| lambda-audio-v6 | 10.75 / 121.55 | 23.87 / 167.84 | 6.05 / 20.75 |
| Whisper large-v3-turbo | 11.42 / 354.04 | 26.40 / 592.25 | 5.83 / 47.82 |
| Kotoba-Whisper v2.2 | 11.70 / 322.53 | 22.89 / 541.39 | 7.35 / 46.15 |
| ReazonSpeech wav2vec2 | 14.01 / 49.35 | 31.43 / 68.22 | 15.04 / 9.22 |
| Parakeet tdt_ctc-0.6b-ja | 10.62 / 55.10 | 19.99 / 82.28 | 5.94 / 9.74 |
| Cohere Transcribe 03-2026 | 6.45 / 176.00 | 21.58 / 227.42 | 4.36 / 30.77 |

Table 1: 3つの日本語 test データセットにおける CER と処理時間の比較。形式は CER (%) / 時間 (s)

![Figure 3: ReazonSpeech benchmark](/Users/keisukemiyamoto/Project/LambdaAudio/docs/plots/reazonspeech_test.png)

Figure 3: ReazonSpeech test における CER と処理時間の比較。

![Figure 4: Common Voice 25 Japanese benchmark](/Users/keisukemiyamoto/Project/LambdaAudio/docs/plots/common_voice_25_ja_test.png)

Figure 4: Common Voice 25 の日本語 test における CER と処理時間の比較。

![Figure 5: FLEURS Japanese benchmark](/Users/keisukemiyamoto/Project/LambdaAudio/docs/plots/fleurs_ja_test.png)

Figure 5: FLEURS の日本語 test における CER と処理時間の比較。

Figure 3〜5 は、Table 1 の結果を散布図で示したものである。横軸はデータセット全体の処理時間、縦軸は CER であり、左下に位置するほど短い処理時間で高い認識精度を達成している。緑の点は lambda-audio を示し、各モデル名の後の括弧内はパラメータ数を表す。

Figure 3 の ReazonSpeech では、lambda-audio の CER は10.75%であり、Whisper の11.42%、Kotoba-Whisper の11.70%より低かった。処理時間も121.55秒と、Whisper の354.04秒に対して約65.7%短く、散布図では両モデルより左下に位置している。一方、Cohere Transcribe は6.45%と最も低い CER を示し、Parakeet は lambda-audio と近い CER の10.62%を55.10秒で達成した。

Figure 4 の Common Voice では、lambda-audio の CER は23.87%であり、Whisper の26.40%より低かった。処理時間は167.84秒と、Whisper の592.25秒に対して約71.7%短かった。Kotoba-Whisper と Cohere Transcribe は lambda-audio より低い CER を示したが、処理時間は長かった。Parakeet は CER が19.99%、処理時間が82.28秒であり、lambda-audio より左下に位置している。今回の比較では Parakeet が最も低い CER を示した。

Figure 5 の FLEURS では、lambda-audio の CER は6.05%であり、Whisper の5.83%、Parakeet の5.94%と近い値となった。Whisper と比べて CER は0.22 percentage point 高かったが、処理時間は47.82秒から20.75秒へ約56.6%短縮した。また、Kotoba-Whisper に対しては CER と処理時間の両方で上回った。Cohere Transcribe は4.36%と最も低い CER を示したが、処理時間は lambda-audio より約48.3%長かった。

これらの結果から、lambda-audio は Whisper large-v3-turbo と比べ、3つのデータセットすべてで処理時間を短縮し、ReazonSpeech と Common Voice では CER も改善した。一方、Parakeet は3つのデータセットすべてで lambda-audio より低い CER と短い処理時間を示しており、精度と速度には依然として改善の余地があることがわかった。

## Conclusion (Summarize the topic, methodology, and advantages)



## References

[1] J.-B. Alayrac et al., [Flamingo: a Visual Language Model for Few-Shot Learning](https://arxiv.org/abs/2204.14198), 2022.

[2] A. Radford et al., [Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356), 2022.

[3] Z. Yao et al., [Zipformer: A faster and better encoder for automatic speech recognition](https://arxiv.org/abs/2310.11230), 2023.

[4] A. Amini et al., [LFM2 Technical Report](https://arxiv.org/abs/2511.23404), 2025.

[5] Y. Sasaki, [japanese-zipformer-base-k2](https://huggingface.co/reazon-research/japanese-zipformer-base-k2), 2025.

[6] k2-fsa, [icefall: Zipformer SSL pretraining recipe](https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/SSL/zipformer/pretrain.py).

[7] K. Miyamoto, [Japanese Zipformer Base CPT](https://huggingface.co/KeisukeMiyamoto/zipformer-cpt).

[8] K. Miyamoto, [Whisper Cluster Audio](https://huggingface.co/datasets/KeisukeMiyamoto/whisper-cluster-audio).

[9] Liquid AI, [LFM2.5-230M-Base](https://huggingface.co/LiquidAI/LFM2.5-230M-Base), 2026.

[10] K. Miyamoto, [LFM2.5-230M-CPT](https://huggingface.co/KeisukeMiyamoto/lfm2.5-230m-cpt).

[11] K. Miyamoto, [Japanese Audio Transcripts for Language Modeling](https://huggingface.co/datasets/KeisukeMiyamoto/audio-transcript).

[12] K. Miyamoto, [Lambda Audio Corpus](https://huggingface.co/datasets/KeisukeMiyamoto/lambda-audio-courpus).
