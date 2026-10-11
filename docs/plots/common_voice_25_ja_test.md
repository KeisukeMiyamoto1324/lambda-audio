((venv) ) ubuntu@0521-dca-csp6bk-prxmx160198:~/lambda-audio$ for model in lambda-audio whisper reazonspeech kotoba-whisper parakeet cohere-transcribe; do   python3 -m src.benchmark     --model "$model"     --checkpoint outputs/training/model.ckpt     --input common_voice_25_ja_test.parquet     --batch-size 16     --no-show-table; done
Loading weights: 100%|█████████████████████████████████████████████████████████████████████| 729/729 [00:00<00:00, 10838.49it/s]
Loading weights: 100%|██████████████████████████████████████████████████████████████████████| 132/132 [00:00<00:00, 3296.53it/s]
Inference ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 9019/9019 CER: 23.87% Skipped: 30 0:00:00
Model name: lambda-audio
CER: 23.87%
Elapsed time: 167.84 seconds
Loading weights: 100%|██████████████████████████████████████████████████████████████████████| 587/587 [00:00<00:00, 4078.49it/s]
Inference ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 9019/9019 CER: 26.40% Skipped: 30 0:00:00
Model name: whisper
CER: 26.40%
Elapsed time: 592.25 seconds
Loading weights: 100%|█████████████████████████████████████████████████████████████████████| 213/213 [00:00<00:00, 10386.65it/s]
Inference ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 9019/9019 CER: 31.43% Skipped: 30 0:00:00
Model name: reazonspeech
CER: 31.43%
Elapsed time: 68.22 seconds
Loading weights: 100%|█████████████████████████████████████████████████████████████████████| 539/539 [00:00<00:00, 11823.72it/s]
Inference ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 9019/9019 CER: 22.89% Skipped: 30 0:00:00
Model name: kotoba-whisper
CER: 22.89%
Elapsed time: 541.39 seconds
Inference ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 9019/9019 CER: 19.99% Skipped: 30 0:00:00
Model name: parakeet
CER: 19.99%
Elapsed time: 82.28 seconds
Loading weights: 100%|████████████████████████████████████████████████████████████████████| 2150/2150 [00:00<00:00, 5049.67it/s]
Inference ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 9019/9019 CER: 21.58% Skipped: 30 0:00:00
Model name: cohere-transcribe
CER: 21.58%
Elapsed time: 227.42 seconds