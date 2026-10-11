((venv) ) ubuntu@0521-dca-csp6bk-prxmx160198:~/lambda-audio$ for model in lambda-audio whisper reazonspeech kotoba-whisper parake
et cohere-transcribe; do                                                                
  python3 -m src.benchmark \
    --model "$model" \
    --checkpoint outputs/training/model.ckpt \
    --input reazonspeech_test.parquet \
    --batch-size 16 \
    --no-show-table
done
Loading weights: 100%|█████████████████████████████████████████████████████████████████████| 729/729 [00:00<00:00, 12245.92it/s]
Loading weights: 100%|██████████████████████████████████████████████████████████████████████| 132/132 [00:00<00:00, 4334.59it/s]
Inference ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 5263/5263 CER: 10.75% Skipped: 2 0:00:00
Model name: lambda-audio
CER: 10.75%
Elapsed time: 121.55 seconds
Loading weights: 100%|██████████████████████████████████████████████████████████████████████| 587/587 [00:00<00:00, 4159.23it/s]
Inference ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 5263/5263 CER: 11.42% Skipped: 2 0:00:00
Model name: whisper
CER: 11.42%
Elapsed time: 354.04 seconds
Loading weights: 100%|██████████████████████████████████████████████████████████████████████| 213/213 [00:00<00:00, 8608.72it/s]
Inference ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 5263/5263 CER: 14.01% Skipped: 2 0:00:00
Model name: reazonspeech
CER: 14.01%
Elapsed time: 49.35 seconds
Loading weights: 100%|██████████████████████████████████████████████████████████████████████| 539/539 [00:00<00:00, 9466.29it/s]
Inference ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 5263/5263 CER: 11.70% Skipped: 2 0:00:00
Model name: kotoba-whisper
CER: 11.70%
Elapsed time: 322.53 seconds
Inference ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 5263/5263 CER: 10.62% Skipped: 2 0:00:00
Model name: parakeet
CER: 10.62%
Elapsed time: 55.10 seconds
Loading weights: 100%|████████████████████████████████████████████████████████████████████| 2150/2150 [00:00<00:00, 4087.27it/s]
Inference ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 5263/5263 CER: 6.45% Skipped: 2 0:00:00
Model name: cohere-transcribe
CER: 6.45%
Elapsed time: 176.00 seconds