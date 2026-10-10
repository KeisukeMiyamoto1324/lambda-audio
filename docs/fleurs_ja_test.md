((venv) ) ubuntu@0521-dca-csp6bk-prxmx160198:~/lambda-audio$ for model in lambda-audio whisper reazonspeech kotoba-whisper parakeet cohere-transcribe; do   python3 -m src.benchmark     --model "$model"     --checkpoint outputs/training/model.ckpt     --input fleurs_ja_test.parquet     --batch-size 16     --no-show-table; done
Loading weights: 100%|██████████████████████████████████████████████████████████████████████| 729/729 [00:00<00:00, 8979.85it/s]
Loading weights: 100%|██████████████████████████████████████████████████████████████████████| 132/132 [00:00<00:00, 2456.62it/s]
Inference ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 650/650 CER: 6.05% Skipped: 0 0:00:00
Model name: lambda-audio
CER: 6.05%
Elapsed time: 20.75 seconds
Loading weights: 100%|██████████████████████████████████████████████████████████████████████| 587/587 [00:00<00:00, 2442.94it/s]
Inference ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 650/650 CER: 5.83% Skipped: 0 0:00:00
Model name: whisper
CER: 5.83%
Elapsed time: 47.82 seconds
mkdir -p failed for path /home/ubuntu/.config/matplotlib: [Errno 13] Permission denied: '/home/ubuntu/.config/matplotlib'
Matplotlib created a temporary cache directory at /tmp/matplotlib-oni_gi5i because there was an issue with the default path (/home/ubuntu/.config/matplotlib); it is highly recommended to set the MPLCONFIGDIR environment variable to a writable directory, in particular to speed up the import of Matplotlib and to better support multiprocessing.
Loading weights: 100%|██████████████████████████████████████████████████████████████████████| 213/213 [00:00<00:00, 9179.51it/s]
Inference ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 650/650 CER: 15.04% Skipped: 0 0:00:00
Model name: reazonspeech
CER: 15.04%
Elapsed time: 9.22 seconds
mkdir -p failed for path /home/ubuntu/.config/matplotlib: [Errno 13] Permission denied: '/home/ubuntu/.config/matplotlib'
Matplotlib created a temporary cache directory at /tmp/matplotlib-_a9kaeg5 because there was an issue with the default path (/home/ubuntu/.config/matplotlib); it is highly recommended to set the MPLCONFIGDIR environment variable to a writable directory, in particular to speed up the import of Matplotlib and to better support multiprocessing.
Loading weights: 100%|█████████████████████████████████████████████████████████████████████| 539/539 [00:00<00:00, 10714.56it/s]
Inference ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 650/650 CER: 7.35% Skipped: 0 0:00:00
Model name: kotoba-whisper
CER: 7.35%
Elapsed time: 46.15 seconds
Inference ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 650/650 CER: 5.94% Skipped: 0 0:00:00
Model name: parakeet
CER: 5.94%
Elapsed time: 9.74 seconds
processor_config.json: 100%|███████████████████████████████████████████████████████████████████| 131/131 [00:00<00:00, 1.41MB/s]
preprocessor_config.json: 100%|████████████████████████████████████████████████████████████████| 420/420 [00:00<00:00, 4.72MB/s]
config.json: 100%|█████████████████████████████████████████████████████████████████████████| 4.00k/4.00k [00:00<00:00, 31.1MB/s]
tokenizer_config.json: 100%|████████████████████████████████████████████████████████████████| 48.1k/48.1k [00:00<00:00, 222MB/s]
tokenizer.json: 100%|██████████████████████████████████████████████████████████████████████| 1.82M/1.82M [00:00<00:00, 79.2MB/s]
tokenizer.model: downloading bytes: ████████████████████████████████████████████████████████████████████████|  405kB, 40.3kB/s  
tokenizer.model: reconstructing file: 100%|████████████████████████████████████████████████████████|  493kB /  493kB, 49.1kB/s  
special_tokens_map.json: 100%|█████████████████████████████████████████████████████████████| 4.09k/4.09k [00:00<00:00, 30.6MB/s]
model.safetensors: downloading bytes: ██████████████████████████████████████████████████████████████████████| 3.53GB,  257MB/s  
model.safetensors: reconstructing file: 100%|██████████████████████████████████████████████████████| 4.13GB / 4.13GB,  313MB/s  
Loading weights: 100%|████████████████████████████████████████████████████████████████████| 2150/2150 [00:00<00:00, 4977.16it/s]
generation_config.json: 100%|██████████████████████████████████████████████████████████████████| 234/234 [00:00<00:00, 2.31MB/s]
Inference ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 650/650 CER: 4.36% Skipped: 0 0:00:00
Model name: cohere-transcribe
CER: 4.36%
Elapsed time: 30.77 seconds