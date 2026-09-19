# LambdaAudio

## Purpose
- 日本語ASRモデルを構築する。
- 以下のモデルを参考にする。
  - OpenAI Whisper
  - Cohere Transcribe


## Coding Style
- For Python, comments may be written for each processing block (NOT line by line) in the following format:

    # ---------------------------------------------------------
    # Comment here in English (up to 3 lines)
    # ---------------------------------------------------------
    Some code ...

- Follow single responsible principal when create new files and directries.
- All function arguments and return values must have type annotations.
- Do NOT write doc string unless the user requests.
- Avoid create deep nesting by using method other than creating functions.
- NEVER write fallback logic, as it adds unnecessary complexity to the code.
- Use src/ to organize architecture.

## Scripts
- Use scripts/ to write one time code. This folder should be use only when the user clearly ask for.

## Testing
- Use pytest for testing your code. Write test code under tests/ directory.

## Code Quality

After modifying Python code, run:

```bash
pytest
ruff check . --fix
ruff format .
pyright
```

Keep changes automatically made by Ruff, including changes outside the immediate task scope.

## Libraries
- You can freely install necessary libraries using pip3.

## Other
- Please respond in Japanese.
- ALWAYS make your answer SHORT and CONCISE. 
- Always use HF_TOKEN in .env when you access Hugging Face.
