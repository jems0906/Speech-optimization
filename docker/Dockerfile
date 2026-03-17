FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

WORKDIR /app

# Base image ships torch; install remaining runtime and ASR deps.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir \
        "torchaudio>=2.3" \
        "transformers>=4.42" \
        "accelerate>=0.33"

COPY src ./src
COPY configs ./configs

# /app must be on PYTHONPATH so `from src.<pkg>` resolves correctly.
ENV PYTHONPATH=/app
ENV ASR_BACKEND=transformers
ENV ASR_MODEL_ID=openai/whisper-small

EXPOSE 8000

CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
