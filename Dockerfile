# BristolBot - deployment Dockerfile
#
# Drop-in replacement for the existing Dockerfile. Application code is
# untouched; the only real change is HOW PyTorch gets installed.
#
# Why it matters: requirements.txt has no torch in it, but
# langchain-huggingface -> sentence-transformers -> torch. The default
# PyPI wheel for torch bundles the NVIDIA CUDA runtime (multiple GB of
# libraries this app can never use, since EC2 t-series has no GPU).
# Installing the CPU-only wheel first satisfies the dependency and cuts
# the image down considerably.

FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.5.1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# The feedback log is a single file mounted from a named volume (see
# docker-compose.yml) so 👍/👎 responses survive a redeploy instead of
# vanishing when the container is recreated. Docker needs the path to
# already exist as a FILE at image-build time - otherwise, the first time
# it attaches an empty named volume to a path that doesn't exist yet, it
# creates a directory there instead, and Python's open(path, 'a') then
# fails with "Is a directory".
RUN touch feedback_log.csv

ENV PYTHONUNBUFFERED=1
ENV TOKENIZERS_PARALLELISM=false

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=4).status==200 else 1)"

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
