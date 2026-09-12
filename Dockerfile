# BristolBot - deployment Dockerfile
#
# Drop-in replacement for the existing Dockerfile. Application code is
# untouched; the only real changes are HOW PyTorch gets installed and how
# the feedback log survives a redeploy.
#
# Why the PyTorch step matters: requirements.txt has no torch in it, but
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

# Making feedback_log.csv survive a redeploy, take two.
#
# The first attempt mounted a named volume directly onto the file path
# itself. That's the wrong shape for Docker: when a named volume is empty
# and gets attached to a path that doesn't already exist as a file *inside
# that volume*, Docker creates a directory there - it does not turn into a
# file to match. The mount then fails outright, because a file can't bind
# onto a directory. That's exactly what broke the last deploy.
#
# Every real Docker Compose volume - including hf_cache and caddy_data
# already working correctly in this same file - mounts onto a directory.
# That's the one genuinely reliable shape. So: create a small data
# directory, and point feedback_log.csv at a file inside it via a plain
# filesystem symlink. app.py and config.py are completely unaware of any
# of this - they still just open() "feedback_log.csv" relative to /app,
# exactly as they always have. The symlink is what quietly redirects that
# into the persisted volume.
RUN mkdir -p /app/data && ln -s data/feedback_log.csv feedback_log.csv

ENV PYTHONUNBUFFERED=1
ENV TOKENIZERS_PARALLELISM=false

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=4).status==200 else 1)"

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
