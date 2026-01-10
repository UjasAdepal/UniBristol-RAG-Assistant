# 1. Base Image: Use a slim version of Python to keep file size down
FROM python:3.11-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Install System Dependencies
# 'build-essential' is often needed for installing heavy math libraries (numpy/pandas)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy the Recipe file first (Caching Strategy)
# This makes rebuilding faster if you only change your code, not your libraries
COPY requirements.txt .

# 5. Install Python Libraries
# --no-cache-dir reduces the image size
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy the Application Code
# We copy everything (.) to the container's folder (.)
COPY . .

# 7. Environment Setup
# Prevents Python from buffering stdout/stderr (so you see logs instantly)
ENV PYTHONUNBUFFERED=1
# Disable Tokenizer parallelism to prevent crashes (like we did in Python)
ENV TOKENIZERS_PARALLELISM=false

# 8. Expose the Port (Streamlit uses 8501 by default)
EXPOSE 8501

# 9. The Launch Command
# This runs your Streamlit app when the container starts
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]