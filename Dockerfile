# Padarth Khoj web app - CPU image.   docker build -t padarthkhoj . && docker run -p 8000:8000 padarthkhoj
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PK_DATA_DIR=/data PK_HOST=0.0.0.0 PK_PORT=8000
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app

# CPU-only PyTorch first (keeps the image several GB smaller than the default CUDA wheels)
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt ./
COPY engine/requirements.txt engine/requirements.txt
RUN pip install -r requirements.txt

COPY engine ./engine
COPY server ./server
COPY web ./web

# bake the MACE-MH-1 model (57 MB) into the image so the first relaxation does not need to download it
RUN useradd -m padarth && mkdir -p /data && chown padarth /data
USER padarth
RUN python server/app.py --prefetch-model || echo "model prefetch failed; the app will offer to download it on first use"

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/info', timeout=4)"
CMD ["python", "server/app.py"]
