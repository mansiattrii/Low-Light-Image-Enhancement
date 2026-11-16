# CPU-only image for the API (default) or the Streamlit demo (APP=demo).
#   docker build -t low-light-enhancement .
#   docker run --rm -p 8000:8000 low-light-enhancement                   # API on :8000
#   docker run --rm -p 8501:8501 -e APP=demo low-light-enhancement       # demo on :8501
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# The default torch wheel on Linux bundles CUDA libraries (several GB). Install the CPU build
# of the pinned version first; requirements.txt then sees torch as already satisfied.
COPY requirements.txt .
RUN pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

# Only what inference needs: code, config, default weights, the tuned classical parameters and
# the demo's bundled sample image.
COPY config.yaml app.py ./
COPY src/ src/
COPY weights/zerodce.pth weights/
COPY results/tuned_params.yaml results/
COPY assets/sample_low.png assets/

RUN useradd --create-home --uid 1000 app
USER app

ENV APP=api
EXPOSE 8000 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health' if os.environ['APP'] == 'demo' else 'http://localhost:8000/health')" || exit 1
# exec replaces the shell, so the server receives Docker's stop signal directly.
CMD ["sh", "-c", "if [ \"$APP\" = demo ]; then exec streamlit run app.py --server.address=0.0.0.0 --server.port=8501 --server.headless=true; else exec uvicorn src.api:create_app --factory --host 0.0.0.0 --port 8000; fi"]
