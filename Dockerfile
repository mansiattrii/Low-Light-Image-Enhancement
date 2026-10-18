# CPU-only image for the API (default) or the Streamlit demo.
#   docker build -t low-light-enhancement .
#   docker run --rm -p 8000:8000 low-light-enhancement                     # API on :8000
#   docker run --rm -p 8501:8501 low-light-enhancement streamlit run app.py --server.address=0.0.0.0
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

# Only what inference needs: code, config, default weights and the tuned classical parameters.
COPY config.yaml app.py ./
COPY src/ src/
COPY weights/zerodce.pth weights/
COPY results/tuned_params.yaml results/

RUN useradd --create-home --uid 1000 app
USER app

EXPOSE 8000 8501
# Checks the API; when the container runs the Streamlit demo instead it reports "unhealthy".
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1
CMD ["uvicorn", "src.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
