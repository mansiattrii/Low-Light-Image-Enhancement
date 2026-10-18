"""REST API for low-light enhancement.

Usage:
    uvicorn src.api:create_app --factory --host 0.0.0.0 --port 8000

Then open http://localhost:8000/docs for the interactive documentation. The config file is
``config.yaml`` unless the ``LLIE_CONFIG`` environment variable points elsewhere.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Annotated, Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from src.config import load_config
from src.data import decode_image, encode_image
from src.enhance import METHODS, enhance, get_enhancer

MethodName = Literal["gamma", "hist_eq", "clahe", "zerodce"]


def create_app(config_path: str | Path | None = None) -> FastAPI:
    """Build the API around one config; Zero-DCE is loaded here so the first request is fast."""
    config = load_config(config_path or os.environ.get("LLIE_CONFIG", "config.yaml"))
    max_bytes = int(config["inference"]["max_upload_mb"] * 1024 * 1024)
    get_enhancer("zerodce", config)

    app = FastAPI(
        title="Low-Light Image Enhancement",
        description="Brighten dark photos with Zero-DCE or classical baselines.",
        version="0.1.0",
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/methods")
    def methods() -> dict[str, object]:
        return {"methods": list(METHODS), "default": "zerodce"}

    # A plain (non-async) endpoint: FastAPI runs it in a worker thread, so a slow
    # enhancement does not block the server from answering other requests.
    @app.post(
        "/enhance",
        response_class=Response,
        responses={200: {"content": {"image/png": {}}, "description": "Enhanced image (PNG)"}},
    )
    def enhance_image(
        file: Annotated[UploadFile, File(description="Image file (PNG, JPEG, BMP, TIFF, ...)")],
        method: Annotated[MethodName, Form()] = "zerodce",
        denoise: Annotated[bool, Form(description="Bilateral filter after enhancement")] = False,
    ) -> Response:
        # Read one byte past the limit so oversized uploads are rejected without reading
        # them completely into memory.
        data = file.file.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise HTTPException(413, f"File larger than {config['inference']['max_upload_mb']} MB")
        try:
            image = decode_image(data)
        except ValueError:
            raise HTTPException(400, "Could not decode the file as an image") from None

        start = time.perf_counter()
        output = enhance(image, method, config, denoise)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        height, width = output.shape[:2]
        return Response(
            encode_image(output, ".png"),
            media_type="image/png",
            headers={
                "X-Processing-Time-Ms": f"{elapsed_ms:.1f}",
                "X-Output-Size": f"{width}x{height}",
            },
        )

    return app
