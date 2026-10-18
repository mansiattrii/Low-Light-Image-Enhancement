from pathlib import Path
from typing import get_args

import cv2
import numpy as np
import pytest
import torch
import yaml
from fastapi.testclient import TestClient

from src.api import MethodName, create_app
from src.data import decode_image
from src.enhance import METHODS
from src.zerodce import DCENet, init_weights

ASSET = Path(__file__).parent / "assets" / "dark.png"


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    torch.manual_seed(0)
    model = DCENet()
    init_weights(model)
    weights = tmp_path / "zerodce.pth"
    torch.save(model.state_dict(), weights)
    tuned = tmp_path / "tuned_params.yaml"
    tuned.write_text(
        yaml.safe_dump({"gamma": {"gamma": 0.4}, "clahe": {"clip_limit": 2.0, "tile_size": 8}})
    )
    config = {
        "paths": {"weights": str(weights), "tuned_params": str(tuned)},
        "zerodce": {"features": 32, "iterations": 8},
        "inference": {
            "max_side": 64,
            "max_upload_mb": 0.01,  # about 10 KB, so the size limit is easy to test
            "denoise": {"sigma_color": 1.0, "sigma_space": 2},
        },
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return TestClient(create_app(path))


def _png(image: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return encoded.tobytes()


def test_method_names_match_enhance() -> None:
    assert get_args(MethodName) == METHODS


def test_health_and_methods(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}
    body = client.get("/methods").json()
    assert body["methods"] == list(METHODS)
    assert body["default"] == "zerodce"


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("denoise", [False, True])
def test_enhance_returns_png(client: TestClient, method: str, denoise: bool) -> None:
    files = {"file": ("dark.png", ASSET.read_bytes(), "image/png")}
    response = client.post(
        "/enhance", files=files, data={"method": method, "denoise": str(denoise).lower()}
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.headers["x-output-size"] == "64x48"
    assert float(response.headers["x-processing-time-ms"]) >= 0
    output = decode_image(response.content)
    assert output.shape == (48, 64, 3)


def test_default_method_is_zerodce(client: TestClient) -> None:
    files = {"file": ("dark.png", ASSET.read_bytes(), "image/png")}
    default = client.post("/enhance", files=files).content
    zerodce = client.post("/enhance", files=files, data={"method": "zerodce"}).content
    assert default == zerodce


def test_large_images_are_downscaled(client: TestClient) -> None:
    image = np.zeros((40, 160), dtype=np.uint8)  # grayscale, compresses well below the limit
    files = {"file": ("wide.png", _png(image), "image/png")}
    response = client.post("/enhance", files=files, data={"method": "gamma"})
    assert response.status_code == 200
    assert response.headers["x-output-size"] == "64x16"


def test_rgba_input(client: TestClient) -> None:
    image = np.full((20, 30, 4), 40, dtype=np.uint8)
    files = {"file": ("rgba.png", _png(image), "image/png")}
    response = client.post("/enhance", files=files, data={"method": "clahe"})
    assert response.status_code == 200
    assert decode_image(response.content).shape == (20, 30, 3)


def test_rejects_non_image(client: TestClient) -> None:
    files = {"file": ("notes.txt", b"not an image", "text/plain")}
    response = client.post("/enhance", files=files)
    assert response.status_code == 400


def test_rejects_unknown_method(client: TestClient) -> None:
    files = {"file": ("dark.png", ASSET.read_bytes(), "image/png")}
    response = client.post("/enhance", files=files, data={"method": "magic"})
    assert response.status_code == 422


def test_rejects_large_upload(client: TestClient) -> None:
    files = {"file": ("big.png", b"\x00" * 20_000, "image/png")}
    response = client.post("/enhance", files=files)
    assert response.status_code == 413


def test_requires_a_file(client: TestClient) -> None:
    assert client.post("/enhance").status_code == 422
