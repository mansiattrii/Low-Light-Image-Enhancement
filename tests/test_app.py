from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).parent.parent
ASSET = Path(__file__).parent / "assets" / "dark.png"


@pytest.fixture
def app(monkeypatch: pytest.MonkeyPatch) -> AppTest:
    # The demo reads config.yaml and the committed weights relative to the repository root.
    monkeypatch.chdir(ROOT)
    return AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()


def test_renders_without_upload(app: AppTest) -> None:
    assert not app.exception
    assert len(app.radio[0].options) == 4
    assert not app.get("image")


@pytest.mark.parametrize("method", ["zerodce", "gamma"])
@pytest.mark.parametrize("denoise", [False, True])
def test_upload_shows_result(app: AppTest, method: str, denoise: bool) -> None:
    app.radio[0].set_value(method)  # the method key; options holds the display labels
    app.checkbox[0].set_value(denoise)
    app.file_uploader[0].upload("dark.png", ASSET.read_bytes(), "image/png").run()
    assert not app.exception
    assert not app.error
    assert len(app.get("image")) == 2  # input and result
    assert app.get("download_button")


def test_invalid_upload_shows_error(app: AppTest) -> None:
    app.file_uploader[0].upload("broken.png", b"not an image", "image/png").run()
    assert not app.exception
    assert "Could not read" in app.error[0].value
