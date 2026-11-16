"""Streamlit demo: upload a dark photo, pick a method, compare before and after.

Usage:
    streamlit run app.py
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import streamlit as st

from src.config import load_config
from src.data import decode_image, encode_image
from src.enhance import enhance, limit_size

# A low-light photo from the LOL eval15 split (image 79), so the demo works without an upload.
SAMPLE_IMAGE = Path(__file__).parent / "assets" / "sample_low.png"

METHOD_LABELS = {
    "zerodce": "Zero-DCE (deep curve estimation)",
    "gamma": "Gamma correction",
    "clahe": "CLAHE (adaptive histogram equalization)",
    "hist_eq": "Histogram equalization",
}


@st.cache_resource
def get_config() -> dict[str, Any]:
    return load_config("config.yaml")


def main() -> None:
    st.set_page_config(page_title="Low-Light Enhancement", layout="wide")
    st.title("Low-light image enhancement")
    st.write(
        "Upload a dark photo. Zero-DCE predicts a brightening curve for every pixel; the "
        "classical methods apply one fixed rule to the whole image."
    )

    config = get_config()
    max_mb = config["inference"]["max_upload_mb"]
    uploaded = st.file_uploader(
        f"Image (PNG, JPEG, BMP or TIFF, up to {max_mb} MB)",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
        max_upload_size=max_mb,
    )
    use_sample = st.checkbox(
        "No photo at hand? Use the bundled sample (a 600×400 night shot from the LOL dataset)",
        value=True,
        help="An uploaded photo always takes precedence over the sample.",
    )
    method = st.radio("Method", list(METHOD_LABELS), format_func=METHOD_LABELS.get, horizontal=True)
    denoise = st.checkbox(
        "Reduce noise (bilateral filter after enhancement)",
        help="Brightening also amplifies sensor noise. The filter removes it but softens fine "
        "texture.",
    )
    if uploaded is not None:
        name = Path(uploaded.name).stem
        data = uploaded.getvalue()
    elif use_sample:
        name = SAMPLE_IMAGE.stem
        data = SAMPLE_IMAGE.read_bytes()
    else:
        return

    try:
        image = decode_image(data)
    except ValueError:
        st.error("Could not read this file as an image.")
        return
    # enhance() applies the same size cap; doing it here too keeps the two images aligned.
    image = limit_size(image, config["inference"]["max_side"])

    start = time.perf_counter()
    with st.spinner("Enhancing..."):
        output = enhance(image, method, config, denoise)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    left, right = st.columns(2)
    left.image(image, caption="Input", width="stretch", clamp=True)
    right.image(
        output,
        caption=f"{METHOD_LABELS[method]} ({elapsed_ms:.0f} ms)",
        width="stretch",
        clamp=True,
    )
    st.download_button(
        "Download result (PNG)",
        encode_image(output, ".png"),
        file_name=f"{name}_{method}.png",
        mime="image/png",
    )


main()
