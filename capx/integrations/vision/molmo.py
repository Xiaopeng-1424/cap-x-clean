from __future__ import annotations

import base64
import io
import os
import re
import time
from collections.abc import Callable
from typing import Any

import matplotlib.pyplot as plt
import requests
from PIL import Image

_PROC: Any | None = None  # kept for backward compatibility; unused with vLLM HTTP API
_MODEL: Any | None = None  # kept for backward compatibility; unused with vLLM HTTP API

# Override per process when needed; all project call sites use this default.
SERVICE_URL = os.environ.get("MOLMO_BASE_URL", "http://127.0.0.1:49188/v1")


def _parse_points(text: str) -> tuple[list[tuple[float, float]], float]:
    """Parse point coordinates from model text output.

    Supports multiple formats:
    - Molmo2: <points coords="obj_idx x y ...">label</points> (normalized 0-1000)
    - Molmo1: <point x="X" y="Y"> (normalized 0-100)
    - Legacy: <points x1=".." y1=".."> (normalized 0-100)
    - Fallback: plain "x, y" pairs (normalized 0-100)

    Args:
        text: Generated text potentially containing point tags.

    Returns:
        Tuple of (points, norm_scale) where points is a list of (x, y) tuples
        and norm_scale is the normalization scale (100.0 for Molmo1, 1000.0 for Molmo2).
    """
    try:
        # 1) Molmo2 format: <points coords="type obj_idx x y ...">label</points>
        coords_match = re.search(
            r'<points\s+coords\s*=\s*["\']([^"\']+)["\']',
            text,
            flags=re.IGNORECASE,
        )
        if coords_match:
            nums = [float(n) for n in coords_match.group(1).split()]
            points: list[tuple[float, float]] = []
            # Skip the type indicator, then parse (obj_idx, x, y) triplets.
            i = 1
            while i + 2 < len(nums):
                x, y = nums[i + 1], nums[i + 2]
                points.append((x, y))
                i += 3
            return points, 1000.0

        # 2) Parse one or more Molmo1 <point x=".." y=".."> tags.
        point_tags = re.findall(r"<point\b[^>]*>", text, flags=re.IGNORECASE)
        points = []
        for tag in point_tags:
            mx = re.search(
                r"\bx\s*=\s*['\"]([0-9]*\.?[0-9]+)['\"]",
                tag,
                flags=re.IGNORECASE,
            )
            my = re.search(
                r"\by\s*=\s*['\"]([0-9]*\.?[0-9]+)['\"]",
                tag,
                flags=re.IGNORECASE,
            )
            if mx and my:
                points.append((float(mx.group(1)), float(my.group(1))))
        if points:
            return [(x, y) for x, y in points if 0.0 <= x <= 100.0 and 0.0 <= y <= 100.0], 100.0

        # 3) Legacy <points x1=".." y1=".." ...> format.
        tag_match = re.search(r"<points\b[^>]*>", text, flags=re.IGNORECASE)
        if tag_match:
            source = tag_match.group(0)
            xs = {
                int(i): float(v)
                for i, v in re.findall(r"x(\d+)\s*=\s*['\"]([0-9]*\.?[0-9]+)['\"]", source)
            }
            ys = {
                int(i): float(v)
                for i, v in re.findall(r"y(\d+)\s*=\s*['\"]([0-9]*\.?[0-9]+)['\"]", source)
            }
            points = [(xs[i], ys[i]) for i in sorted(set(xs) & set(ys))]
            if points:
                return [(x, y) for x, y in points if 0.0 <= x <= 100.0 and 0.0 <= y <= 100.0], 100.0

        # 4) Fallback: parse plain "x, y" pairs anywhere in text.
        pairs = re.findall(r"([0-9]*\.?[0-9]+)\s*,\s*([0-9]*\.?[0-9]+)", text)
        points = [(float(x), float(y)) for x, y in pairs]
        return [(x, y) for x, y in points if 0.0 <= x <= 100.0 and 0.0 <= y <= 100.0], 100.0
    except (IndexError, TypeError, ValueError):
        return [], 100.0


def _overlay_and_save(
    image: Image.Image,
    points: list[tuple[float, float]],
    save_path: str,
    *,
    norm_scale: float = 100.0,
) -> None:
    """Overlay points on image and save figure.

    Args:
        image: PIL image the points refer to.
        points: Normalized coordinates ordered top-down, left-to-right.
        save_path: Path to save the visualization.
        norm_scale: Normalization scale (100.0 for Molmo1, 1000.0 for Molmo2).
    """
    if not points:
        return
    width, height = image.size
    xs = [x / norm_scale * width for x, _ in points]
    ys = [y / norm_scale * height for _, y in points]

    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.scatter(xs, ys, s=60, c="red", edgecolors="white", linewidths=1.5, zorder=3)
    for i, (x, y) in enumerate(zip(xs, ys, strict=True)):
        ax.text(x + 3, y + 3, str(i), color="yellow", fontsize=10, weight="bold", zorder=4)
    ax.set_axis_off()
    plt.tight_layout(pad=0)
    try:
        output_dir = os.path.dirname(save_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", pad_inches=0)
    except OSError as exc:
        raise RuntimeError(f"Could not save Molmo visualization to {save_path}") from exc
    finally:
        plt.close(fig)


def convert_to_pixel_coordinates(
    points: list[tuple[float, float]], image: Image.Image, norm_scale: float = 100.0
) -> list[tuple[int, int]]:
    """Convert normalized point coordinates to pixel coordinates.

    Args:
        points: List of (x, y) tuples in normalized coordinates.
        image: PIL image to get dimensions from.
        norm_scale: Normalization scale (100.0 for Molmo1, 1000.0 for Molmo2).
    """
    width, height = image.size
    try:
        xs = [int(x / norm_scale * width) for x, _ in points]
        ys = [int(y / norm_scale * height) for _, y in points]
    except (OverflowError, ValueError) as exc:
        raise ValueError("Molmo returned invalid point coordinates") from exc
    return list(zip(xs, ys, strict=True))


def _image_to_data_url(image: Image.Image) -> str:
    """Encode a PIL image as a PNG data URL suitable for OpenAI-compatible APIs."""
    try:
        with io.BytesIO() as buf:
            image.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    except OSError as exc:
        raise RuntimeError("Could not encode the Molmo input image") from exc
    return f"data:image/png;base64,{b64}"


def init_molmo(
    # model_name: str = "allenai/moldmo-2-8b:free", # OpenRouter
    # model_name: str = "allenai/Molmo2-O-7B",
    model_name: str = "molmo2-8b",  # served model id on the vLLM endpoint (see /v1/models)
    base_url: str = SERVICE_URL,
    api_key: str | None = None,
) -> Callable[[Image.Image, list[str] | None], dict[str, tuple[int | None, int | None]]]:
    """Initialize a detector that queries a vLLM OpenAI-compatible server.

    Args:
        model_name: The served model name (as registered in vLLM).
        base_url: Base URL of the OpenAI-compatible endpoint (no trailing slash beyond /v1).
        api_key: Optional API key; omit if the server does not require auth.

    Returns:
        A callable that takes an image and a list of object strings, and returns a
        mapping from object name to a pixel coordinate tuple (x, y). If a point
        could not be parsed, the value is (None, None).
    """
    # chat_url = f"{base_url.rstrip('/')}/v1/chat/completions" # SGLANG
    chat_url = f"{base_url.rstrip('/')}/chat/completions"  # vLLM
    session = requests.Session()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers.update({"Authorization": f"Bearer {api_key}"})

    def det_fn(
        image: Image.Image, objects: list[str] | None = None
    ) -> dict[str, tuple[int | None, int | None]]:
        """
        Args:
            image: PIL.Image.Image: The RGB image to process.
            objects: list[str]: The list of object queries to point to.

        Returns:
            dict[str, tuple[int | None, int | None]]: Pixel coordinates for each
            object query; (None, None) if parsing failed.
        """

        if not objects:
            return {}

        img_url = _image_to_data_url(image)
        all_points: dict[str, tuple[int | None, int | None]] = {}
        for obj in objects:
            prompt = f"Point at {obj}"
            payload = {
                "model": model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": img_url}},
                        ],
                    }
                ],
                "max_tokens": 1024,
                "temperature": 0.0,
                "stop": ["<|endoftext|>"],
            }

            generated_text = ""
            max_retries, backoff = 3, 1.0
            for attempt in range(max_retries):
                try:
                    resp = session.post(chat_url, json=payload, headers=headers, timeout=120)
                    resp.raise_for_status()
                    data = resp.json()
                    generated_text = (
                        data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    )
                    print(f"Generated text for '{obj}': {generated_text}")
                    break
                except Exception as e:  # noqa: BLE001
                    print(f"Request failed for '{obj}' (attempt {attempt + 1}/{max_retries}): {e}")
                    if attempt < max_retries - 1:
                        time.sleep(backoff * (2**attempt))
                    else:
                        generated_text = ""

            points, norm_scale = _parse_points(generated_text or "")
            if points:
                abs_coords = convert_to_pixel_coordinates(points, image, norm_scale)[0]
                # save_path = f"outputs/visualization/{obj}_points_{time.time()}.png"
                # _overlay_and_save(image, points, save_path, norm_scale=norm_scale)
                # print(f"Saved visualization to: {save_path}")
            else:
                abs_coords = (None, None)
                print("No points parsed from model output; skipping visualization.")
            all_points[obj] = abs_coords

        return all_points

    # register_object_detector(det_fn)
    return det_fn
