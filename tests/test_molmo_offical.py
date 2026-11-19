"""Compare Molmo2 using the official Hugging Face Transformers backend.

This script mirrors ``tests/test_molmo.py``: it extracts the first video frame,
runs the same pointing prompt repeatedly, and overlays every returned point on
one image. Unlike ``test_molmo.py``, it loads ``allenai/Molmo2-8B`` locally via
the official Transformers implementation instead of calling the vLLM server.

The first run downloads a large model. ``device_map="auto"`` also requires the
``accelerate`` package.
"""

from collections import Counter
from importlib.util import find_spec
import math
from pathlib import Path
import re
from typing import Any

import imageio
import matplotlib.pyplot as plt
from PIL import Image
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ======================== Configuration (edit here) ========================
VIDEO_PATH = PROJECT_ROOT / (
    "outputs/libero-pro-skilled-20260627/libero_goal_swap/"
    "put_the_cream_cheese_in_the_bowl/qwen3.5-122b-a10b-w8a8/run/"
    "trial_01_sandboxrc_0_reward_0.000_taskcompleted_0/video_turn_00.mp4"
)
MOLMO_PROMPT = "The small rectangular box in the upper right corner of the table"
OUTPUT_IMAGE_PATH = PROJECT_ROOT / (
    "tests/molmo_test2/"
    "The small rectangular box in the upper right corner of the table_official.png"
)
NUM_REPEATS = 1

MODEL_ID = "allenai/Molmo2-8B"  # May also be changed to a local model directory.
MODEL_DTYPE = "auto"  # Official default is FP32; use "bfloat16" if memory is limited.
DEVICE_MAP = "auto"
MAX_NEW_TOKENS = 1024
# ===========================================================================

_COORDS_PATTERN = re.compile(
    r'<points\s+coords\s*=\s*["\']([^"\']+)["\']',
    flags=re.IGNORECASE,
)


def load_first_frame(video_path: Path) -> Image.Image:
    """Decode and return the first video frame as an RGB PIL image."""
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    try:
        reader: Any = imageio.get_reader(str(video_path))
        with reader:
            frame_rgb = reader.get_data(0)
    except Exception as exc:
        raise RuntimeError(f"Could not decode the first frame of: {video_path}") from exc

    return Image.fromarray(frame_rgb).convert("RGB")


def parse_first_point(
    generated_text: str,
    image: Image.Image,
) -> tuple[int | None, int | None]:
    """Parse the first Molmo2 0-1000 point and convert it to pixel coordinates."""
    match = _COORDS_PATTERN.search(generated_text)
    if match is None:
        return (None, None)

    try:
        numbers = [float(value) for value in match.group(1).split()]
    except ValueError:
        return (None, None)

    # Molmo2 format: image_id, then one or more (object_id, x, y) triplets.
    for index in range(1, len(numbers) - 2, 3):
        x_normalized = numbers[index + 1]
        y_normalized = numbers[index + 2]
        if not (
            math.isfinite(x_normalized)
            and math.isfinite(y_normalized)
            and 0 <= x_normalized <= 1000
            and 0 <= y_normalized <= 1000
        ):
            continue

        try:
            pixel_x = int(x_normalized / 1000 * image.width)
            pixel_y = int(y_normalized / 1000 * image.height)
        except (OverflowError, ValueError):
            continue
        pixel_x = min(max(pixel_x, 0), image.width - 1)
        pixel_y = min(max(pixel_y, 0), image.height - 1)
        return (pixel_x, pixel_y)

    return (None, None)


def load_official_model() -> tuple[Any, Any]:
    """Load the official processor and Transformers model once."""
    if DEVICE_MAP == "auto" and find_spec("accelerate") is None:
        raise RuntimeError(
            "DEVICE_MAP='auto' requires accelerate. Install it with: "
            "uv pip install accelerate"
        )

    print(f"Loading official model: {MODEL_ID}")
    print(f"dtype={MODEL_DTYPE}, device_map={DEVICE_MAP}")
    try:
        processor: Any = AutoProcessor.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            dtype=MODEL_DTYPE,
            device_map=DEVICE_MAP,
        )
        model: Any = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            dtype=MODEL_DTYPE,
            device_map=DEVICE_MAP,
        )
    except Exception as exc:
        raise RuntimeError(
            "Could not load the official Molmo2-8B model. Check available RAM/GPU "
            "memory, Hugging Face access, and the configured MODEL_ID."
        ) from exc

    model.eval()
    print(f"Model loaded on: {model.device}; dtype: {model.dtype}")
    return processor, model


def prepare_inputs(
    processor: Any,
    model: Any,
    image: Image.Image,
    target_description: str,
) -> dict[str, Any]:
    """Build official chat-template inputs using the same prompt as the vLLM test."""
    full_prompt = f"Point at {target_description}"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": full_prompt},
                {"type": "image", "image": image},
            ],
        }
    ]

    try:
        model_inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
    except Exception as exc:
        raise RuntimeError("Could not preprocess the image and prompt") from exc

    return {
        key: value.to(model.device) if hasattr(value, "to") else value
        for key, value in model_inputs.items()
    }


def generate_point_text(
    processor: Any,
    model: Any,
    model_inputs: dict[str, Any],
) -> str:
    """Run one independent greedy generation and return only newly generated text."""
    input_length = model_inputs["input_ids"].shape[1]
    with torch.inference_mode():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    generated_tokens = generated_ids[0, input_length:]
    return processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)


def save_visualization(
    image: Image.Image,
    prompt: str,
    points: list[tuple[int | None, int | None]],
    output_path: Path,
) -> None:
    """Draw all valid official-model points on one image and save it."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    valid_points: list[tuple[int, int]] = []
    for x, y in points:
        if x is not None and y is not None:
            valid_points.append((x, y))

    width, height = image.size
    fig, ax = plt.subplots(
        figsize=(max(width / 100, 6), max(height / 100, 4)),
        dpi=100,
    )
    ax.imshow(image)
    ax.set_axis_off()

    if valid_points:
        xs = [x for x, _ in valid_points]
        ys = [y for _, y in valid_points]
        ax.scatter(
            xs,
            ys,
            s=30,
            c="red",
            edgecolors="white",
            linewidths=0.5,
            alpha=0.8,
            zorder=3,
        )
        for (x, y), count in Counter(valid_points).items():
            if count > 1:
                ax.annotate(
                    f"x{count}",
                    (x, y),
                    xytext=(5, 5),
                    textcoords="offset points",
                    color="red",
                    fontsize=8,
                    weight="bold",
                    zorder=4,
                )
    else:
        ax.text(
            0.5,
            0.5,
            "Official Molmo2 returned no valid points",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="white",
            weight="bold",
            bbox={"boxstyle": "round,pad=0.5", "facecolor": "red", "alpha": 0.8},
        )

    unique_result_count = len(set(points))
    is_consistent = bool(points) and unique_result_count == 1
    result_summary = (
        f"official Transformers | runs={len(points)}, valid={len(valid_points)}, "
        f"unique={unique_result_count}, consistent={is_consistent}"
    )
    ax.set_title(f"Prompt: {prompt}\n{result_summary}", fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def main() -> None:
    """Run repeated official Transformers inference and save all returned points."""
    if not MOLMO_PROMPT.strip():
        raise ValueError("MOLMO_PROMPT must not be empty")
    if NUM_REPEATS < 1:
        raise ValueError("NUM_REPEATS must be at least 1")

    print(f"Reading first frame from: {VIDEO_PATH}")
    image = load_first_frame(VIDEO_PATH)
    print(f"First-frame size: {image.width}x{image.height}")

    processor, model = load_official_model()
    model_inputs = prepare_inputs(processor, model, image, MOLMO_PROMPT)

    points: list[tuple[int | None, int | None]] = []
    for run_index in range(1, NUM_REPEATS + 1):
        generated_text = generate_point_text(processor, model, model_inputs)
        point = parse_first_point(generated_text, image)
        points.append(point)
        print(f"Run {run_index:02d}/{NUM_REPEATS}: {point}")
        print(f"  Generated text: {generated_text}")

    result_counts = Counter(points)
    print("\nResult frequencies:")
    for point, count in result_counts.items():
        print(f"  {point}: {count} time(s)")
    print(f"Consistent across all runs: {len(result_counts) == 1}")

    save_visualization(image, MOLMO_PROMPT, points, OUTPUT_IMAGE_PATH)
    print(f"Saved visualization to: {OUTPUT_IMAGE_PATH.resolve()}")


if __name__ == "__main__":
    main()
