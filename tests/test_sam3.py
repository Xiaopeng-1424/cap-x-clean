"""Run SAM3 on the first frame of a video and save mask visualizations.

Edit the constants in the configuration section, start the SAM3 service, and run:

    uv run --no-sync --active capx/serving/launch_servers.py --profile default
    uv run python tests/test_sam3.py

The SAM3 client uses the service at ``http://127.0.0.1:8114`` by default.
Add short target descriptions to ``SAM3_PROMPTS``. One visualization is saved
for each prompt, using the prompt itself as the image filename.
"""

from pathlib import Path
from typing import Any

import imageio
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from capx.integrations.vision.sam3 import init_sam3

# ======================== Configuration (edit here) ========================
VIDEO_PATH = Path("/home/xpliu/projects/cap-x/outputs/libero-pro-skilled-20260627/libero_goal_swap/open_the_middle_drawer_of_the_cabinet/qwen3.5-122b-a10b-w8a8/run/trial_01_sandboxrc_0_reward_0.000_taskcompleted_0/video_turn_00.mp4")
SAM3_PROMPTS = (
    "stove knob",
    "stove heating coil",
    "cabinet top surface",
    "plate",
    "wine bottle",
)
OUTPUT_DIR = Path("tests/sam3_test")
MAX_RESULTS = 3
MASK_ALPHA = 0.5
# ===========================================================================

MASK_COLOR = np.array([30, 144, 255], dtype=np.float32)


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


def _result_score(result: dict[str, Any]) -> float:
    """Return a numeric SAM3 score, treating malformed values as zero."""
    try:
        return float(result.get("score", 0.0))
    except (TypeError, ValueError):
        return 0.0


def _add_box(
    ax: Any,
    box: list[float],
    score: float,
    color: str,
) -> None:
    """Draw one SAM3 bounding box and its confidence score."""
    x1, y1, x2, y2 = box
    ax.add_patch(
        patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=2,
            edgecolor=color,
            facecolor="none",
        )
    )
    ax.text(
        x1,
        y1,
        f"{score:.3f}",
        color="white",
        fontsize=8,
        bbox={"facecolor": color, "alpha": 0.8, "pad": 2},
    )


def _prompt_output_path(prompt: str) -> Path:
    """Build a filesystem-safe PNG path whose name comes from the prompt."""
    filename_stem = "".join(
        character if character.isprintable() and character not in "/\\" else "_"
        for character in prompt.strip()
    ).strip(" .")
    if not filename_stem:
        raise ValueError(f"Prompt does not contain a usable filename: {prompt!r}")
    return OUTPUT_DIR / f"{filename_stem}.png"


def run_sam3_prompts(
    image: Image.Image,
    prompts: tuple[str, ...],
) -> dict[str, list[dict[str, Any]]]:
    """Run every prompt and return results keyed by the original prompt."""
    sam3_segment_fn = init_sam3()
    results_by_prompt: dict[str, list[dict[str, Any]]] = {}

    for prompt in prompts:
        print(f"\nRunning SAM3 with prompt: {prompt!r}")
        results = sam3_segment_fn(image, prompt)
        results_by_prompt[prompt] = results
        print(f"SAM3 returned {len(results)} result(s)")

    return results_by_prompt


def save_visualization(
    image: Image.Image,
    prompt: str,
    results: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Save the original detections and the top SAM3 mask overlays in one grid."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    top_results = sorted(
        results,
        key=_result_score,
        reverse=True,
    )[:MAX_RESULTS]

    panel_count = max(1, len(top_results) + 1)
    fig, axes_grid = plt.subplots(
        1,
        panel_count,
        figsize=(5 * panel_count, 5),
        squeeze=False,
    )
    axes = axes_grid[0]

    axes[0].imshow(image)
    axes[0].set_title(f"Prompt: {prompt}\n{len(results)} result(s)")
    axes[0].set_axis_off()

    if not top_results:
        axes[0].text(
            0.5,
            0.5,
            "SAM3 returned no results",
            transform=axes[0].transAxes,
            ha="center",
            va="center",
            color="white",
            weight="bold",
            bbox={"boxstyle": "round,pad=0.5", "facecolor": "red", "alpha": 0.8},
        )
    else:
        for result in top_results:
            _add_box(
                axes[0],
                list(result["box"]),
                _result_score(result),
                color="red",
            )

    image_array = np.asarray(image, dtype=np.uint8)
    for index, result in enumerate(top_results, start=1):
        mask = np.asarray(result["mask"], dtype=bool).squeeze()
        if mask.shape != image_array.shape[:2]:
            raise ValueError(
                f"SAM3 mask {index} has shape {mask.shape}; "
                f"expected {image_array.shape[:2]}"
            )

        overlay = image_array.astype(np.float32).copy()
        overlay[mask] = (
            (1.0 - MASK_ALPHA) * overlay[mask] + MASK_ALPHA * MASK_COLOR
        )
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)

        score = _result_score(result)
        axes[index].imshow(overlay)
        _add_box(axes[index], list(result["box"]), score, color="yellow")
        axes[index].set_title(f"Result {index} | score={score:.3f}")
        axes[index].set_axis_off()

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.05, dpi=150)
    plt.close(fig)


def main() -> None:
    """Run all configured prompts and save one prompt-named visualization each."""
    prompts = tuple(prompt.strip() for prompt in SAM3_PROMPTS)
    if not prompts:
        raise ValueError("SAM3_PROMPTS must contain at least one prompt")
    if any(not prompt for prompt in prompts):
        raise ValueError("SAM3_PROMPTS must not contain empty prompts")
    if len(set(prompts)) != len(prompts):
        raise ValueError("SAM3_PROMPTS must not contain duplicate prompts")
    if MAX_RESULTS < 1:
        raise ValueError("MAX_RESULTS must be at least 1")
    if not 0.0 <= MASK_ALPHA <= 1.0:
        raise ValueError("MASK_ALPHA must be between 0 and 1")

    print(f"Reading first frame from: {VIDEO_PATH}")
    image = load_first_frame(VIDEO_PATH)
    print(f"First-frame size: {image.width}x{image.height}")

    results_by_prompt = run_sam3_prompts(image, prompts)
    print("\nResult summary:")
    for prompt, results in results_by_prompt.items():
        print(f"  {prompt!r}: {len(results)} result(s)")
        top_results = sorted(results, key=_result_score, reverse=True)[:MAX_RESULTS]
        for index, result in enumerate(top_results, start=1):
            print(
                f"    Result {index}: score={_result_score(result):.3f}, "
                f"box={result['box']}"
            )

        output_path = _prompt_output_path(prompt)
        save_visualization(image, prompt, results, output_path)
        print(f"  Saved visualization to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
