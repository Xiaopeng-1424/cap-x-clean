"""Run Molmo on the first frame of a video and save a point visualization.

Edit the constants in the configuration section, then run:

    uv run python tests/test_molmo_fireworks.py

`init_molmo()` automatically prefixes the target description with ``Point at``.
Therefore, ``MOLMO_PROMPT`` should normally describe the target object rather
than contain another ``Point at``/``Point to`` instruction.
"""

from collections import Counter
from pathlib import Path

import imageio
import matplotlib.pyplot as plt
from PIL import Image

from capx.integrations.vision.molmo import init_molmo

# ======================== Configuration (edit here) ========================
VIDEO_PATH = Path("/home/xpliu/projects/cap-x/outputs/libero-pro-skilled-20260627/libero_goal_swap/open_the_middle_drawer_of_the_cabinet/qwen3.5-122b-a10b-w8a8/run/trial_01_sandboxrc_0_reward_0.000_taskcompleted_0/video_turn_00.mp4")
MOLMO_PROMPT = "box-like cream cheese"
OUTPUT_IMAGE_PATH = Path("tests/molmo_comparasion/box-like cream cheese fireworks.png")
NUM_REPEATS = 1

MOLMO_MODEL_NAME = "molmo2-8b"
# ===========================================================================


def load_first_frame(video_path: Path) -> Image.Image:
    """Decode and return the first video frame as an RGB PIL image."""
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    try:
        frame_rgb = imageio.v3.imread(video_path, index=0, plugin="FFMPEG")
    except Exception as exc:
        raise RuntimeError(f"Could not decode the first frame of: {video_path}") from exc

    return Image.fromarray(frame_rgb).convert("RGB")


def save_visualization(
    image: Image.Image,
    prompt: str,
    points: list[tuple[int | None, int | None]],
    output_path: Path,
) -> None:
    """Draw all valid Molmo points on one image and save it."""
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

        # When several runs return exactly the same point, show its multiplicity.
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
            "Molmo returned no valid points",
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
        f"runs={len(points)}, valid={len(valid_points)}, "
        f"unique={unique_result_count}, consistent={is_consistent}"
    )
    ax.set_title(f"Prompt: {prompt}\n{result_summary}", fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def main() -> None:
    """Extract the first frame, query Molmo repeatedly, and save all points."""
    if not MOLMO_PROMPT.strip():
        raise ValueError("MOLMO_PROMPT must not be empty")
    if NUM_REPEATS < 1:
        raise ValueError("NUM_REPEATS must be at least 1")

    print(f"Reading first frame from: {VIDEO_PATH}")
    image = load_first_frame(VIDEO_PATH)
    print(f"First-frame size: {image.width}x{image.height}")

    molmo_point_fn = init_molmo(model_name=MOLMO_MODEL_NAME)

    points: list[tuple[int | None, int | None]] = []
    for run_index in range(1, NUM_REPEATS + 1):
        results = molmo_point_fn(image, [MOLMO_PROMPT])
        point = results.get(MOLMO_PROMPT, (None, None))
        points.append(point)
        print(f"Run {run_index:02d}/{NUM_REPEATS}: {point}")

    result_counts = Counter(points)
    print("\nResult frequencies:")
    for point, count in result_counts.items():
        print(f"  {point}: {count} time(s)")
    print(f"Consistent across all runs: {len(result_counts) == 1}")

    save_visualization(image, MOLMO_PROMPT, points, OUTPUT_IMAGE_PATH)
    print(f"Saved visualization to: {OUTPUT_IMAGE_PATH.resolve()}")


if __name__ == "__main__":
    main()
