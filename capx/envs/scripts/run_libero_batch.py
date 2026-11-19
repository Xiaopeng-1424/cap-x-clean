import os
# Force weights_only=False for PyTorch loading of legacy files
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
import sys
import shutil
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from typing import Literal
import yaml
import pathlib

import tyro  # type: ignore[import-not-found]
from capx.envs.launch import LaunchArgs
from capx.envs.launch import main as launch_main

# Import Libero to discover tasks
try:
    from libero import benchmark  # type: ignore[import-not-found]
except ImportError:
    print("Error: Libero not found. Make sure it is installed and PYTHONPATH is set.")
    sys.exit(1)


@dataclass
class LiberoBatchLaunchArgs:
    """Command-line arguments for automated Libero batch execution."""

    # Base configuration file
    base_config_path: str = "env_configs/libero/franka_libero_cap_agent0.yaml"

    # Suites to run
    suites: list[str] = field(
        default_factory=lambda: [
            "libero_object_swap",
            "libero_object_task",
            "libero_goal_swap",
            "libero_goal_task",
            "libero_spatial_swap",
            "libero_spatial_task",
        ]
    )

    # Optional task IDs to run within each suite (all tasks by default)
    task_ids: list[int] | None = None

    # Coding models and OpenAI-compatible endpoint.
    models: list[str] = field(default_factory=lambda: ["deepseek-v4-flash-s1"])
    server_url: str = "http://183.230.173.202:36789/v1/chat/completions"

    # Output directory base
    output_dir: str = "./outputs/libero_batch_run"

    # Other LaunchArgs overrides
    temperature: float = 1.0
    max_tokens: int = 2**15
    reasoning_effort: str = "medium"
    # Optional thinking override for both coding and visual differencing models.
    enable_thinking: bool | None = None
    # Qwen3.8 effort for both coding and visual differencing; other models ignore it.
    thinking_effort: Literal["low", "medium", "xhigh"] = "low"
    api_key: str | None = None
    use_visual_feedback: bool | None = None
    use_img_differencing: bool | None = None
    use_legacy_multi_turn_decision_prompt: bool | None = None
    total_trials: int | None = None
    num_workers: int | None = None
    record_video: bool | None = None
    debug: bool = False
    use_oracle_code: bool | None = None
    use_parallel_ensemble: bool | None = None
    use_multimodel: bool | None = None
    visual_differencing_model: str | None = "Qwen3-VL-235B-A22B"
    visual_differencing_model_server_url: str | None = (
        "http://183.230.173.202:36789/v1/chat/completions"
    )
    visual_differencing_model_api_key: str | None = None


def main(args: LiberoBatchLaunchArgs) -> None:
    benchmark_dict = benchmark.get_benchmark_dict()
    
    # Load base configuration
    if not os.path.exists(args.base_config_path):
        print(f"Error: Base config file not found: {args.base_config_path}")
        sys.exit(1)
        
    try:
        with open(args.base_config_path, "r") as f:
            base_config = yaml.safe_load(f)
    except (OSError, yaml.YAMLError) as e:
        print(f"Error: Could not load base config '{args.base_config_path}': {e}")
        sys.exit(1)

    if not isinstance(base_config, dict):
        print(f"Error: Base config must contain a YAML mapping: {args.base_config_path}")
        sys.exit(1)

    env_config = base_config.get("env")
    cfg_config = env_config.get("cfg") if isinstance(env_config, dict) else None
    if not isinstance(cfg_config, dict):
        print(
            "Error: Base config must contain an 'env.cfg' mapping: "
            f"{args.base_config_path}"
        )
        sys.exit(1)

    tasks_to_run = []
    
    print(f"Collecting tasks for suites: {args.suites}")
    
    for suite_name in args.suites:
        if suite_name not in benchmark_dict:
            print(f"Warning: Suite '{suite_name}' not found in Libero benchmarks.")
            continue
            
        task_suite = benchmark_dict[suite_name]()
        num_tasks = task_suite.n_tasks
        
        print(f"Suite {suite_name} has {num_tasks} tasks.")
        
        task_ids = range(num_tasks) if args.task_ids is None else args.task_ids
        for task_id in task_ids:
            if task_id < 0 or task_id >= num_tasks:
                print(
                    f"Warning: Task ID {task_id} is out of range for suite "
                    f"'{suite_name}' (0-{num_tasks - 1})."
                )
                continue

            task = task_suite.get_task(task_id)
            task_name = task.name
            
            # Determine number of initial states (trials)
            try:
                # TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD supports legacy Libero checkpoints.
                init_states = task_suite.get_task_init_states(task_id)
                assert init_states is not None, f"No initial states found for task {task_name}"
                num_trials = len(init_states)
            except Exception as e:
                print(f"Warning: Could not determine init states for {task_name}, defaulting to 50. Error: {e}")
                num_trials = 50

            tasks_to_run.append((suite_name, task_id, task_name, num_trials))

    print(f"Total tasks to run: {len(tasks_to_run)}")
    
    total_runs = len(args.models) * len(tasks_to_run)
    print(f"Total experimental runs (models x tasks): {total_runs}")
    
    experiment_idx = 1
    failed_runs = []
    
    for model in args.models:
        print(f"\n{'=' * 80}")
        print(f"Running model: {model}")
        print(f"{'=' * 80}")
        
        for suite_name, task_id, task_name, num_trials in tasks_to_run:
            print(f"\n{'-' * 80}")
            print(f"Running Experiment {experiment_idx}/{total_runs}")
            print(f"Model: {model}")
            print(f"Suite: {suite_name}, Task ID: {task_id}, Task Name: {task_name}")
            print(f"Trials: {num_trials}")
            print(f"{'-' * 80}\n")
            
            # Construct config from base template
            config = base_config.copy()
            
            # Deep copy to ensure we don't modify the shared base_config for subsequent runs
            import copy
            config = copy.deepcopy(base_config)

            # Inject task details
            # We overwrite the 'low_level' entry to be a dictionary definition for FrankaLiberoEnv
            # instead of the string reference (e.g. 'franka_libero_pick_place_low_level')
            
            # Check if env/cfg/privileged exists, use it if so
            is_privileged = config.get("env", {}).get("cfg", {}).get("privileged", False)
            
            config["env"]["cfg"]["low_level"] = {
                "_target_": "capx.envs.simulators.libero.FrankaLiberoEnv",
                "suite_name": suite_name,
                "task_id": task_id,
                "privileged": is_privileged,
                "max_steps": 8000,
                "seed": None,
                "enable_render": True,
                "viser_debug": False
            }
            
            # Set trials count dynamically
            config["trials"] = num_trials
            
            # Customize output directory
            # Structure: output_dir/suite_name/task_name/run
            # launch.py will append model_name automatically to the last component if we rely on its default behavior
            # But we want: output_dir/suite/task/model/run (or similar)
            
            # To get output_dir/suite/task/model/run:
            # We set config["output_dir"] = .../suite/task/run
            # launch.py splits: [..., suite, task, run]
            # Inserts model at -1: [..., suite, task, model, run]
            # Result: .../suite/task/model/run
            
            config_dir_base = os.path.join(os.path.abspath(args.output_dir), suite_name, task_name)
            config["output_dir"] = os.path.join(config_dir_base, "run")
            
            # Save config permanently to the run directory (parent of "run" subfolder effectively)
            config_filename = "config.yaml"
            config_path = os.path.join(config_dir_base, config_filename)

            try:
                os.makedirs(config_dir_base, exist_ok=True)
                with open(config_path, "w") as f:
                    yaml.dump(config, f)
            except (OSError, yaml.YAMLError) as e:
                print(f"ERROR saving config for {suite_name}/{task_name}: {e}")
                failed_runs.append((model, suite_name, task_name))
                experiment_idx += 1
                continue

            # Create LaunchArgs
            launch_args = LaunchArgs(
                config_path=config_path,
                server_url=args.server_url,
                model=model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                reasoning_effort=args.reasoning_effort,
                enable_thinking=args.enable_thinking,
                thinking_effort=args.thinking_effort,
                api_key=args.api_key,
                use_visual_feedback=args.use_visual_feedback,
                use_img_differencing=args.use_img_differencing,
                use_legacy_multi_turn_decision_prompt=args.use_legacy_multi_turn_decision_prompt,
                total_trials=args.total_trials,
                num_workers=args.num_workers,
                record_video=args.record_video,
                # We do NOT pass output_dir here to avoid the logic that uses config_stem
                output_dir=None, 
                debug=args.debug,
                use_oracle_code=args.use_oracle_code,
                use_parallel_ensemble=args.use_parallel_ensemble,
                use_multimodel=args.use_multimodel,
                visual_differencing_model=args.visual_differencing_model,
                visual_differencing_model_server_url=args.visual_differencing_model_server_url,
                visual_differencing_model_api_key=args.visual_differencing_model_api_key,
            )
            
            try:
                launch_main(launch_args)
            except Exception as e:
                print(f"\nERROR running {suite_name}/{task_name} with model {model}: {e}")
                traceback.print_exc()
                failed_runs.append((model, suite_name, task_name))
            
            experiment_idx += 1

    if failed_runs:
        print(f"\n{'=' * 80}")
        print(f"Batch execution completed with {len(failed_runs)} failures:")
        for model, suite, task in failed_runs:
            print(f"  - model={model}, suite={suite}, task={task}")
        print(f"{'=' * 80}")
        sys.exit(1)
    else:
        print(f"\n{'=' * 80}")
        print("Batch execution completed successfully.")
        print(f"{'=' * 80}")


if __name__ == "__main__":
    tyro.cli(main)
