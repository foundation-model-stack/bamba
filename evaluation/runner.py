import argparse
import glob
import json
import logging
import logging.handlers
import os
import signal

from runner_tasks import runner_tasks

from evaluation.lsf_runner_utils import (
    get_job_id,
    monitor_progress,
    setup_logging,
    signal_handler,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run leaderboard evaluation.")

    # Constants (now configurable via command-line arguments)
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["HFV2"],
        help="List of benchmarks to evaluate on. (default: ['HFV2'])",
    )
    parser.add_argument(
        "--only_subtasks_to_run",
        nargs="+",
        default=[],
        help="List of specific subtasks to run within the chosen benchmarks. If empty, runs all subtasks. (default: [])",
    )

    parser.add_argument(
        "--output_dir_path",
        default="debug",
        help="Path to the directory where evaluation results and logs will be saved. (default: 'debug')",
    )

    parser.add_argument(
        "--memory",
        default="64g",
        help="Amount of memory to request for the job. (default: '64g')",
    )
    parser.add_argument(
        "--req_gpu",
        default="a100_80gb",
        help="Type of GPU to request for the job. (default: 'a100_80gb')",
    )
    parser.add_argument(
        "--cores",
        default="8+1",
        help="Number of CPU cores to request. (default: '8+1')",
    )
    parser.add_argument(
        "--queue",
        default="nonstandard",
        help="Name of the queue to submit the job to. (default: 'nonstandard')",
    )
    parser.add_argument(
        "--python_executable",
        default="python",
        help="Path to the Python executable to use. (default: 'python')",
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=[],
        help="List of model names to evaluate. (default: [])",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of examples to evaluate per task. Useful for debugging. (default: None, meaning no limit)",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size to use during evaluation. (default: 4)",
    )

    parser.add_argument(
        "--fp_precision",
        type=int,
        default=16,
        help="Floating-point precision to use (e.g., 16 for fp16, 32 for fp32). (default: 16)",
    )

    parser.add_argument(
        "--debug_run_single_task_per_model",
        action="store_true",
        help="If set, runs only one subtask per model for quick debugging. (default: False)",
    )

    args = parser.parse_args()

    return args


# Logging setup
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def sanitize_model_id(model_id):
    return model_id.replace("/", "_").replace(":", "_")


def run_job(model_id, task_to_run, args):
    sanitized_model_id = sanitize_model_id(model_id)
    cache_dir = f"{os.environ['XDG_CACHE_HOME']}/hf_cache_{sanitized_model_id}"
    output_path = os.path.join(args.output_dir_path, sanitized_model_id)
    results_file_pattern = os.path.join(
        output_path, "**", "results_*"
    )  # pattern to check

    # Check if results files already exist
    done_subtasks = []
    result_files = glob.glob(results_file_pattern)
    if result_files:
        # check which tasks were evaluated already
        for result_file in result_files:
            done_subtasks.extend(list(json.load(open(result_file))["results"].keys()))

    if args.only_subtasks_to_run:
        subtasks_to_run = [
            subtask
            for subtask in args.only_subtasks_to_run
            if subtask in task_to_run["subtasks"]
        ]
    else:
        subtasks_to_run = task_to_run["subtasks"]

    subtasks_to_run = [
        subtask for subtask in subtasks_to_run if subtask not in done_subtasks
    ]
    if len(subtasks_to_run) == 0:
        logging.info(f"All {len(task_to_run['subtasks'])} subtasks already exist")
        return None
    elif len(subtasks_to_run) < len(task_to_run["subtasks"]):
        logging.info(
            f"Skipped: {len(subtasks_to_run) - len(task_to_run['subtasks'])} already evaluated for {model_id}\n"
            f"{len(subtasks_to_run)} subtasks left to run"
        )

    model_args = f"pretrained={model_id},"
    if args.fp_precision == 16:
        model_args += "dtype=float16"
    elif args.fp_precision in [8, 4]:
        model_args += f"load_in_{args.fp_precision}_bit=True"
    else:
        raise NotImplementedError(
            f"current precision {args.fp_precision} is not supported, only [4,8,16]"
        )

    command = [
        "jbsub",
        "-name",
        task_to_run["task"] + "_" + model_id,
        "-mem",
        args.memory,
        "-cores",
        args.cores,
        "-require",
        args.req_gpu,
        "-q",
        args.queue,
        "cd /dccstor/eval-research/code/lm-evaluation-harness",
        "&&",
        f"HF_HOME={cache_dir}",
        args.python_executable,
        "/dccstor/eval-research/code/lm-evaluation-harness/lm_eval",
        "--model_args",
        model_args,
        "--batch_size",
        args.batch_size,
        "--tasks",
        ",".join(subtasks_to_run),
        "--output_path",
        output_path,
        "--cache_requests",
        "true",
        "--log_samples",
        "--trust_remote_code",
        # f"--use_cache={cache_dir}",
    ]

    if args.limit:
        command.append(
            f"--limit={args.limit}",
        )
    if task_to_run["num_fewshot"]:
        command.append(
            f"--num_fewshot={task_to_run['num_fewshot']}",
        )

    # print(" ".join(command))
    job_id = get_job_id(model_id, output_path, command)

    return job_id


if __name__ == "__main__":
    args = parse_args()

    signal.signal(signal.SIGINT, signal_handler)
    models_to_run = args.models.copy()
    job_ids = []

    # Create output directory if it doesn't exist
    os.makedirs(args.output_base_path, exist_ok=True)

    # Set up logging to write to a file within OUTPUT_BASE_PATH
    setup_logging(args.output_base_path)

    job_id2model = {}
    for model in models_to_run:
        runs_per_model = 0
        for benchmark in args.benchmarks:
            tasks_to_run = runner_tasks[benchmark]
            for task_to_run in tasks_to_run:
                if runs_per_model > 0 and args.debug_run_single_task_per_model:
                    continue

                runs_per_model += 1
                job_id = run_job(model, task_to_run, args)
                job_id2model[job_id] = model

                if job_id:
                    job_ids.append(job_id)

    if not monitor_progress(job_ids, job_id2model):
        logging.error("Some models failed to complete within timeout")

    logging.info("Experiment finished.")
