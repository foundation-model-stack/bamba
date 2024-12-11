import argparse
import glob
import json
import logging
import os
import re
import signal
import subprocess
import sys
import time

from tqdm import tqdm


tasks = {
    "Other": [
        {"task": "BoolQ", "num_fewshot": 5, "subtasks": ["boolq"]},
        {"task": "OpenbookQA", "num_fewshot": 5, "subtasks": ["openbookqa"]},
        {"task": "PIQA", "num_fewshot": 5, "subtasks": ["piqa"]},
    ],
    "HFV2": [
        {
            "task": "leaderboard_mmlu_pro",
            "num_fewshot": None,
            "subtasks": ["leaderboard_mmlu_pro"],
        },
        {
            "task": "leaderboard_bbh",
            "num_fewshot": None,
            "subtasks": ["leaderboard_bbh"],
        },
        {
            "task": "leaderboard_gpqa",
            "num_fewshot": None,
            "subtasks": ["leaderboard_gpqa"],
        },
        {
            "task": "leaderboard_ifeval",
            "num_fewshot": None,
            "subtasks": ["leaderboard_ifeval"],
        },
        {
            "task": "leaderboard_musr",
            "num_fewshot": None,
            "subtasks": ["leaderboard_musr"],
        },
        {
            "task": "leaderboard_math_hard",
            "num_fewshot": None,
            "subtasks": ["leaderboard_math_hard"],
        },
    ],
    "HFV1": [
        {
            "task": "MMLU",
            "num_fewshot": 5,
            "subtasks": [
                "mmlu_abstract_algebra",
                "mmlu_anatomy",
                "mmlu_astronomy",
                "mmlu_business_ethics",
                "mmlu_clinical_knowledge",
                "mmlu_college_biology",
                "mmlu_college_chemistry",
                "mmlu_college_computer_science",
                "mmlu_college_mathematics",
                "mmlu_college_medicine",
                "mmlu_college_physics",
                "mmlu_computer_security",
                "mmlu_conceptual_physics",
                "mmlu_econometrics",
                "mmlu_electrical_engineering",
                "mmlu_elementary_mathematics",
                "mmlu_formal_logic",
                "mmlu_global_facts",
                "mmlu_high_school_biology",
                "mmlu_high_school_chemistry",
                "mmlu_high_school_computer_science",
                "mmlu_high_school_european_history",
                "mmlu_high_school_geography",
                "mmlu_high_school_government_and_politics",
                "mmlu_high_school_macroeconomics",
                "mmlu_high_school_mathematics",
                "mmlu_high_school_microeconomics",
                "mmlu_high_school_physics",
                "mmlu_high_school_psychology",
                "mmlu_high_school_statistics",
                "mmlu_high_school_us_history",
                "mmlu_high_school_world_history",
                "mmlu_human_aging",
                "mmlu_human_sexuality",
                "mmlu_international_law",
                "mmlu_jurisprudence",
                "mmlu_logical_fallacies",
                "mmlu_machine_learning",
                "mmlu_management",
                "mmlu_marketing",
                "mmlu_medical_genetics",
                "mmlu_miscellaneous",
                "mmlu_moral_disputes",
                "mmlu_moral_scenarios",
                "mmlu_nutrition",
                "mmlu_philosophy",
                "mmlu_prehistory",
                "mmlu_professional_accounting",
                "mmlu_professional_law",
                "mmlu_professional_medicine",
                "mmlu_professional_psychology",
                "mmlu_public_relations",
                "mmlu_security_studies",
                "mmlu_sociology",
                "mmlu_us_foreign_policy",
                "mmlu_virology",
                "mmlu_world_religions",
            ],
        },
        {"task": "ARC", "num_fewshot": 25, "subtasks": ["arc_challenge"]},
        {"task": "HellaSwag", "num_fewshot": 10, "subtasks": ["hellaswag"]},
        {"task": "TruthfulQA", "num_fewshot": 0, "subtasks": ["truthfulqa_mc2"]},
        {"task": "Winogrande", "num_fewshot": 5, "subtasks": ["winogrande"]},
        {"task": "GSM8k", "num_fewshot": 5, "subtasks": ["gsm8k"]},
    ],
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run leaderboard evaluation.")

    # Constants (now configurable via command-line arguments)
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["HFV2"],
        help="Task name (default: leaderboard)",
    )
    parser.add_argument(
        "--subtasks_to_run",
        nargs="+",
        default=[],
    )

    parser.add_argument(
        "--output_dir_name",
        default="debug",
        help="Output directory name (default: bamba_leaderboard_full_leaderboard_branch)",
    )

    parser.add_argument(
        "--cache_dir_path",
        default="/dccstor/fme/users/yotam/cache",
        help="Output directory name (default: bamba_leaderboard_full_leaderboard_branch)",
    )

    parser.add_argument("--memory", default="64g", help="Memory request (default: 64g)")
    parser.add_argument(
        "--req_gpu", default="a100_80gb", help="Required GPU type (default: a100_80gb)"
    )
    parser.add_argument("--cores", default="8+1", help="Number of cores (default: 8+1)")
    parser.add_argument(
        "--queue", default="nonstandard", help="Queue name (default: nonstandard)"
    )
    parser.add_argument(
        "--python_executable",
        default="/dccstor/eval-research/miniforge3/envs/bamba/bin/python",
        help="Path to Python executable (default: /dccstor/eval-research/miniforge3/envs/bamba/bin/python)",
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=[],
        help="List of models to evaluate",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--fp_precision",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--debug_run_single_task_per_model",
        action="store_true",
    )

    args = parser.parse_args()

    # Derive output_base_path if not provided explicitly
    args.output_base_path = os.path.join(
        "/dccstor/eval-research/code/lm-evaluation-harness/output",
        args.output_dir_name,
        "",  # Add trailing slash
    )

    return args


# Logging setup
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def sanitize_model_id(model_id):
    return model_id.replace("/", "_").replace(":", "_")


def run_job(model_id, task_to_run, args):
    sanitized_model_id = sanitize_model_id(model_id)
    cache_dir = f"{args.cache_dir_path}/hf_cache_{sanitized_model_id}"
    output_path = os.path.join(args.output_base_path, sanitized_model_id)
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

    if args.subtasks_to_run:
        subtasks_to_run = [
            subtask
            for subtask in args.subtasks_to_run
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
        "HF_TOKEN=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",
        f"LM_HARNESS_CACHE_PATH={cache_dir}",
        args.python_executable,
        "/dccstor/eval-research/code/lm-evaluation-harness/lm_eval",
        "--model_args",
        model_args,
        "--batch_size",
        "1",
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

    print(" ".join(command))

    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        logging.info(f"Submitted job for {model_id}. Output will be in {output_path}")

        match = re.search(r"Job <(\d+)>", result.stdout)
        if match:
            job_id = match.group(1)
            logging.info(f"Job ID: {job_id}")
            return job_id
        else:
            logging.warning(f"Could not parse job ID from output: {result.stdout}")
            return None

    except subprocess.CalledProcessError as e:
        logging.error(f"Error submitting job for {model_id}:")
        logging.error(f"Return code: {e.returncode}")
        logging.error(f"Stdout: {e.stdout}")
        logging.error(f"Stderr: {e.stderr}")
        return None


def get_all_job_statuses():
    """Gets the status of all jobs using bjobs."""
    try:
        result = subprocess.run(
            ["bjobs", "-a", "-o", "jobid stat"],
            capture_output=True,
            text=True,
            check=True,
        )
        output = result.stdout.strip()
        all_job_statuses = {}
        for line in output.splitlines()[1:]:  # Skip header
            job_id_str, status = line.split()
            all_job_statuses[job_id_str] = status
        return all_job_statuses
    except subprocess.CalledProcessError as e:
        logging.error(f"Error getting job statuses: {e}")


def monitor_progress(job_ids: list, job_id2model):
    total_jobs = len(job_ids)
    completed_jobs = 0
    failed_jobs = []

    with tqdm(total=total_jobs, desc="Processing Jobs") as pbar:
        while completed_jobs < total_jobs - len(failed_jobs):  # Exit early on failure
            time.sleep(60)

            all_job_statuses = get_all_job_statuses()
            if all_job_statuses is None:
                break

            for job_id in job_ids:
                # job_id = job_ids.get(model_id)
                if job_id:
                    status = all_job_statuses.get(str(job_id))
                    if status == "DONE":
                        completed_jobs += 1
                        pbar.update(1)
                        job_ids.remove(job_id)
                        logging.info(f"Job {job_id} completed successfully.")

                    elif status == "EXIT":
                        failed_jobs.append(job_id)
                        job_ids.remove(job_id)
                        logging.error(
                            f"Job {job_id} failed, with model {job_id2model[job_id]}. Check LSF logs."
                        )
                        pbar.update(1)

                    # Ignore other statuses (RUN, PEND, etc.)

            time.sleep(10)  # Adjust as needed

    if failed_jobs:
        logging.error(f"The following jobs failed: {', '.join(map(str, failed_jobs))}")

    return completed_jobs == total_jobs


def signal_handler(sig, frame):
    logging.warning("Experiment interrupted. Exiting.")
    sys.exit(1)


import logging
import logging.handlers


def setup_logging(output_dir):
    """Configures logging to write to a file and the console."""
    log_file = os.path.join(output_dir, "run_bluebench.log")

    # Create the output directory if it doesn't exist.
    os.makedirs(output_dir, exist_ok=True)

    try:
        handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10 * 1024 * 1024, backupCount=5
        )  # 10MB max size, 5 backups
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logging.getLogger().addHandler(handler)
        logging.getLogger().setLevel(logging.INFO)

    except Exception as e:  # Catch any potential errors
        print(
            f"Error setting up logging: {e}.  Log messages will only be printed to the console."
        )


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
            tasks_to_run = tasks[benchmark]
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
