import logging
import os
import re
import subprocess
import sys
import time

from tqdm import tqdm


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


def get_job_id(model_id, output_path, command):
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        logging.info(f"Submitted job for {model_id}. Output will be in {output_path}")

        match = re.search(r"Job <(\d+)>", result.stdout)
        if match:
            job_id = match.group(1)
            logging.info(f"Job ID: {job_id}")
        else:
            logging.warning(f"Could not parse job ID from output: {result.stdout}")
            job_id = None

    except subprocess.CalledProcessError as e:
        logging.error(f"Error submitting job for {model_id}:")
        logging.error(f"Return code: {e.returncode}")
        logging.error(f"Stdout: {e.stdout}")
        logging.error(f"Stderr: {e.stderr}")
        job_id = None
