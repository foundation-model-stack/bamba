# Bamba Evaluation

### installation

```bash
pip install -r requirements_evaluation.txt
```

## Gateway scripts

### Running
`evaluation/runner.py` is used as orchastrator for (currently) (lm-eval-harness)[https://github.com/EleutherAI/lm-evaluation-harness] using LSF.

It will probably only work for you while running on the CCC.

use it like this:
```bash
python evaluation/runner.py \
--models \
allenai/OLMo-7B-hf \
allenai/OLMo-7B-0424-hf \
allenai/OLMo-2-1124-7B \
--benchmarks HFV1 HFV2 Other \
output_dir_path evaluation/evaluation_results/debug
--python_executable \
/dccstor/eval-research/miniforge3/envs/lmeval_olmo/bin/python \
```

if you want to debug, you can use:
`--debug_run_single_task_per_model` and `--limit=10` to limit the number of runs and examples.

in `evaluation/runner_tasks.py` you will find all the tasks and corrsponding benchmarks, they can be run full by naming the benchmark or if only a specific subtask is required, the `--only_subtasks_to_run` flag should be used. Note! the subtasks should be of the subtasks of the requested benchmarks.

for other configs.

```bash
usage: runner.py [-h] [--benchmarks BENCHMARKS [BENCHMARKS ...]]
                 [--only_subtasks_to_run ONLY_SUBTASKS_TO_RUN [ONLY_SUBTASKS_TO_RUN ...]]
                 [--output_dir_path OUTPUT_DIR_PATH] [--memory MEMORY]
                 [--req_gpu REQ_GPU] [--cores CORES] [--queue QUEUE]
                 [--python_executable PYTHON_EXECUTABLE] [--models MODELS [MODELS ...]]
                 [--limit LIMIT] [--batch_size BATCH_SIZE] [--fp_precision FP_PRECISION]
                 [--debug_run_single_task_per_model]

Run leaderboard evaluation.

options:
  -h, --help            show this help message and exit
  --benchmarks BENCHMARKS [BENCHMARKS ...]
                        List of benchmarks to evaluate on. (default: ['HFV2'])
  --only_subtasks_to_run ONLY_SUBTASKS_TO_RUN [ONLY_SUBTASKS_TO_RUN ...]
                        List of specific subtasks to run within the chosen benchmarks. If
                        empty, runs all subtasks. (default: [])
  --output_dir_path OUTPUT_DIR_PATH
                        Path to the directory where evaluation results and logs will be
                        saved. (default: 'debug')
  --memory MEMORY       Amount of memory to request for the job. (default: '64g')
  --req_gpu REQ_GPU     Type of GPU to request for the job. (default: 'a100_80gb')
  --cores CORES         Number of CPU cores to request. (default: '8+1')
  --queue QUEUE         Name of the queue to submit the job to. (default: 'nonstandard')
  --python_executable PYTHON_EXECUTABLE
                        Path to the Python executable to use. (default: 'python')
  --models MODELS [MODELS ...]
                        List of model names to evaluate. (default: [])
  --limit LIMIT         Limit the number of examples to evaluate per task. Useful for
                        debugging. (default: None, meaning no limit)
  --batch_size BATCH_SIZE
                        Batch size to use during evaluation. (default: 4)
  --fp_precision FP_PRECISION
                        Floating-point precision to use (e.g., 16 for fp16, 32 for fp32).
                        (default: 16)
  --debug_run_single_task_per_model
                        If set, runs only one subtask per model for quick debugging.
                        (default: False)
```
