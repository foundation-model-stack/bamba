#!/bin/bash

# Define paths and variables in one place
harness_path="path/to/lm-evaluation-harness"
python_path="python"
lm_eval_script="${harness_path}/lm_eval"
pretrained_model="ibm-fms/Bamba-9B"
output_base_path="evaluation_results/debug/ibm-fms_Bamba-9B"
batch_size=4

# Function to run lm_eval with common arguments
run_bamba_eval() {
  local tasks="$1"
  local num_fewshot="$2" # Optional, can be empty
  local extra_args="$3" # Optional, for task-specific arguments

  cd "${harness_path}" || exit
  "${python_path}" "${lm_eval_script}" \
    --model hf \
    --model_args "pretrained=${pretrained_model},dtype=float16" \
    --batch_size "${batch_size}" \
    --tasks "${tasks}" \
    --output_path "${output_base_path}" \
    --cache_requests true \
    --log_samples \
    --trust_remote_code \
    --num_fewshot="${num_fewshot}" \
    "${extra_args}"
}

# Run evaluations for each task
run_bamba_eval "mmlu_abstract_algebra,mmlu_anatomy,mmlu_astronomy,mmlu_business_ethics,mmlu_clinical_knowledge,mmlu_college_biology,mmlu_college_chemistry,mmlu_college_computer_science,mmlu_college_mathematics,mmlu_college_medicine,mmlu_college_physics,mmlu_computer_security,mmlu_conceptual_physics,mmlu_econometrics,mmlu_electrical_engineering,mmlu_elementary_mathematics,mmlu_formal_logic,mmlu_global_facts,mmlu_high_school_biology,mmlu_high_school_chemistry,mmlu_high_school_computer_science,mmlu_high_school_european_history,mmlu_high_school_geography,mmlu_high_school_government_and_politics,mmlu_high_school_macroeconomics,mmlu_high_school_mathematics,mmlu_high_school_microeconomics,mmlu_high_school_physics,mmlu_high_school_psychology,mmlu_high_school_statistics,mmlu_high_school_us_history,mmlu_high_school_world_history,mmlu_human_aging,mmlu_human_sexuality,mmlu_international_law,mmlu_jurisprudence,mmlu_logical_fallacies,mmlu_machine_learning,mmlu_management,mmlu_marketing,mmlu_medical_genetics,mmlu_miscellaneous,mmlu_moral_disputes,mmlu_moral_scenarios,mmlu_nutrition,mmlu_philosophy,mmlu_prehistory,mmlu_professional_accounting,mmlu_professional_law,mmlu_professional_medicine,mmlu_professional_psychology,mmlu_public_relations,mmlu_security_studies,mmlu_sociology,mmlu_us_foreign_policy,mmlu_virology,mmlu_world_religions" 5
run_bamba_eval "arc_challenge" 25
run_bamba_eval "hellaswag" 10
run_bamba_eval "truthfulqa_mc2" ""
run_bamba_eval "winogrande" 5
run_bamba_eval "gsm8k" 5
run_bamba_eval "leaderboard_mmlu_pro" ""
run_bamba_eval "leaderboard_bbh" ""
run_bamba_eval "leaderboard_gpqa" ""
run_bamba_eval "leaderboard_ifeval" ""
run_bamba_eval "leaderboard_musr" ""
run_bamba_eval "leaderboard_math_hard" ""
run_bamba_eval "boolq" 5
run_bamba_eval "openbookqa" 5
run_bamba_eval "piqa" 5

echo "Evaluations completed."
