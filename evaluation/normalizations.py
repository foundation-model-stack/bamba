import numpy as np

needs_normalization = lambda all_res_entries: "leaderboard" in all_res_entries[0]

# code from https://colab.research.google.com/drive/1-aPrFJjwdifhVLxzJcsYXeebqNi_5vaw?usp=sharing
# Normalization function

hfv2_tasks = [
    "bbh",
    "musr",
    "mmlu_pro",
    "gpqa",
    "math_hard",
    "ifeval",
]


def normalize_within_range(value, lower_bound=0, higher_bound=1):
    return (np.clip(value - lower_bound, 0, None)) / (higher_bound - lower_bound) * 100


bbh_subtasks = {
    "sports_understanding": 2,
    "tracking_shuffled_objects_three_objects": 3,
    "navigate": 2,
    "snarks": 2,
    "date_understanding": 6,
    "reasoning_about_colored_objects": 18,
    "object_counting": 19,
    "logical_deduction_seven_objects": 7,
    "geometric_shapes": 11,
    "web_of_lies": 2,
    "movie_recommendation": 6,
    "logical_deduction_five_objects": 5,
    "salient_translation_error_detection": 6,
    "disambiguation_qa": 3,
    "temporal_sequences": 4,
    "hyperbaton": 2,
    "logical_deduction_three_objects": 3,
    "causal_judgement": 2,
    "formal_fallacies": 2,
    "tracking_shuffled_objects_seven_objects": 7,
    "ruin_names": 6,
    "penguins_in_a_table": 5,
    "boolean_expressions": 2,
    "tracking_shuffled_objects_five_objects": 5,
}

musr_subtasks = {"murder_mysteries": 2, "object_placements": 5, "team_allocation": 3}


def get_hfv2_noramlized_scores(task_name, data):
    if task_name == "bbh":
        # Normalize BBH subtasks scores
        bbh_scores = []
        for subtask, num_choices in bbh_subtasks.items():
            subtask_key = f"leaderboard_bbh_{subtask}"
            if subtask_key in data:
                bbh_raw_score = data[subtask_key]["acc_norm,none"]
                lower_bound = 1 / num_choices
                normalized_score = normalize_within_range(
                    bbh_raw_score, lower_bound, 1.0
                )
                bbh_scores.append(normalized_score)

        # Average BBH score
        score = sum(bbh_scores) / len(bbh_scores)

    elif task_name == "math_hard":
        # Calculate the MATH score
        math_raw_score = data["leaderboard_math_hard"]["exact_match,none"]
        score = normalize_within_range(math_raw_score, 0, 1.0)

    elif task_name == "gpqa":
        # Normalize GPQA scores
        gpqa_raw_score = data["leaderboard_gpqa"]["acc_norm,none"]
        score = normalize_within_range(gpqa_raw_score, 0.25, 1.0)

    elif task_name == "mmlu_pro":
        # Normalize MMLU PRO scores
        mmlu_pro_raw_score = data["leaderboard_mmlu_pro"]["acc,none"]
        score = normalize_within_range(mmlu_pro_raw_score, 0.1, 1.0)

    elif task_name == "ifeval":
        # Compute IFEval
        ifeval_inst_score = (
            data["leaderboard_ifeval"]["inst_level_strict_acc,none"] * 100
        )
        ifeval_prompt_score = (
            data["leaderboard_ifeval"]["prompt_level_strict_acc,none"] * 100
        )
        # Average IFEval scores
        score = (ifeval_inst_score + ifeval_prompt_score) / 2

    elif task_name == "musr":
        # Normalize MUSR scores
        musr_scores = []

        for subtask, num_choices in musr_subtasks.items():
            musr_raw_score = data[f"leaderboard_musr_{subtask}"]["acc_norm,none"]
            lower_bound = 1 / num_choices
            normalized_score = normalize_within_range(musr_raw_score, lower_bound, 1.0)
            musr_scores.append(normalized_score)

        score = sum(musr_scores) / len(musr_scores)

    else:
        raise NotImplementedError(f"Not supproting task_name {task_name}")

    return score
