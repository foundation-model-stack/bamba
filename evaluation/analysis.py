import glob
import json

import numpy as np
import pandas as pd


from loguru import logger

# python -m debugpy --connect cccxl010.pok.ibm.com:1222 /dccstor/eval-research/code/lm-evaluation-harness/.vscode/launch.json

# streamlit run /dccstor/eval-research/code/lm-evaluation-harness/output/analysis.py --server.port 8090
# code from https://colab.research.google.com/drive/1-aPrFJjwdifhVLxzJcsYXeebqNi_5vaw?usp=sharing


# Normalization function
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


def get_noramlized_scores(task_name, data):
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


def handle_duplicates(res_df):
    """
    Handles duplicate entries in a DataFrame based on 'model' and 'scenario' columns.

    Args:
        res_df: The input DataFrame.

    Returns:
        The DataFrame with duplicates removed if scores are consistent,
        otherwise raises a ValueError.
    """

    if len(res_df[res_df.duplicated(subset=["model", "scenario"])]) > 0:
        duplicates = res_df[
            res_df.duplicated(subset=["model", "scenario"], keep=False)
        ]  # Keep all duplicates for comparison

        for index, row in duplicates.iterrows():  # Iterate efficiently
            model = row["model"]
            scenario = row["scenario"]
            score = row["score"]
            other_scores = duplicates[
                (duplicates["model"] == model)
                & (duplicates["scenario"] == scenario)
                & (duplicates.index != index)
            ]["score"].tolist()

            if not all(
                (abs(s - score) < (s / 100))
                for s in other_scores  # difference is smaller than 1%
            ):  # Check consistency across *all* duplicates, not just pairwise
                raise ValueError(
                    f"Inconsistent scores found for model '{model}' and scenario '{scenario}'. Scores: {score}, {other_scores}"
                )

        res_df = res_df.drop_duplicates(
            subset=["model", "scenario"], keep="first"
        )  # Remove duplicates, keeping the first occurrence

    return res_df


hfv2_tasks = [
    "bbh",
    "musr",
    "mmlu_pro",
    "gpqa",
    "math_hard",
    "ifeval",
]

name2tag = {
    "mmlu": "MMLU",
    "hellaswag": "Hellaswag",
    "winogrande": "Winogrande",
    "piqa": "Piqa",
    "openbookqa": "OpenbookQA",
    "arc_challenge": "ARC-C",
    "truthfulqa_mc2": "TruthfulQA",
    "gsm8k": "GSM8K",
    "bbh": "BBH",
    "musr": "MuSR",
    "mmlu_pro": "MMLU-PRO",
    "gpqa": "GPQA",
    "math_hard": "MATH Lvl 5",
    "ifeval": "IFEval",
}

scenario2metric = {
    # HFV1 https://huggingface.co/docs/leaderboards/en/open_llm_leaderboard/archive
    "arc_challenge": "acc_norm",
    "hellaswag": "acc_norm",
    "truthfulqa_mc2": "acc",
    "winogrande": "acc",
    "gsm8k": "exact_match,strict-match",  # this was changed as a later version of lmeval acc->exact_match,strict-match
    "mmlu_abstract_algebra": "acc",
    "mmlu_anatomy": "acc",
    "mmlu_astronomy": "acc",
    "mmlu_business_ethics": "acc",
    "mmlu_clinical_knowledge": "acc",
    "mmlu_college_biology": "acc",
    "mmlu_college_chemistry": "acc",
    "mmlu_college_computer_science": "acc",
    "mmlu_college_mathematics": "acc",
    "mmlu_college_medicine": "acc",
    "mmlu_college_physics": "acc",
    "mmlu_computer_security": "acc",
    "mmlu_conceptual_physics": "acc",
    "mmlu_econometrics": "acc",
    "mmlu_electrical_engineering": "acc",
    "mmlu_elementary_mathematics": "acc",
    "mmlu_formal_logic": "acc",
    "mmlu_global_facts": "acc",
    "mmlu_high_school_biology": "acc",
    "mmlu_high_school_chemistry": "acc",
    "mmlu_high_school_computer_science": "acc",
    "mmlu_high_school_european_history": "acc",
    "mmlu_high_school_geography": "acc",
    "mmlu_high_school_government_and_politics": "acc",
    "mmlu_high_school_macroeconomics": "acc",
    "mmlu_high_school_mathematics": "acc",
    "mmlu_high_school_microeconomics": "acc",
    "mmlu_high_school_physics": "acc",
    "mmlu_high_school_psychology": "acc",
    "mmlu_high_school_statistics": "acc",
    "mmlu_high_school_us_history": "acc",
    "mmlu_high_school_world_history": "acc",
    "mmlu_human_aging": "acc",
    "mmlu_human_sexuality": "acc",
    "mmlu_international_law": "acc",
    "mmlu_jurisprudence": "acc",
    "mmlu_logical_fallacies": "acc",
    "mmlu_machine_learning": "acc",
    "mmlu_management": "acc",
    "mmlu_marketing": "acc",
    "mmlu_medical_genetics": "acc",
    "mmlu_miscellaneous": "acc",
    "mmlu_moral_disputes": "acc",
    "mmlu_moral_scenarios": "acc",
    "mmlu_nutrition": "acc",
    "mmlu_philosophy": "acc",
    "mmlu_prehistory": "acc",
    "mmlu_professional_accounting": "acc",
    "mmlu_professional_law": "acc",
    "mmlu_professional_medicine": "acc",
    "mmlu_professional_psychology": "acc",
    "mmlu_public_relations": "acc",
    "mmlu_security_studies": "acc",
    "mmlu_sociology": "acc",
    "mmlu_us_foreign_policy": "acc",
    "mmlu_virology": "acc",
    "mmlu_world_religions": "acc",
    # Other
    "piqa": "acc_norm",
    "openbookqa": "acc_norm",
}


def main(res_dirs):
    res_list = []
    for res_dir in res_dirs:
        res_file_paths = glob.glob(f"{res_dir}/**/results_*", recursive=True)

        for file_path in res_file_paths:
            all_artifacts = json.load(open(file_path, "r"))
            res_dict = all_artifacts["results"]
            model_name = all_artifacts["model_name"]

            all_res_entries = list(res_dict.keys())

            if (
                "leaderboard" in all_res_entries[0]
            ):  # all entries in the leaderboard has this prefix
                for hfv2_task in hfv2_tasks:
                    if " " in res_dict.get(f"leaderboard_{hfv2_task}", {}).keys():
                        # some results comes without a score, this is how we recognize them
                        continue

                    if any([hfv2_task in entry for entry in all_res_entries]):
                        score = get_noramlized_scores(
                            task_name=hfv2_task, data=res_dict
                        )

                        res_list.append(
                            {
                                "model": model_name,
                                "scenario": hfv2_task,
                                "score": float(score) / 100,
                            }
                        )

            else:
                for scenario, res in res_dict.items():
                    scenario = scenario.replace("leaderboard_", "")
                    # dropping the aggregate here
                    if scenario not in scenario2metric.keys():
                        continue

                    metric_key = [
                        key
                        for key in list(res.keys())
                        if scenario2metric[scenario] == key.replace(",none", "")
                        and "stderr" not in key
                    ]

                    assert len(metric_key) == 1, "More than one metric?"

                    res_list.append(
                        {
                            "model": model_name,
                            "scenario": scenario,
                            "score": res[metric_key[0]],
                        }
                    )

    res_df = pd.DataFrame(res_list)

    if len(res_df[res_df.duplicated(subset=["model", "scenario"])]) > 0:
        res_df = handle_duplicates(res_df)

    # aggregating scenarios
    multi_subset_scenarios = ["mmlu"]
    scenario_to_avoid = "mmlu_pro"

    for scenario_name in multi_subset_scenarios:
        res_df["scenario"] = res_df["scenario"].apply(
            lambda x: scenario_name
            if (scenario_name + "_" in x and x != scenario_to_avoid)
            else x
        )

    res_df = res_df.groupby(["model", "scenario"]).agg({"score": "mean"}).reset_index()
    res_df["score"] = res_df["score"] * 100

    df_from_papers = pd.read_csv("output/results_from_papers.csv")
    df_from_papers = pd.melt(
        df_from_papers,
        id_vars="scenario",
        var_name="model",
        value_name="score",
    )
    df_from_papers = df_from_papers.dropna()

    res_df["scenario"] = res_df["scenario"].apply(lambda x: name2tag[x])
    res_df = pd.concat([res_df, df_from_papers])

    if len(res_df[res_df.duplicated(subset=["model", "scenario"])]) > 0:
        res_df = handle_duplicates(res_df)

    res_df["score"] = res_df["score"].round(2)

    # Pivot the DataFrame
    df_pivot_score = res_df.pivot(
        index="model", columns="scenario", values=["score"]
    ).reset_index()
    # df_pivot_stderr = res_df.pivot(index='model', columns='scenario', values='stderr').reset_index()
    flat_index = [
        "model" if level0 == "model" else level1
        for level0, level1 in df_pivot_score.columns
    ]
    df_pivot_score.columns = flat_index
    df_pivot_score.to_csv("output/combined_results.csv", index=False)

    return df_pivot_score


if __name__ == "__main__":
    df = main(
        res_dirs=[
            "/dccstor/eval-research/code/lm-evaluation-harness/output/Bamba_eval",
            "/dccstor/eval-research/code/lm-evaluation-harness/output/Bamba_eval_last_models",
            # "output/241205_HFV2",
            # "output/241205_Other",
            # "output/241205_HFV1",
            # "output/251205_HFV2",
            # "output/251205_more_gsm8k",
        ]
    )

    print()

    import streamlit as st

    st.set_page_config(page_title="Bamba evaluations", page_icon="🧊", layout="wide")
    # Create format dict that rounds all numeric columns
    format_dict = {"Predictions": "{:.2f}"}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            format_dict[col] = "{:.2f}"

    st.title("🚀🚀🚀 Evals for Bamba model release 🚀🚀🚀")

    styled_df = df.style.background_gradient(cmap="Greens").format(format_dict)
    st.dataframe(styled_df, use_container_width=True, hide_index=True, height=550)

    st.write("* results taken from paper")

    st.markdown(
        """
        Results gatherd using lm-evaluation-harness (bcb4cbf)
        with the additional task relevant changes from https://github.com/huggingface/lm-evaluation-harness/tree/main required from the HF Open LLM leaderboard V2 tasks
        using evaluation parameters used are as defined in:
        - https://huggingface.co/docs/leaderboards/open_llm_leaderboard/about
        - https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/leaderboard
        - https://huggingface.co/docs/leaderboards/en/open_llm_leaderboard/archive
        """
    )

    st.markdown(
        """
        ### Tasks:
        - IFEval (https://arxiv.org/abs/2311.07911): IFEval is a dataset designed to test a model's ability to follow explicit instructions, such as “include keyword x” or “use format y.” The focus is on the model’s adherence to formatting instructions rather than the content generated, allowing for the use of strict and rigorous metrics.
        - BBH (Big Bench Hard) (https://arxiv.org/abs/2210.09261): A subset of 23 challenging tasks from the BigBench dataset to evaluate language models. The tasks use objective metrics, are highly difficult, and have sufficient sample sizes for statistical significance. They include multistep arithmetic, algorithmic reasoning (e.g., boolean expressions, SVG shapes), language understanding (e.g., sarcasm detection, name disambiguation), and world knowledge. BBH performance correlates well with human preferences, providing valuable insights into model capabilities.
        - MATH (https://arxiv.org/abs/2103.03874):  MATH is a compilation of high-school level competition problems gathered from several sources, formatted consistently using Latex for equations and Asymptote for figures. Generations must fit a very specific output format. We keep only level 5 MATH questions and call it MATH Lvl 5.
        - GPQA (Graduate-Level Google-Proof Q&A Benchmark) (https://arxiv.org/abs/2311.12022): GPQA is a highly challenging knowledge dataset with questions crafted by PhD-level domain experts in fields like biology, physics, and chemistry. These questions are designed to be difficult for laypersons but relatively easy for experts. The dataset has undergone multiple rounds of validation to ensure both difficulty and factual accuracy. Access to GPQA is restricted through gating mechanisms to minimize the risk of data contamination. Consequently, we do not provide plain text examples from this dataset, as requested by the authors.
        - MuSR (Multistep Soft Reasoning) (https://arxiv.org/abs/2310.16049): MuSR is a new dataset consisting of algorithmically generated complex problems, each around 1,000 words in length. The problems include murder mysteries, object placement questions, and team allocation optimizations. Solving these problems requires models to integrate reasoning with long-range context parsing. Few models achieve better than random performance on this dataset.
        - MMLU-PRO (Massive Multitask Language Understanding - Professional) (https://arxiv.org/abs/2406.01574): MMLU-Pro is a refined version of the MMLU dataset, which has been a standard for multiple-choice knowledge assessment. Recent research identified issues with the original MMLU, such as noisy data (some unanswerable questions) and decreasing difficulty due to advances in model capabilities and increased data contamination. MMLU-Pro addresses these issues by presenting models with 10 choices instead of 4, requiring reasoning on more questions, and undergoing expert review to reduce noise. As a result, MMLU-Pro is of higher quality and currently more challenging than the original.
        - AI2 Reasoning Challenge (https://arxiv.org/abs/1803.05457) - a set of grade-school science questions.
        - HellaSwag (https://arxiv.org/abs/1905.07830) - a test of commonsense inference, which is easy for humans (~95%) but challenging for SOTA models.
        - MMLU (https://arxiv.org/abs/2009.03300) - a test to measure a text model's multitask accuracy. The test covers 57 tasks including elementary mathematics, US history, computer science, law, and more.
        - TruthfulQA (https://arxiv.org/abs/2109.07958) - a test to measure a model's propensity to reproduce falsehoods commonly found online. Note: TruthfulQA is technically a 6-shot task in the Harness because each example is prepended with 6 Q/A pairs, even in the 0-shot setting.
        - Winogrande (https://arxiv.org/abs/1907.10641) - an adversarial and difficult Winograd benchmark at scale, for commonsense reasoning.
        - GSM8k (https://arxiv.org/abs/2110.14168) - diverse grade school math word problems to measure a model's ability to solve multi-step mathematical reasoning problems.
        -
        """
    )

    # nvidia/mamba2-hybrid-8b-3t-4k from the paper
    # allenai/OLMo-7B-hf I took the HFV2 results and from HFV1
    # meta-llama/Llama-2-7b-hf I took the HFV2 results and from HFV1
    # meta-llama/Llama-3-8B I took the HFV2 results and from HFV1
    # other models from https://huggingface.co/tiiuae/falcon-mamba-7b after varifying consistency

    # try:
    #     import getpass

    #     import pandas as pd
    #     from lh_eval_api import EvaluationResultsUploader, RunRecord

    #     # get your variables
    #     benchmark = "Bamba-eval"
    #     score_name = ""
    #     framework = "LM-Eval_Harness"
    #     time = "12021988"
    #     is_official = False
    #     owner = getpass.getuser()

    #     # prepare run records
    #     run_records = []
    #     long_df = df.melt(id_vars="model", var_name="dataset", value_name="score")
    #     result_dicts = long_df.to_dict(orient="records")
    #     for result in result_dicts:
    #         run_records.append(
    #             RunRecord(
    #                 owner=owner,
    #                 started_at=time,
    #                 framework=framework,
    #                 inference_platform="",
    #                 model_name=result["model"],
    #                 execution_env="",
    #                 benchmark=benchmark,
    #                 dataset=result["dataset"],
    #                 task="",
    #                 run_params={"framework": framework},
    #                 score=result["score"],
    #                 score_name=score_name,
    #             )
    #         )

    #     # upload
    #     uploader = EvaluationResultsUploader(runs=run_records)
    #     uploader.upload()
    #     st.write("uploaded")

    # except:
    #     st.write("failed")
