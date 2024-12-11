import os

import pandas as pd

from evaluation.aggregation import get_results_df, parse_args

if __name__ == "__main__":
    args = parse_args()

    df = get_results_df(
        res_dir_paths=[
            os.path.join(args.output_dir_path, res_dir) for res_dir in args.res_dirs
        ],
        results_from_papers_path=os.path.join(
            args.output_dir_path, "results_from_papers.csv"
        ),
    )

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
        """
    )
