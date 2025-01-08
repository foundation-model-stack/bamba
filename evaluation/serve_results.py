# .venv/bin/streamlit run evaluation/serve_results.py --server.port 8091 -- --output_dir_path /dccstor/eval-research/code/bamba/evaluation/evaluation_results --res_dirs Bamba_eval Bamba_eval_last_models safety_full

import os

import pandas as pd
import streamlit as st

from evaluation.aggregation import add_mwr_col, get_results_df, parse_args, pivot_df


@st.cache_data
def get_results_df_cached(output_dir_path, res_dirs):
    # Fetch data from URL here, and then clean it up.
    return pivot_df(
        add_mwr_col(
            get_results_df(
                res_dir_paths=[
                    os.path.join(output_dir_path, res_dir) for res_dir in res_dirs
                ],
                # results_from_papers_path=os.path.join(
                #     output_dir_path, "results_from_papers.csv"
                # ),
            )
        )
    )


if __name__ == "__main__":
    st.set_page_config(page_title="Bamba evaluations", page_icon="🧊", layout="wide")

    args = parse_args()

    df = get_results_df_cached(args.output_dir_path, args.res_dirs)
    # Create format dict that rounds all numeric columns
    format_dict = {"Predictions": "{:.2f}"}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            format_dict[col] = "{:.2f}"

    # st.title("🚧 FMS LLM training Dev Evals 🚧")
    st.markdown(
        "<h1 style='text-align: center; color: black;'>🚧 FMS LLM Training Evals 🚧</h1>",
        unsafe_allow_html=True,
    )
    st.write("##")

    df = df[
        [
            "model",
            "MWR",
            "MMLU",
            "ARC-C",
            "GSM8K",
            "Hellaswag",
            "OpenbookQA",
            "Piqa",
            "TruthfulQA",
            "Winogrande",
            "Boolq",
            "MMLU-PRO",
            "BBH",
            "GPQA",
            "IFEval",
            "MATH Lvl 5",
            "MuSR",
        ]
    ]

    metadata = pd.read_csv("evaluation/assets/eval_metadata.csv")
    models_not_in_metadata = [
        model
        for model in df["model"].unique()
        if model not in metadata["model"].unique()
    ]
    if models_not_in_metadata:
        st.write(f"{models_not_in_metadata} missing from the metadata")

    df = df.merge(metadata)

    df["model"] = df["model"].apply(
        lambda x: x.replace("9.8b", "9B")
        .replace("9b", "9B")
        .replace("-hf", "")
        .replace("-2T", "-2.0T")
        .replace("9B-fp8", "9B-2.2T-fp8")
        .replace("ibm-fms/", "")
        .replace("instruct_models/", "")
        .replace("Bamba_annealed_models/", "")
    )

    # with st.expander("All models"):
    # with st.form("Configuration selector"):
    with st.sidebar:
        selected_phases = st.multiselect(
            "Choose phase", options=df.phase.unique(), default=df.phase.unique()
        )
        selected_sizes = st.slider(
            label="Choose size",
            min_value=0,
            max_value=df["n_params"].max(),
            value=(0, df["n_params"].max() + 1),
        )
        selected_family = st.multiselect(
            "Choose Family",
            options=df.model_family.unique(),
            default=df.model_family.unique(),
        )

        model_substring = st.text_input("type model substring", value="")
    # submitted = st.form_submit_button("Show model subset")
    # if submitted:
    df = df.query("phase in @selected_phases")
    df = df.query(f"{selected_sizes[0]}<n_params<{selected_sizes[-1]}")
    df = df.query("model_family in @selected_family")
    if model_substring:
        df = df[df["model"].str.contains(model_substring)]

    column_order = [
        col
        for col in df.columns
        if col not in ["model", "MWR", "n_params", "phase", "model_family"]
    ]
    column_order.insert(0, "phase")
    column_order.insert(0, "MWR")
    column_order.insert(0, "model")

    # Apply background gradient to all numeric columns except 'n_params'
    numeric_cols_to_style = [
        col
        for col in df.columns
        if pd.api.types.is_numeric_dtype(df[col]) and col != "n_params"
    ]

    df = df.sort_values("MWR", ascending=False)

    styled_df = df.style.format(format_dict)
    if numeric_cols_to_style:
        styled_df = styled_df.background_gradient(
            cmap="Greens", subset=numeric_cols_to_style
        )

    st.dataframe(
        styled_df,
        use_container_width=True,
        hide_index=True,
        height=850,
        column_order=column_order,
    )

    # st.write("*results taken from paper")

    with st.expander("Docs"):
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
