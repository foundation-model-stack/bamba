import argparse
import glob
import json
import os

import pandas as pd
from metrics_mapping import scenario2metric
from normalizations import get_hfv2_noramlized_scores, hfv2_tasks, needs_normalization
from pretty_names import name2tag

from evaluation.aggregation_utils import handle_duplicates


def get_results_df(res_dir_paths, results_from_papers_path):
    res_list = []
    for res_dir in res_dir_paths:
        res_file_paths = glob.glob(f"{res_dir}/**/results_*", recursive=True)

        for file_path in res_file_paths:
            all_artifacts = json.load(open(file_path, "r"))
            res_dict = all_artifacts["results"]
            model_name = all_artifacts["model_name"]

            all_res_entries = list(res_dict.keys())

            if (  # HFV2 leaderboard should be normalized
                needs_normalization(all_res_entries)
            ):  # all entries in the leaderboard has this prefix
                for hfv2_task in hfv2_tasks:
                    if " " in res_dict.get(f"leaderboard_{hfv2_task}", {}).keys():
                        # some results comes without a score, this is how we recognize them
                        continue

                    if any([hfv2_task in entry for entry in all_res_entries]):
                        score = get_hfv2_noramlized_scores(
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
                    # dropping the aggregate here
                    if scenario not in scenario2metric.keys():
                        continue

                    metric_key = [
                        key
                        for key in list(res.keys())
                        if scenario2metric[scenario] == key.replace(",none", "")
                        and "stderr" not in key
                    ]

                    assert len(metric_key) == 1, "More/Less than one metric?"

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

    # TODO: aggregating subtasks
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

    df_from_papers = pd.read_csv(results_from_papers_path)
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
    res_df["model"] = res_df["model"].apply(
        lambda x: x.replace("/dccstor/fme/users/yotam/models/", "ibm-fms/")
    )

    def calculate_win_rate(series):
        assert len(series) > 1, "no meaning for a win rate with only one object"

        def win_rate(x):
            win_count = sum(1 for value in series if x > value)
            return win_count / (len(series) - 1)

        return series.transform(win_rate)

    res_df["wr"] = res_df.groupby(["scenario"])["score"].transform(calculate_win_rate)

    mean_df = pd.DataFrame(columns=res_df.columns)
    mean_df = res_df.groupby(["model"]).agg({"wr": "mean"}).reset_index()
    mean_df["score"] = mean_df["wr"]
    mean_df["scenario"] = "MWR"
    res_df = pd.concat([res_df, mean_df]).drop(columns=["wr"])

    # Pivot the DataFrame
    df_pivot_score = res_df.pivot(
        index="model", columns="scenario", values=["score"]
    ).reset_index()
    flat_index = [
        "model" if level0 == "model" else level1
        for level0, level1 in df_pivot_score.columns
    ]
    df_pivot_score.columns = flat_index
    # df_pivot_score.to_csv("output/combined_results.csv", index=False)

    return df_pivot_score


def parse_args():
    parser = argparse.ArgumentParser(description="Run leaderboard evaluation.")

    parser.add_argument(
        "--output_dir_path",
        default="debug",
        help="Output directory path",
    )

    parser.add_argument(
        "--res_dirs",
        nargs="+",
        default=[
            "Bamba_eval",
            "Bamba_eval_last_models",
        ],
        help="Task name (default: leaderboard)",
    )

    return parser.parse_args()


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
