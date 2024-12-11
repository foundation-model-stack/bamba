import os

from aggregation import get_results_df, parse_args

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

    try:
        from lh_eval_api import EvaluationResultsUploader, RunRecord
    except:
        raise ImportError(
            "lh_eval_api is not installed, "
            "\nwhich is OK if you are not from IBM "
            "\nif you are: install it with"
            "\npip install git+ssh://git@github.ibm.com/IBM-Research-AI/lakehouse-eval-api.git@v1.1.10#egg=lh_eval_api"
        )

    import getpass

    # get your variables
    benchmark = "Bamba-eval"
    score_name = ""
    framework = "LM-Eval_Harness"
    import datetime

    time = datetime.datetime(2024, 12, 11, 8, 53, 38, 409455)
    is_official = False
    owner = getpass.getuser()

    # prepare run records
    run_records = []
    long_df = df.melt(id_vars="model", var_name="dataset", value_name="score")
    result_dicts = long_df.to_dict(orient="records")
    for result in result_dicts:
        run_records.append(
            RunRecord(
                owner=owner,
                started_at=time,
                framework=framework,
                inference_platform="",
                model_name=result["model"],
                execution_env="",
                benchmark=benchmark,
                dataset=result["dataset"],
                task="",
                run_params={"framework": framework},
                score=result["score"],
                score_name=score_name,
            )
        )

    # upload
    uploader = EvaluationResultsUploader(runs=run_records)
    uploader.upload()
