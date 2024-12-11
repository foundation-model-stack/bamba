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
