import pandas as pd


def build_final_forecast_table(
    selected_models_df,
    snap_only_forecasts,
    calendar_forecasts,
    entity_col,
    snap_forecast_col="sarimax_forecast",
    calendar_forecast_col="calendar_sarimax_forecast"
):
    """
    Build final forecast table using selected model source.

    selected_models_df should contain:
    - entity_col
    - feature_set
    - model
    """
    final_forecast_list = []

    for _, row in selected_models_df.iterrows():
        entity = row[entity_col]
        feature_set = row["feature_set"]

        if feature_set == "snap_only":
            forecast_df = snap_only_forecasts[entity].copy()
            forecast_df = forecast_df.rename(
                columns={snap_forecast_col: "final_forecast"}
            )
        else:
            forecast_df = calendar_forecasts[entity].copy()
            forecast_df = forecast_df.rename(
                columns={calendar_forecast_col: "final_forecast"}
            )

        forecast_df["feature_set"] = feature_set
        forecast_df["selected_model"] = row["model"]

        final_forecast_list.append(forecast_df)

    final_forecasts_df = pd.concat(
        final_forecast_list,
        ignore_index=True
    )

    return final_forecasts_df


def select_final_stable_models(comparison_df, group_col):
    """
    Select final model per group.

    Priority:
    1. converged=True
    2. lowest MAPE
    """
    final_selected_models_df = (
        comparison_df
        .sort_values(
            [group_col, "converged", "MAPE"],
            ascending=[True, False, True]
        )
        .groupby(group_col, as_index=False)
        .first()
    )

    return final_selected_models_df