import pandas as pd

from src.train_sarimax import run_sarimax_pipeline


def default_sarimax_configs():
    """
    Standard SARIMAX configs used across category, department, and store levels.
    """
    return [
        {"order": (1, 1, 1), "seasonal_order": (1, 1, 1, 7)},
        {"order": (1, 1, 0), "seasonal_order": (1, 1, 1, 7)},
        {"order": (0, 1, 1), "seasonal_order": (1, 1, 1, 7)},
        {"order": (1, 1, 1), "seasonal_order": (0, 1, 1, 7)},
        {"order": (1, 1, 1), "seasonal_order": (1, 1, 0, 7)},
        {"order": (0, 1, 1), "seasonal_order": (0, 1, 1, 7)},
    ]


def item_level_sarimax_configs():
    """
    Smaller SARIMAX config set for item-level forecasting.
    """
    return [
        {"order": (1, 1, 1), "seasonal_order": (1, 1, 1, 7)},
        {"order": (0, 1, 1), "seasonal_order": (1, 1, 1, 7)},
        {"order": (1, 1, 1), "seasonal_order": (0, 1, 1, 7)},
        {"order": (0, 1, 1), "seasonal_order": (0, 1, 1, 7)},
    ]


def tune_sarimax_models(
    ts,
    entity_name,
    configs,
    forecast_horizon=28,
    sales_col="sales",
    model_prefix="SARIMAX"
):
    """
    Tune multiple SARIMAX configurations for one time series.
    """
    results = []
    models = {}
    forecasts = {}

    for config in configs:
        order = config["order"]
        seasonal_order = config["seasonal_order"]

        model_label = f"{model_prefix} {order}{seasonal_order} - {entity_name}"

        print("Training:", model_label)

        try:
            metrics, forecast_df, model_result = run_sarimax_pipeline(
                ts=ts,
                entity_name=entity_name,
                forecast_horizon=forecast_horizon,
                order=order,
                seasonal_order=seasonal_order,
                sales_col=sales_col,
                model_prefix=model_prefix
            )

            metrics["model_label"] = model_label

            results.append(metrics)
            models[model_label] = model_result
            forecasts[model_label] = forecast_df

            print("Done:", model_label)
            print("MAPE:", round(metrics["MAPE"], 2), "%")
            print("Converged:", metrics["converged"])
            print("-" * 50)

        except Exception as error:
            print("Failed:", model_label)
            print("Error:", error)
            print("-" * 50)

    results_df = pd.DataFrame(results)

    if results_df.empty:
        return results_df, models, forecasts

    results_df = results_df.sort_values("MAPE")

    return results_df, models, forecasts


def select_best_model(results_df):
    """
    Select best SARIMAX model by MAPE.
    """
    if results_df.empty:
        return None

    return results_df.sort_values("MAPE").iloc[0]