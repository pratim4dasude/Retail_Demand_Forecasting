import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def evaluate_forecast(y_true, y_pred, model_name):
    """
    Evaluate forecast using MAE, RMSE, and MAPE.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    y_true_safe = np.where(y_true == 0, np.nan, y_true)
    mape = np.nanmean(np.abs((y_true - y_pred) / y_true_safe)) * 100

    return {
        "model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape
    }


def add_forecast_errors(forecast_df, actual_col="actual_sales", forecast_col="final_forecast"):
    """
    Add error, absolute error, and absolute percentage error columns.
    """
    forecast_df = forecast_df.copy()

    forecast_df["error"] = forecast_df[actual_col] - forecast_df[forecast_col]
    forecast_df["absolute_error"] = forecast_df["error"].abs()

    forecast_df["absolute_percentage_error"] = (
        forecast_df["absolute_error"]
        / forecast_df[actual_col].replace(0, np.nan)
    ) * 100

    return forecast_df


def summarize_forecast_performance(
    forecast_df,
    group_col,
    actual_col="actual_sales",
    forecast_col="final_forecast"
):
    """
    Summarize forecast performance for each group.
    """
    forecast_df = add_forecast_errors(
        forecast_df=forecast_df,
        actual_col=actual_col,
        forecast_col=forecast_col
    )

    summary_df = (
        forecast_df
        .groupby(group_col, as_index=False)
        .agg(
            forecast_days=("date", "count"),
            avg_actual_sales=(actual_col, "mean"),
            avg_forecast_sales=(forecast_col, "mean"),
            avg_absolute_error=("absolute_error", "mean"),
            avg_percentage_error=("absolute_percentage_error", "mean")
        )
        .sort_values("avg_percentage_error")
    )

    return summary_df