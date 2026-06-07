import numpy as np
import pandas as pd

from src.evaluate import evaluate_forecast


def run_baseline_forecasts(
    ts,
    entity_name,
    forecast_horizon=28,
    sales_col="sales"
):
    """
    Run naive and 7-day moving average baseline forecasts.

    Parameters:
    - ts: time series DataFrame with date index and sales column
    - entity_name: category, department, store, or item name
    - forecast_horizon: number of days to forecast
    - sales_col: sales column name
    """

    train_data = ts.iloc[:-forecast_horizon].copy()
    test_data = ts.iloc[-forecast_horizon:].copy()

    y_train = train_data[sales_col]
    y_test = test_data[sales_col]

    last_observed_value = y_train.iloc[-1]
    naive_forecast = np.repeat(last_observed_value, forecast_horizon)

    last_7_day_avg = y_train.iloc[-7:].mean()
    moving_avg_forecast = np.repeat(last_7_day_avg, forecast_horizon)

    naive_results = evaluate_forecast(
        y_true=y_test.values,
        y_pred=naive_forecast,
        model_name=f"Naive Forecast - {entity_name}"
    )

    moving_avg_results = evaluate_forecast(
        y_true=y_test.values,
        y_pred=moving_avg_forecast,
        model_name=f"7-Day Moving Average - {entity_name}"
    )

    forecast_df = pd.DataFrame({
        "date": y_test.index,
        "entity": entity_name,
        "actual_sales": y_test.values,
        "naive_forecast": naive_forecast,
        "moving_avg_forecast": moving_avg_forecast
    })

    return [naive_results, moving_avg_results], forecast_df