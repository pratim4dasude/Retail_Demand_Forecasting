import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.evaluate import evaluate_forecast


def train_sarimax_model(
    y_train,
    X_train=None,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 7),
    maxiter=100
):
    """
    Train a SARIMAX model.
    """
    model = SARIMAX(
        y_train,
        exog=X_train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    result = model.fit(disp=False, maxiter=maxiter)

    return result


def forecast_sarimax_model(
    model_result,
    steps,
    X_test=None,
    clip_negative=True
):
    """
    Generate SARIMAX forecast.
    """
    forecast_result = model_result.get_forecast(
        steps=steps,
        exog=X_test
    )

    forecast_values = forecast_result.predicted_mean

    if clip_negative:
        forecast_values = forecast_values.clip(lower=0)

    return forecast_values


def run_sarimax_pipeline(
    ts,
    entity_name,
    forecast_horizon=28,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 7),
    sales_col="sales",
    model_prefix="SARIMAX"
):
    """
    Run complete SARIMAX train, forecast, and evaluation pipeline.
    """
    train_data = ts.iloc[:-forecast_horizon].copy()
    test_data = ts.iloc[-forecast_horizon:].copy()

    y_train = train_data[sales_col]
    y_test = test_data[sales_col]

    X_train = train_data.drop(columns=[sales_col])
    X_test = test_data.drop(columns=[sales_col])

    model_result = train_sarimax_model(
        y_train=y_train,
        X_train=X_train,
        order=order,
        seasonal_order=seasonal_order
    )

    forecast_values = forecast_sarimax_model(
        model_result=model_result,
        steps=forecast_horizon,
        X_test=X_test,
        clip_negative=True
    )

    metrics = evaluate_forecast(
        y_true=y_test.values,
        y_pred=forecast_values.values,
        model_name=f"{model_prefix} {order}{seasonal_order} - {entity_name}"
    )

    metrics["AIC"] = model_result.aic
    metrics["order"] = order
    metrics["seasonal_order"] = seasonal_order
    metrics["converged"] = model_result.mle_retvals.get("converged", None)

    forecast_df = pd.DataFrame({
        "date": y_test.index,
        "entity": entity_name,
        "actual_sales": y_test.values,
        "forecast": forecast_values.values
    })

    return metrics, forecast_df, model_result