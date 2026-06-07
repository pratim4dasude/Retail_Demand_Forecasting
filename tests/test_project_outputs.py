import os
import sys
import pandas as pd


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)


from main import (
    load_project_level_summary,
    load_business_recommendations,
    get_best_forecasting_level,
    get_available_entities,
    get_forecast_data,
)


def test_project_level_summary_exists_and_loads():
    summary_df = load_project_level_summary()

    assert isinstance(summary_df, pd.DataFrame)
    assert not summary_df.empty

    required_columns = {
        "forecast_level",
        "scope",
        "number_of_series",
        "best_series",
        "best_mape",
        "weakest_series",
        "weakest_mape",
        "average_mape",
    }

    assert required_columns.issubset(set(summary_df.columns))


def test_project_has_all_forecasting_levels():
    summary_df = load_project_level_summary()

    expected_levels = {"category", "department", "store", "item"}
    actual_levels = set(summary_df["forecast_level"].str.lower())

    assert expected_levels.issubset(actual_levels)


def test_average_mape_values_are_valid():
    summary_df = load_project_level_summary()

    assert summary_df["average_mape"].notna().all()
    assert (summary_df["average_mape"] >= 0).all()


def test_best_forecasting_level_function():
    best_level = get_best_forecasting_level()

    assert isinstance(best_level, dict)
    assert "forecast_level" in best_level
    assert "average_mape" in best_level
    assert "best_series" in best_level
    assert "best_mape" in best_level

    assert best_level["average_mape"] >= 0


def test_business_recommendations_load():
    recommendations_df = load_business_recommendations()

    assert isinstance(recommendations_df, pd.DataFrame)
    assert not recommendations_df.empty

    required_columns = {
        "forecast_level",
        "model_reliability",
        "recommended_use",
        "final_note",
    }

    assert required_columns.issubset(set(recommendations_df.columns))


def test_forecast_data_loads_for_each_level():
    forecast_levels = ["category", "department", "store", "item"]

    for level in forecast_levels:
        forecast_df = get_forecast_data(level)

        assert isinstance(forecast_df, pd.DataFrame)
        assert not forecast_df.empty
        assert "date" in forecast_df.columns
        assert "actual_sales" in forecast_df.columns


def test_available_entities_exist_for_each_level():
    forecast_levels = ["category", "department", "store", "item"]

    for level in forecast_levels:
        entities = get_available_entities(level)

        assert isinstance(entities, list)
        assert len(entities) > 0