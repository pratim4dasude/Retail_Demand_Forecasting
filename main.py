import os
import pandas as pd


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
MODELS_DIR = os.path.join(BASE_DIR, "models")


# ---------------------------------------------------------
# Output file paths
# ---------------------------------------------------------

CATEGORY_SUMMARY_PATH = os.path.join(
    OUTPUTS_DIR,
    "final_category_forecasting_summary_CA_1.csv"
)

DEPARTMENT_SUMMARY_PATH = os.path.join(
    OUTPUTS_DIR,
    "final_department_forecasting_summary_CA_1.csv"
)

STORE_SUMMARY_PATH = os.path.join(
    OUTPUTS_DIR,
    "final_store_forecasting_summary.csv"
)

ITEM_SUMMARY_PATH = os.path.join(
    OUTPUTS_DIR,
    "final_item_forecasting_summary_top_10_CA_1.csv"
)

PROJECT_LEVEL_SUMMARY_PATH = os.path.join(
    OUTPUTS_DIR,
    "final_project_level_summary.csv"
)

BUSINESS_RECOMMENDATIONS_PATH = os.path.join(
    OUTPUTS_DIR,
    "final_business_recommendations.csv"
)


# Forecast output paths
CATEGORY_FORECAST_PATH = os.path.join(
    OUTPUTS_DIR,
    "all_category_tuned_sarimax_forecasts_CA_1.csv"
)

DEPARTMENT_FORECAST_PATH = os.path.join(
    OUTPUTS_DIR,
    "final_department_sarimax_forecasts_CA_1.csv"
)

STORE_FORECAST_PATH = os.path.join(
    OUTPUTS_DIR,
    "final_store_sarimax_forecasts.csv"
)

ITEM_FORECAST_PATH = os.path.join(
    OUTPUTS_DIR,
    "final_item_sarimax_forecasts_top_10_CA_1.csv"
)


# ---------------------------------------------------------
# Utility functions
# ---------------------------------------------------------

def load_csv(path):
    """
    Load a CSV file with a clear error message if missing.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"File not found: {path}\n"
            "Please run the related notebook first so this output file is generated."
        )

    return pd.read_csv(path)


def get_project_paths():
    """
    Return project path dictionary.
    """
    return {
        "base_dir": BASE_DIR,
        "raw_data_dir": RAW_DATA_DIR,
        "processed_data_dir": PROCESSED_DATA_DIR,
        "outputs_dir": OUTPUTS_DIR,
        "models_dir": MODELS_DIR,
    }


# ---------------------------------------------------------
# Summary loaders
# ---------------------------------------------------------

def load_category_summary():
    return load_csv(CATEGORY_SUMMARY_PATH)


def load_department_summary():
    return load_csv(DEPARTMENT_SUMMARY_PATH)


def load_store_summary():
    return load_csv(STORE_SUMMARY_PATH)


def load_item_summary():
    return load_csv(ITEM_SUMMARY_PATH)


def load_project_level_summary():
    return load_csv(PROJECT_LEVEL_SUMMARY_PATH)


def load_business_recommendations():
    return load_csv(BUSINESS_RECOMMENDATIONS_PATH)


def load_all_summary_tables():
    """
    Load all final summary tables.
    """
    return {
        "category": load_category_summary(),
        "department": load_department_summary(),
        "store": load_store_summary(),
        "item": load_item_summary(),
        "project_level": load_project_level_summary(),
        "recommendations": load_business_recommendations(),
    }


# ---------------------------------------------------------
# Forecast loaders
# ---------------------------------------------------------

def load_category_forecasts():
    return load_csv(CATEGORY_FORECAST_PATH)


def load_department_forecasts():
    return load_csv(DEPARTMENT_FORECAST_PATH)


def load_store_forecasts():
    return load_csv(STORE_FORECAST_PATH)


def load_item_forecasts():
    return load_csv(ITEM_FORECAST_PATH)


def load_all_forecast_tables():
    """
    Load all final forecast tables.
    """
    return {
        "category": load_category_forecasts(),
        "department": load_department_forecasts(),
        "store": load_store_forecasts(),
        "item": load_item_forecasts(),
    }


# ---------------------------------------------------------
# Dashboard helper functions
# ---------------------------------------------------------

def get_best_forecasting_level():
    """
    Return the forecasting level with the lowest average MAPE.
    """
    project_summary_df = load_project_level_summary()

    best_row = project_summary_df.sort_values("average_mape").iloc[0]

    return {
        "forecast_level": best_row["forecast_level"],
        "average_mape": best_row["average_mape"],
        "best_series": best_row["best_series"],
        "best_mape": best_row["best_mape"],
    }


def get_level_summary(forecast_level):
    """
    Return summary for one forecasting level.
    """
    project_summary_df = load_project_level_summary()

    level_df = project_summary_df[
        project_summary_df["forecast_level"] == forecast_level
    ]

    if level_df.empty:
        raise ValueError(f"Invalid forecast level: {forecast_level}")

    return level_df


def get_forecast_data(forecast_level):
    """
    Return forecast data based on selected forecasting level.
    """
    forecast_level = forecast_level.lower()

    if forecast_level == "category":
        return load_category_forecasts()

    if forecast_level == "department":
        return load_department_forecasts()

    if forecast_level == "store":
        return load_store_forecasts()

    if forecast_level == "item":
        return load_item_forecasts()

    raise ValueError(
        "Invalid forecast level. Choose from: category, department, store, item."
    )


def get_available_entities(forecast_level):
    """
    Return available entity names for selected forecasting level.
    """
    df = get_forecast_data(forecast_level)

    forecast_level = forecast_level.lower()

    if forecast_level == "category":
        if "category" in df.columns:
            return sorted(df["category"].dropna().unique().tolist())
        if "cat_id" in df.columns:
            return sorted(df["cat_id"].dropna().unique().tolist())

    if forecast_level == "department":
        return sorted(df["dept_id"].dropna().unique().tolist())

    if forecast_level == "store":
        return sorted(df["store_id"].dropna().unique().tolist())

    if forecast_level == "item":
        return sorted(df["item_id"].dropna().unique().tolist())

    return []


def filter_forecast_by_entity(forecast_level, entity_name):
    """
    Filter forecast table for one selected entity.
    """
    df = get_forecast_data(forecast_level)

    forecast_level = forecast_level.lower()

    if forecast_level == "category":
        if "category" in df.columns:
            return df[df["category"] == entity_name].copy()
        if "cat_id" in df.columns:
            return df[df["cat_id"] == entity_name].copy()

    if forecast_level == "department":
        return df[df["dept_id"] == entity_name].copy()

    if forecast_level == "store":
        return df[df["store_id"] == entity_name].copy()

    if forecast_level == "item":
        return df[df["item_id"] == entity_name].copy()

    raise ValueError(f"Invalid forecast level or entity: {forecast_level}, {entity_name}")


def get_forecast_columns(forecast_level):
    """
    Return actual and forecast column names for a selected level.
    """
    forecast_level = forecast_level.lower()

    if forecast_level == "category":
        return "actual_sales", "sarimax_forecast"

    if forecast_level == "department":
        return "actual_sales", "sarimax_forecast"

    if forecast_level == "store":
        return "actual_sales", "final_forecast"

    if forecast_level == "item":
        return "actual_sales", "final_forecast"

    raise ValueError(f"Invalid forecast level: {forecast_level}")


# ---------------------------------------------------------
# Quick CLI test
# ---------------------------------------------------------

if __name__ == "__main__":
    print("M5 Retail Demand Forecasting Project")
    print("-" * 50)

    paths = get_project_paths()

    print("Base directory:", paths["base_dir"])
    print("Outputs directory:", paths["outputs_dir"])
    print()

    try:
        project_summary = load_project_level_summary()
        print("Project-level summary loaded successfully.")
        print(project_summary)

        print()
        best_level = get_best_forecasting_level()
        print("Best forecasting level:")
        print(best_level)

    except Exception as error:
        print("Error while loading project outputs:")
        print(error)