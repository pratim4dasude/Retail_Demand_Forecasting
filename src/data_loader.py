import os
import pandas as pd


def load_m5_data(raw_data_dir):
    """
    Load M5 raw dataset files.

    Expected files:
    - calendar.csv
    - sales_train_validation.csv
    - sell_prices.csv
    """
    calendar_path = os.path.join(raw_data_dir, "calendar.csv")
    sales_path = os.path.join(raw_data_dir, "sales_train_validation.csv")
    prices_path = os.path.join(raw_data_dir, "sell_prices.csv")

    calendar_df = pd.read_csv(calendar_path)
    sales_df = pd.read_csv(sales_path)
    prices_df = pd.read_csv(prices_path)

    return calendar_df, sales_df, prices_df


def get_day_columns(sales_df):
    """
    Return all day columns from sales data.
    """
    return [col for col in sales_df.columns if col.startswith("d_")]


def filter_store_sales(sales_df, store_id):
    """
    Filter sales data for a selected store.
    """
    return sales_df[sales_df["store_id"] == store_id].copy()


def load_csv_if_exists(file_path):
    """
    Load a CSV file if it exists.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    return pd.read_csv(file_path)


def create_project_directories(base_dir):
    """
    Create common project directories if they do not exist.
    """
    directories = {
        "raw_data_dir": os.path.join(base_dir, "data", "raw"),
        "processed_data_dir": os.path.join(base_dir, "data", "processed"),
        "outputs_dir": os.path.join(base_dir, "outputs"),
        "models_dir": os.path.join(base_dir, "models"),
    }

    for directory in directories.values():
        os.makedirs(directory, exist_ok=True)

    return directories