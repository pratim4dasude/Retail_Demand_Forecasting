import pandas as pd


def convert_sales_wide_to_long(sales_df, day_columns):
    """
    Convert M5 sales data from wide format to long format.

    Input format:
    item_id | store_id | d_1 | d_2 | ...

    Output format:
    item_id | store_id | d | sales
    """
    id_columns = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]

    sales_long_df = sales_df.melt(
        id_vars=id_columns,
        value_vars=day_columns,
        var_name="d",
        value_name="sales"
    )

    return sales_long_df


def merge_calendar_features(sales_long_df, calendar_df, snap_columns=None):
    """
    Merge calendar information into long sales data.
    """
    if snap_columns is None:
        snap_columns = ["snap_CA"]

    calendar_columns = [
        "d",
        "date",
        "wm_yr_wk",
        "weekday",
        "wday",
        "month",
        "year",
        "event_name_1",
        "event_type_1",
    ] + snap_columns

    merged_df = sales_long_df.merge(
        calendar_df[calendar_columns],
        on="d",
        how="left"
    )

    merged_df["date"] = pd.to_datetime(merged_df["date"])

    return merged_df


def aggregate_category_level(store_sales_long_df):
    """
    Aggregate item-level sales into category-level daily sales.
    """
    category_daily_df = (
        store_sales_long_df
        .groupby(["date", "cat_id"], as_index=False)
        .agg(
            sales=("sales", "sum"),
            snap_CA=("snap_CA", "max")
        )
    )

    return category_daily_df


def aggregate_department_level(store_sales_long_df):
    """
    Aggregate item-level sales into department-level daily sales.
    """
    department_daily_df = (
        store_sales_long_df
        .groupby(["date", "dept_id", "cat_id"], as_index=False)
        .agg(
            sales=("sales", "sum"),
            snap_CA=("snap_CA", "max")
        )
    )

    return department_daily_df


def aggregate_store_level(sales_long_df):
    """
    Aggregate item-level sales into store-level daily sales.
    Assumes snap column already exists.
    """
    store_daily_df = (
        sales_long_df
        .groupby(["date", "store_id", "state_id"], as_index=False)
        .agg(
            sales=("sales", "sum"),
            snap=("snap", "max")
        )
    )

    return store_daily_df


def add_store_snap_column(sales_long_df):
    """
    Create one unified SNAP column based on store state.
    """
    sales_long_df = sales_long_df.copy()

    sales_long_df["snap"] = 0

    sales_long_df.loc[
        sales_long_df["state_id"] == "CA", "snap"
    ] = sales_long_df.loc[
        sales_long_df["state_id"] == "CA", "snap_CA"
    ]

    sales_long_df.loc[
        sales_long_df["state_id"] == "TX", "snap"
    ] = sales_long_df.loc[
        sales_long_df["state_id"] == "TX", "snap_TX"
    ]

    sales_long_df.loc[
        sales_long_df["state_id"] == "WI", "snap"
    ] = sales_long_df.loc[
        sales_long_df["state_id"] == "WI", "snap_WI"
    ]

    return sales_long_df


def select_top_items(store_sales_df, day_columns, top_n=10):
    """
    Select top N items by total sales.
    """
    store_sales_df = store_sales_df.copy()
    store_sales_df["total_sales"] = store_sales_df[day_columns].sum(axis=1)

    top_items_df = (
        store_sales_df[
            ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id", "total_sales"]
        ]
        .sort_values("total_sales", ascending=False)
        .reset_index(drop=True)
    )

    selected_item_ids = top_items_df.head(top_n)["id"].tolist()

    selected_items_df = store_sales_df[
        store_sales_df["id"].isin(selected_item_ids)
    ].copy()

    return top_items_df, selected_items_df