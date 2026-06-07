import pandas as pd


def add_event_flag(df, event_col="event_name_1"):
    """
    Add binary event flag.
    """
    df = df.copy()
    df["is_event"] = df[event_col].notna().astype(int)
    return df


def create_calendar_dummies(df):
    """
    Create weekday and month dummy variables.
    """
    df = df.copy()

    weekday_dummies = pd.get_dummies(
        df["weekday"],
        prefix="weekday",
        drop_first=True,
        dtype=int
    )

    month_dummies = pd.get_dummies(
        df["month"].astype(str),
        prefix="month",
        drop_first=True,
        dtype=int
    )

    df = pd.concat([df, weekday_dummies, month_dummies], axis=1)

    return df


def prepare_basic_timeseries(
    df,
    entity_col,
    entity_name,
    date_col="date",
    sales_col="sales",
    exog_col="snap_CA"
):
    """
    Prepare daily time series with sales and one exogenous variable.
    """
    entity_df = df[df[entity_col] == entity_name].copy()
    entity_df = entity_df.sort_values(date_col)

    ts = entity_df[[date_col, sales_col, exog_col]].copy()
    ts = ts.set_index(date_col)
    ts = ts.asfreq("D")
    ts = ts.fillna(0)

    return ts


def prepare_calendar_feature_timeseries(
    df,
    entity_col,
    entity_name,
    date_col="date",
    sales_col="sales",
    snap_col="snap_CA"
):
    """
    Prepare daily time series with SNAP, weekday, month, and event features.
    """
    entity_df = df[df[entity_col] == entity_name].copy()
    entity_df = entity_df.sort_values(date_col)

    entity_df["is_event"] = entity_df["event_name_1"].notna().astype(int)

    ts = entity_df[
        [date_col, sales_col, snap_col, "weekday", "month", "is_event"]
    ].copy()

    ts = ts.set_index(date_col)
    ts = ts.asfreq("D")

    weekday_dummies = pd.get_dummies(
        ts["weekday"],
        prefix="weekday",
        drop_first=True,
        dtype=int
    )

    month_dummies = pd.get_dummies(
        ts["month"].astype(str),
        prefix="month",
        drop_first=True,
        dtype=int
    )

    ts = pd.concat(
        [
            ts[[sales_col, snap_col, "is_event"]],
            weekday_dummies,
            month_dummies
        ],
        axis=1
    )

    ts = ts.fillna(0)
    ts = ts.apply(pd.to_numeric, errors="coerce").fillna(0)
    ts = ts.astype(float)

    return ts