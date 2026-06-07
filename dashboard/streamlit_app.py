import os
import sys

import pandas as pd
import streamlit as st
import plotly.express as px


# ---------------------------------------------------------
# Make root project importable
# ---------------------------------------------------------

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)


from main import (
    load_project_level_summary,
    load_business_recommendations,
    get_best_forecasting_level,
    get_available_entities,
    filter_forecast_by_entity,
    get_forecast_columns,
)


# ---------------------------------------------------------
# Page setup
# ---------------------------------------------------------

st.set_page_config(
    page_title="M5 Retail Demand Forecasting",
    page_icon="📈",
    layout="wide"
)


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

@st.cache_data
def load_project_summary_cached():
    return load_project_level_summary()


@st.cache_data
def load_recommendations_cached():
    return load_business_recommendations()


@st.cache_data
def get_entities_cached(forecast_level):
    return get_available_entities(forecast_level)


@st.cache_data
def get_filtered_forecast_cached(forecast_level, entity_name):
    df = filter_forecast_by_entity(forecast_level, entity_name)
    df["date"] = pd.to_datetime(df["date"])
    return df


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select page",
    [
        "Project Overview",
        "Forecast Explorer",
        "Business Recommendations"
    ]
)


# ---------------------------------------------------------
# Page 1: Project Overview
# ---------------------------------------------------------

if page == "Project Overview":
    st.title("M5 Retail Demand Forecasting Dashboard")

    st.markdown(
        """
        This dashboard summarizes a multi-level retail demand forecasting project using the M5 Walmart dataset.

        Forecasting was performed at four levels:

        - Category level
        - Department level
        - Store level
        - Item level
        """
    )

    project_summary_df = load_project_summary_cached()
    best_level = get_best_forecasting_level()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Best Forecasting Level",
        str(best_level["forecast_level"]).title()
    )

    col2.metric(
        "Best Avg MAPE",
        f"{best_level['average_mape']:.2f}%"
    )

    col3.metric(
        "Best Series",
        str(best_level["best_series"])
    )

    col4.metric(
        "Best Series MAPE",
        f"{best_level['best_mape']:.2f}%"
    )

    st.subheader("Forecasting Level Summary")
    st.dataframe(project_summary_df, use_container_width=True)

    fig = px.bar(
        project_summary_df,
        x="forecast_level",
        y="average_mape",
        text="average_mape",
        title="Average MAPE by Forecasting Level",
        labels={
            "forecast_level": "Forecast Level",
            "average_mape": "Average MAPE (%)"
        }
    )

    fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    fig.update_layout(yaxis_title="Average MAPE (%)")

    st.plotly_chart(fig, use_container_width=True)

    comparison_df = project_summary_df.melt(
        id_vars=["forecast_level"],
        value_vars=["best_mape", "weakest_mape"],
        var_name="series_type",
        value_name="MAPE"
    )

    fig2 = px.bar(
        comparison_df,
        x="forecast_level",
        y="MAPE",
        color="series_type",
        barmode="group",
        title="Best vs Weakest Forecast Performance by Level",
        labels={
            "forecast_level": "Forecast Level",
            "MAPE": "MAPE (%)",
            "series_type": "Series Type"
        }
    )

    st.plotly_chart(fig2, use_container_width=True)


# ---------------------------------------------------------
# Page 2: Forecast Explorer
# ---------------------------------------------------------

elif page == "Forecast Explorer":
    st.title("Forecast Explorer")

    st.markdown(
        """
        Explore actual vs forecasted demand for each forecasting level.
        """
    )

    forecast_level = st.selectbox(
        "Select forecasting level",
        ["category", "department", "store", "item"]
    )

    entities = get_entities_cached(forecast_level)

    if not entities:
        st.warning("No entities found for this forecasting level.")
        st.stop()

    selected_entity = st.selectbox(
        "Select entity",
        entities
    )

    forecast_df = get_filtered_forecast_cached(
        forecast_level,
        selected_entity
    )

    actual_col, forecast_col = get_forecast_columns(forecast_level)

    st.subheader(f"{forecast_level.title()} Forecast: {selected_entity}")

    col1, col2, col3 = st.columns(3)

    avg_actual = forecast_df[actual_col].mean()
    avg_forecast = forecast_df[forecast_col].mean()
    avg_error = (forecast_df[actual_col] - forecast_df[forecast_col]).abs().mean()

    col1.metric("Avg Actual Sales", f"{avg_actual:.2f}")
    col2.metric("Avg Forecast Sales", f"{avg_forecast:.2f}")
    col3.metric("Avg Absolute Error", f"{avg_error:.2f}")

    plot_df = forecast_df[["date", actual_col, forecast_col]].copy()

    plot_df = plot_df.rename(
        columns={
            actual_col: "Actual Sales",
            forecast_col: "Forecast Sales"
        }
    )

    plot_long_df = plot_df.melt(
        id_vars=["date"],
        value_vars=["Actual Sales", "Forecast Sales"],
        var_name="Series",
        value_name="Sales"
    )

    fig = px.line(
        plot_long_df,
        x="date",
        y="Sales",
        color="Series",
        markers=True,
        title=f"Actual vs Forecast Sales - {selected_entity}"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Forecast Data")
    st.dataframe(forecast_df, use_container_width=True)


# ---------------------------------------------------------
# Page 3: Business Recommendations
# ---------------------------------------------------------

elif page == "Business Recommendations":
    st.title("Business Recommendations")

    recommendations_df = load_recommendations_cached()

    st.markdown(
        """
        The final recommendation depends on the forecasting level and the business use case.
        """
    )

    st.dataframe(recommendations_df, use_container_width=True)

    st.subheader("Final Project Takeaways")

    st.markdown(
        """
        - Store-level forecasting performed best because total demand is more stable after aggregation.
        - Category-level forecasting also performed strongly and is useful for category planning.
        - Department-level forecasting gives more granular visibility but becomes weaker for low-volume departments.
        - Item-level forecasting is the hardest due to sparse demand, zero-sales days, and sudden spikes.
        - For production item-level forecasting, price, promotion, stockout, and rolling lag features should be added.
        """
    )