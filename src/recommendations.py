import pandas as pd


def create_business_recommendations():
    """
    Create final business recommendation table.
    """
    recommendations_df = pd.DataFrame([
        {
            "forecast_level": "store",
            "model_reliability": "high",
            "recommended_use": "overall demand planning, staffing, inventory allocation",
            "final_note": "Best performing level due to aggregation stability"
        },
        {
            "forecast_level": "category",
            "model_reliability": "high",
            "recommended_use": "category-level replenishment and sales planning",
            "final_note": "Strong performance for high-volume categories"
        },
        {
            "forecast_level": "department",
            "model_reliability": "medium",
            "recommended_use": "department-level stock planning",
            "final_note": "Works well for high-volume departments, weaker for low-volume ones"
        },
        {
            "forecast_level": "item",
            "model_reliability": "low to medium",
            "recommended_use": "baseline item-level forecasting only",
            "final_note": "Needs price, promotion, stockout, and ML-based features"
        }
    ])

    return recommendations_df


def create_project_summary_text():
    """
    Return final project summary text.
    """
    summary = """
    This project built a multi-level retail demand forecasting pipeline using the M5 Walmart dataset.

    Forecasting was performed at category, department, store, and item levels.
    Store-level and category-level forecasting performed best because aggregated demand is more stable.
    Department-level forecasting worked well for high-volume departments but weakened for low-volume departments.
    Item-level forecasting was the hardest due to sparse demand, zero-sales days, and sudden demand spikes.

    SARIMAX captured weekly seasonality well at aggregated levels.
    For item-level forecasting, future improvements should include price, promotions, stock availability,
    rolling lag features, and machine learning models such as LightGBM or XGBoost.
    """

    return summary.strip()