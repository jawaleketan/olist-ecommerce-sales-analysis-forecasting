# =============================== Importing libraries =============================== #

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
from statsmodels.tsa.api import ExponentialSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# =============================== Choose and Apply a Forecasting Model =============================== #

# Apply the Holt-Winters Exponential Smoothing model to forecast future sales for the next 12 months.
# Evaluate the model's performance using RMSE and MAPE.
# Generate a plot showing historical sales, test predictions, and future forecasts.

# =============================== Make Predictions & Compare with Actual Sales =============================== #

def perform_sales_forecasting(df):
    """
    Prepares time series data, trains a Holt-Winters forecasting model,
    makes predictions, evaluates the model, and visualizes the forecast.
    """
    print("\n--- Starting Sales Forecasting ---")

    # 1. Prepare Daily Sales Time Series
    # Ensure order_purchase_timestamp is datetime type
    df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])

    # Extract order_purchase_date
    df['order_purchase_date'] = df['order_purchase_timestamp'].dt.date
    df.dropna(subset=['order_purchase_date'], inplace=True)

    # Aggregate sales data daily
    daily_sales = df.groupby('order_purchase_date').agg(
        total_sales=('item_total_revenue', 'sum')
    ).reset_index()

    daily_sales['order_purchase_date'] = pd.to_datetime(daily_sales['order_purchase_date'])

    # Handle missing dates (fill with 0 for total_sales)
    min_date = daily_sales['order_purchase_date'].min()
    max_date = daily_sales['order_purchase_date'].max()
    full_date_range = pd.date_range(start=min_date, end=max_date, freq='D')

    daily_sales = daily_sales.set_index('order_purchase_date')
    daily_sales = daily_sales.reindex(full_date_range, fill_value=0)

    # The final time series DataFrame for forecasting
    daily_sales_ts = daily_sales.copy()
    daily_sales_ts.index.name = 'order_date'
    daily_sales_ts.sort_index(inplace=True)

    print("\nPrepared Daily Sales Time Series (head):")
    print(daily_sales_ts.head().to_markdown(numalign="left", stralign="left"))
    print("\nPrepared Daily Sales Time Series (info):")
    daily_sales_ts.info()

    # 2. Split Data into Training & Testing Sets
    train_size = int(len(daily_sales_ts) * 0.8)
    train_data = daily_sales_ts.iloc[:train_size]['total_sales']
    test_data = daily_sales_ts.iloc[train_size:]['total_sales']

    print(f"\nTraining data size: {len(train_data)} records")
    print(f"Testing data size: {len(test_data)} records")

    # 3. Train the Forecasting Model (Holt-Winters Exponential Smoothing)
    # Using additive trend and additive seasonality, with weekly seasonal_periods=7.
    try:
        model_hw = ExponentialSmoothing(
            train_data,
            seasonal_periods=7,
            trend='add',
            seasonal='add',
            initialization_method="estimated"
        ).fit()
        print("\nHolt-Winters Exponential Smoothing model trained successfully.")
    except Exception as e:
        print(f"\nError training Holt-Winters model: {e}")
        print("This might happen if the training data is too short or has insufficient variation for 'estimated' initialization. Consider alternative initialization_method or check data.")
        return # Exit function if model training fails

    # 4. Make Predictions & Compare with Actual Sales
    # Generate predictions for the test set
    start_pred_date = test_data.index[0]
    end_pred_date = test_data.index[-1]
    predictions_test = model_hw.predict(start=start_pred_date, end=end_pred_date)

    # Generate future sales predictions for the next 12 months (365 days)
    future_start_date = daily_sales_ts.index[-1] + pd.Timedelta(days=1)
    future_end_date = daily_sales_ts.index[-1] + pd.Timedelta(days=365)
    future_predictions = model_hw.predict(start=future_start_date, end=future_end_date)

    print("\nPredictions for Test Set (head):")
    print(predictions_test.head().to_markdown(numalign="left", stralign="left"))
    print("\nFuture Predictions (head):")
    print(future_predictions.head().to_markdown(numalign="left", stralign="left"))

    # 5. Evaluate the model on the test set
    performance_df = pd.DataFrame({'Actual': test_data, 'Predicted': predictions_test})
    performance_df.dropna(inplace=True)

    rmse = np.sqrt(mean_squared_error(performance_df['Actual'], performance_df['Predicted']))
    mape = np.mean(np.abs((performance_df['Actual'] - performance_df['Predicted']) / performance_df['Actual'].replace(0, np.nan))) * 100
    if pd.isna(mape):
        mape = 0.0 # If all actuals are 0, MAPE is undefined or 0.

    print(f"\nModel Evaluation on Test Set:")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE: {mape:.2f}%")

    # 6. Analyze the Forecasting Results (Visualization)
    plt.figure(figsize=(18, 8))
    plt.plot(train_data.index, train_data, label='Training Data', color='blue')
    plt.plot(test_data.index, test_data, label='Actual Test Data', color='green')
    plt.plot(predictions_test.index, predictions_test, label='Predicted Test Data', color='orange', linestyle='--')
    plt.plot(future_predictions.index, future_predictions, label='Future Forecast (12 Months)', color='red', linestyle=':')

    plt.title('Holt-Winters Exponential Smoothing Sales Forecast')
    plt.xlabel('Date')
    plt.ylabel('Total Sales ($)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('holt_winters_sales_forecast.png')
    plt.close()

    print("\n--- Sales Forecasting Complete ---")

# Perform Sales Forecasting
df_merged_cleaned = pd.read_csv("olist_merged_data_cleaned.csv")
perform_sales_forecasting(df_merged_cleaned.copy())