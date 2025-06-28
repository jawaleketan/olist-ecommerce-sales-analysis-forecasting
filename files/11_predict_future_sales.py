import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.api import ExponentialSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from datetime import timedelta


# =============================== Predict Future Sales ======================================= #

# --- Data Loading and Preprocessing (Consolidated and robustified) ---
def load_data_for_task():
    """
    Loads and preprocesses the Olist merged data.
    Attempts to load 'olist_merged_data_cleaned.csv' first.
    If not found, it tries to merge from original Olist datasets.
    Returns the cleaned and preprocessed DataFrame.
    """
    try:
        df = pd.read_csv('olist_merged_data_cleaned.csv')
        print("Loaded 'olist_merged_data_cleaned.csv'")
    except FileNotFoundError:
        print("Warning: 'olist_merged_data_cleaned.csv' not found. Attempting to load and merge original Olist datasets. This might take a moment...")
        try:
            customers = pd.read_csv('olist_customers_dataset.csv')
            orders = pd.read_csv('olist_orders_dataset.csv')
            order_items = pd.read_csv('olist_order_items_dataset.csv')
            products = pd.read_csv('olist_products_dataset.csv')
            sellers = pd.read_csv('olist_sellers_dataset.csv')
            payments = pd.read_csv('olist_order_payments_dataset.csv')
            reviews = pd.read_csv('olist_order_reviews_dataset.csv')
            category_translation = pd.read_csv('product_category_name_translation.csv')

            # Merge DataFrames
            df = pd.merge(orders, customers, on='customer_id', how='left')
            df = pd.merge(df, order_items, on='order_id', how='left')
            df = pd.merge(df, products, on='product_id', how='left')
            df = pd.merge(df, sellers, on='seller_id', how='left')
            df = pd.merge(df, payments, on='order_id', how='left')
            df = pd.merge(df, reviews, on='order_id', how='left')
            df = pd.merge(df, category_translation, on='product_category_name', how='left')

            # Cleaning and Feature Engineering (simplified for dashboard)
            date_cols = [col for col in df.columns if 'date' in col or 'timestamp' in col]
            for col in date_cols:
                df[col] = pd.to_datetime(df[col], errors='coerce')

            df['product_category_name_english'].fillna(df['product_category_name'], inplace=True)
            df['review_comment_message'].fillna('no_comment', inplace=True)
            df['review_comment_title'].fillna('no_comment', inplace=True)
            df['payment_type'].fillna('unknown', inplace=True)
            df['order_status'].fillna('unknown', inplace=True)
            df['product_category_name'].fillna('unknown', inplace=True)

            # Ensure numeric columns are correct and fill small NAs for calculation
            numeric_cols = ['price', 'freight_value', 'product_name_lenght', 'product_description_lenght',
                            'product_photos_qty', 'product_weight_g', 'product_length_cm',
                            'product_height_cm', 'product_width_cm', 'payment_sequential',
                            'payment_installments', 'payment_value', 'review_score']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(df[col].median() if df[col].dtype != 'object' else 0)

            df['item_total_revenue'] = df['price'] + df['freight_value']
            df.dropna(subset=['order_id', 'customer_id', 'order_purchase_timestamp', 'item_total_revenue'], inplace=True)
            df.drop_duplicates(inplace=True)

            # Standardize text columns
            text_cols = ['product_category_name_english', 'customer_city', 'customer_state', 'seller_city', 'seller_state', 'payment_type', 'order_status', 'product_category_name']
            for col in text_cols:
                df[col] = df[col].astype(str).str.lower().str.strip()

            print("Successfully loaded and merged original datasets.")

        except FileNotFoundError as e:
            print(f"Critical Error: Required original Olist CSV files not found: {e}. Please ensure all files are in the directory.")
            return None
        except Exception as e:
            print(f"An error occurred during data loading or merging: {e}")
            return None
    return df

# --- Function to prepare time series data for forecasting ---
def prepare_time_series_data(df):
    """
    Prepares daily sales time series from the cleaned DataFrame.
    """
    df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
    df['order_purchase_date'] = df['order_purchase_timestamp'].dt.date
    df.dropna(subset=['order_purchase_date'], inplace=True)

    daily_sales = df.groupby('order_purchase_date').agg(
        total_sales=('item_total_revenue', 'sum')
    ).reset_index()

    daily_sales['order_purchase_date'] = pd.to_datetime(daily_sales['order_purchase_date'])

    min_date = daily_sales['order_purchase_date'].min()
    max_date = daily_sales['order_purchase_date'].max()
    full_date_range = pd.date_range(start=min_date, end=max_date, freq='D')

    daily_sales = daily_sales.set_index('order_purchase_date')
    daily_sales = daily_sales.reindex(full_date_range, fill_value=0)

    daily_sales_ts = daily_sales.copy()
    daily_sales_ts.index.name = 'order_date'
    daily_sales_ts.sort_index(inplace=True)

    return daily_sales_ts

# --- Task 11: Predict Future Sales ---
def predict_future_sales(df_original):
    print("\n--- Task 11: Predict Future Sales ---")

    # 1. Prepare Time Series Data (re-run from original df to ensure freshness)
    daily_sales_ts = prepare_time_series_data(df_original.copy())

    # 2. Split Data into Training & Testing Sets
    # Using 80/20 split for consistency with previous tasks
    train_size = int(len(daily_sales_ts) * 0.8)
    train_data = daily_sales_ts.iloc[:train_size]['total_sales']
    test_data = daily_sales_ts.iloc[train_size:]['total_sales']

    print(f"\nTraining data size: {len(train_data)} records")
    print(f"Testing data size: {len(test_data)} records")

    # 3. Train the Forecasting Model (Holt-Winters Exponential Smoothing)
    # Based on prior analysis, Holt-Winters is suitable for trends and seasonality.
    try:
        model_hw = ExponentialSmoothing(
            train_data,
            seasonal_periods=7, # Weekly seasonality
            trend='add',        # Additive trend
            seasonal='add',     # Additive seasonality
            initialization_method="estimated" # Let statsmodels estimate initial values
        ).fit()
        print("\nHolt-Winters Exponential Smoothing model trained successfully.")
    except Exception as e:
        print(f"\nError training Holt-Winters model: {e}")
        print("Cannot proceed with forecasting due to model training failure.")
        return None, None, None, None, None # Return Nones if model fails

    # 4. Generate Future Sales Predictions (Next 6-12 months, setting to 12 months/365 days)
    future_forecast_days = 365 # Predicting for next 12 months (approx 365 days)
    future_start_date = daily_sales_ts.index[-1] + pd.Timedelta(days=1)
    future_end_date = daily_sales_ts.index[-1] + pd.Timedelta(days=future_forecast_days)
    future_predictions = model_hw.predict(start=future_start_date, end=future_end_date)

    print(f"\nGenerated future sales predictions for {future_forecast_days} days (next 12 months).")
    print("Future Predictions (head):")
    print(future_predictions.head().to_markdown(numalign="left", stralign="left"))

    # 5. Visualize the Forecasted Sales Trends
    plt.figure(figsize=(18, 8))
    plt.plot(train_data.index, train_data, label='Training Data', color='blue', alpha=0.7)
    plt.plot(test_data.index, test_data, label='Actual Test Data', color='green')
    plt.plot(future_predictions.index, future_predictions, label='Future Forecast (12 Months)', color='red', linestyle=':')
    plt.plot(model_hw.predict(start=train_data.index[0], end=train_data.index[-1]).index, model_hw.predict(start=train_data.index[0], end=train_data.index[-1]), label='In-sample Fit', color='purple', linestyle='--') # Show in-sample fit
    plt.plot(model_hw.predict(start=test_data.index[0], end=test_data.index[-1]).index, model_hw.predict(start=test_data.index[0], end=test_data.index[-1]), label='Test Period Prediction', color='orange', linestyle='-.') # Show test period prediction


    plt.title('Holt-Winters Exponential Smoothing Sales Forecast: Historical, Test & Future')
    plt.xlabel('Date')
    plt.ylabel('Total Sales (R$)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('sales_forecast_combined.png')
    plt.close()
    print("\nSales forecast visualization generated: 'sales_forecast_combined.png'")


    # 6. Validate the Forecast Accuracy (on Test Data)
    predictions_test = model_hw.predict(start=test_data.index[0], end=test_data.index[-1])
    performance_df = pd.DataFrame({'Actual': test_data, 'Predicted': predictions_test})
    performance_df.dropna(inplace=True)

    rmse = np.sqrt(mean_squared_error(performance_df['Actual'], performance_df['Predicted']))
    mape = np.mean(np.abs((performance_df['Actual'] - performance_df['Predicted']) / performance_df['Actual'].replace(0, np.nan))) * 100
    if pd.isna(mape):
        mape = 0.0

    print(f"\nModel Validation on Test Set:")
    print(f"RMSE (Root Mean Squared Error): {rmse:.2f}")
    print(f"MAPE (Mean Absolute Percentage Error): {mape:.2f}%")

    print("--- Predict Future Sales Task Complete ---")
    return future_predictions, rmse, mape, daily_sales_ts, model_hw.predict(start=test_data.index[0], end=test_data.index[-1]) # Return test predictions for separate plot if needed

# Main execution block
if __name__ == "__main__":
    print("--- Executing Task 11: Predict Future Sales ---")
    df = load_data_for_task()

    if df is not None:
        future_preds, rmse_val, mape_val, daily_sales_data, test_preds_plot = predict_future_sales(df.copy())

        if future_preds is not None:
            print("\n--- Summary Report of Sales Predictions ---")
            print("Expected sales growth or decline over the forecasted period:")

            # Analyze trends in future predictions
            if not future_preds.empty:
                initial_forecast_val = future_preds.iloc[0]
                final_forecast_val = future_preds.iloc[-1]
                total_forecast_revenue = future_preds.sum()

                print(f"\nTotal forecasted revenue for the next 12 months: R$ {total_forecast_revenue:,.2f}")
                if final_forecast_val > initial_forecast_val:
                    print(f"The forecast predicts an increasing trend from R$ {initial_forecast_val:,.2f} to R$ {final_forecast_val:,.2f} over the next 12 months.")
                else:
                    print(f"The forecast predicts a decreasing trend from R$ {initial_forecast_val:,.2f} to R$ {final_forecast_val:,.2f} over the next 12 months.")

                # Identify seasonal peaks/troughs in future predictions (monthly aggregation)
                # Check if future_preds has enough data for monthly resampling
                if len(future_preds.index) >= 30: # At least one month of data to resample monthly
                    future_monthly_preds = future_preds.resample('M').sum()
                    if not future_monthly_preds.empty:
                        peak_month = future_monthly_preds.idxmax().strftime('%Y-%m')
                        peak_value = future_monthly_preds.max()
                        trough_month = future_monthly_preds.idxmin().strftime('%Y-%m')
                        trough_value = future_monthly_preds.min()

                        print(f"\nForecasted Peak Sales Month: {peak_month} with Revenue: R$ {peak_value:,.2f}")
                        print(f"Forecasted Trough Sales Month: {trough_month} with Revenue: R$ {trough_value:,.2f}")
                    else:
                        print("\nNot enough data for monthly aggregation in future forecast.")
                else:
                    print("\nFuture forecast period is too short for meaningful monthly aggregation analysis.")


            print("\nBusiness Recommendations based on Sales Forecast:")
            print("- **Inventory & Logistics Planning:** Based on forecasted peaks, prepare for higher inventory levels and scale logistics operations (warehousing, shipping capacity) in advance to meet demand and avoid stockouts. This is crucial for months like [Identify Peak Forecast Months].")
            print("- **Marketing & Promotional Strategies:** For forecasted periods of lower sales (e.g., [Identify Trough Forecast Months]), plan targeted marketing campaigns, special discounts, or product bundles to stimulate demand and mitigate potential revenue dips.")
            print("- **Resource Allocation:** Adjust staffing levels in customer service, sales, and fulfillment centers in line with expected sales fluctuations to optimize operational efficiency and customer experience.")
            print("- **Strategic Product Launches:** Consider launching new products or running major promotional events just before forecasted peak seasons to maximize their impact.")
            print("- **Continuous Monitoring & Refinement:** The forecast is a tool; regularly compare actual sales against predictions. If accuracy deviates significantly (e.g., if RMSE or MAPE are consistently high), re-evaluate the model or incorporate new data/factors (like economic indicators, competitor actions, or marketing spend) to improve its predictive power.")

            print("\n--- Sales Prediction Summary Report Complete ---")
    else:
        print("Skipping sales prediction as data could not be loaded or prepared.")


# =============================== Sales Forecast Visualization ==================================== #

# Re-run necessary data preparation for the plot
# This block will try to load 'olist_merged_data_cleaned.csv' first,
# then fall back to loading and merging original datasets if not found.

df_original = None
try:
    df_original = pd.read_csv('olist_merged_data_cleaned.csv')
    df_original['order_purchase_timestamp'] = pd.to_datetime(df_original['order_purchase_timestamp'], errors='coerce')
    # Ensure item_total_revenue is numeric and fill NaNs
    df_original['item_total_revenue'] = pd.to_numeric(df_original['item_total_revenue'], errors='coerce').fillna(0)
    df_original.dropna(subset=['order_purchase_timestamp'], inplace=True) # Ensure valid timestamps

except FileNotFoundError:
    print("Pre-cleaned data not found for plotting. Attempting to load and merge original datasets (might be slow).")
    try:
        customers = pd.read_csv('olist_customers_dataset.csv')
        orders = pd.read_csv('olist_orders_dataset.csv')
        order_items = pd.read_csv('olist_order_items_dataset.csv')
        products = pd.read_csv('olist_products_dataset.csv')
        sellers = pd.read_csv('olist_sellers_dataset.csv')
        payments = pd.read_csv('olist_order_payments_dataset.csv')
        reviews = pd.read_csv('olist_order_reviews_dataset.csv')
        category_translation = pd.read_csv('product_category_name_translation.csv')

        df_original = pd.merge(orders, customers, on='customer_id', how='left')
        df_original = pd.merge(df_original, order_items, on='order_id', how='left')
        df_original = pd.merge(df_original, products, on='product_id', how='left')
        df_original = pd.merge(df_original, sellers, on='seller_id', how='left')
        df_original = pd.merge(df_original, payments, on='order_id', how='left')
        df_original = pd.merge(df_original, reviews, on='order_id', how='left')
        df_original = pd.merge(df_original, category_translation, on='product_category_name', how='left')

        # Minimal cleaning for time series (match what's needed for item_total_revenue)
        df_original['order_purchase_timestamp'] = pd.to_datetime(df_original['order_purchase_timestamp'], errors='coerce')
        df_original['price'] = pd.to_numeric(df_original['price'], errors='coerce').fillna(0)
        df_original['freight_value'] = pd.to_numeric(df_original['freight_value'], errors='coerce').fillna(0)
        df_original['item_total_revenue'] = df_original['price'] + df_original['freight_value']
        df_original.dropna(subset=['order_purchase_timestamp', 'item_total_revenue'], inplace=True)
        df_original.drop_duplicates(inplace=True)

    except Exception as e:
        print(f"Could not load or regenerate data for plotting: {e}")
        exit() # Cannot plot without data

# Prepare time series data
df_original['order_purchase_date'] = df_original['order_purchase_timestamp'].dt.date
daily_sales = df_original.groupby('order_purchase_date').agg(
    total_sales=('item_total_revenue', 'sum')
).reset_index()
daily_sales['order_purchase_date'] = pd.to_datetime(daily_sales['order_purchase_date'])
min_date = daily_sales['order_purchase_date'].min()
max_date = daily_sales['order_purchase_date'].max()
full_date_range = pd.date_range(start=min_date, end=max_date, freq='D')
daily_sales = daily_sales.set_index('order_purchase_date')
daily_sales = daily_sales.reindex(full_date_range, fill_value=0)
daily_sales_ts = daily_sales.copy()
daily_sales_ts.index.name = 'order_date'
daily_sales_ts.sort_index(inplace=True)

# Split Data (80/20)
train_size = int(len(daily_sales_ts) * 0.8)
train_data = daily_sales_ts.iloc[:train_size]['total_sales']
test_data = daily_sales_ts.iloc[train_size:]['total_sales']

# Train Model
# This try-except block handles potential issues if train_data is too short or problematic for the model
try:
    model_hw = ExponentialSmoothing(
        train_data,
        seasonal_periods=7,
        trend='add',
        seasonal='add',
        initialization_method="estimated"
    ).fit()
except Exception as e:
    print(f"Error training model for plotting: {e}. Plotting will be limited.")
    model_hw = None # Set to None if training fails

# Generate Predictions for plotting
future_predictions = pd.Series()
if model_hw:
    future_forecast_days = 365 # Predicting for next 12 months
    future_start_date = daily_sales_ts.index[-1] + pd.Timedelta(days=1)
    future_end_date = daily_sales_ts.index[-1] + pd.Timedelta(days=future_forecast_days)
    future_predictions = model_hw.predict(start=future_start_date, end=future_end_date)


# Plotting
plt.figure(figsize=(18, 8))
plt.plot(train_data.index, train_data, label='Training Data', color='blue', alpha=0.7)
plt.plot(test_data.index, test_data, label='Actual Test Data', color='green')

if model_hw:
    # Plot in-sample fit
    in_sample_fit = model_hw.predict(start=train_data.index[0], end=train_data.index[-1])
    plt.plot(in_sample_fit.index, in_sample_fit, label='In-sample Fit', color='purple', linestyle='--')

    # Plot test period prediction
    test_predictions_plot = model_hw.predict(start=test_data.index[0], end=test_data.index[-1])
    plt.plot(test_predictions_plot.index, test_predictions_plot, label='Test Period Prediction', color='orange', linestyle='-.')

    # Plot future predictions
    if not future_predictions.empty:
        plt.plot(future_predictions.index, future_predictions, label='Future Forecast (12 Months)', color='red', linestyle=':')
    else:
        print("Future predictions could not be generated for the plot.")
else:
    print("Model did not train, so no predictions or fits are available for the plot.")


plt.title('Holt-Winters Exponential Smoothing Sales Forecast: Historical, Test & Future')
plt.xlabel('Date')
plt.ylabel('Total Sales (R$)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# =============================== Summary report of sales prediction ======================================# 

with open('sales_predictions_summary.txt', 'w') as f:
    f.write("Summary Report of Sales Predictions\n\n")

    f.write("The sales forecasting was performed using the Holt-Winters Exponential Smoothing model on daily sales data, which was prepared to account for missing dates and ensure chronological order. The data was split into an 80% training set and a 20% testing set.\n\n")

    f.write("Summary of Forecasting Results:\n")
    f.write("Prepared Daily Sales Time Series (Head):\n")
    f.write("| order_date | total_sales |\n")
    f.write("|:-----------|:------------|\n")
    f.write("| 2016-09-04 | 0.00 |\n")
    f.write("| 2016-09-05 | 0.00 |\n")
    f.write("| 2016-09-06 | 0.00 |\n")
    f.write("| 2016-09-07 | 0.00 |\n")
    f.write("| 2016-09-08 | 0.00 |\n\n")

    f.write("Time Series Data Information: The dataset covers 729 days (approximately 2 years) with daily sales data.\n")
    f.write("Training and Testing Split: 583 records for training and 146 records for testing.\n")
    f.write("Model: Holt-Winters Exponential Smoothing (additive trend, additive seasonality with a period of 7 for weekly patterns).\n")
    f.write("Future Predictions Generated: Sales forecasts for the next 365 days (12 months) beyond the last available historical date.\n\n")

    f.write("Future Predictions (Head):\n")
    f.write("| ds | yhat |\n")
    f.write("|:-----------|:-------|\n")
    f.write("| 2018-09-03 | 2408.06 |\n")
    f.write("| 2018-09-04 | 2439.11 |\n")
    f.write("| 2018-09-05 | 2435.53 |\n")
    f.write("| 2018-09-06 | 2309.28 |\n")
    f.write("| 2018-09-07 | 2007.82 |\n\n")

    f.write("Validation of Forecast Accuracy (on Test Set):\n")
    f.write("RMSE (Root Mean Squared Error): 1974.77\n")
    f.write("MAPE (Mean Absolute Percentage Error): 362.46%\n")
    f.write("The high MAPE indicates that while the model generally captures the trend and seasonality, its point-to-point predictions on daily data have a significant average percentage error. Daily sales data can be inherently volatile, and zero-sales days might inflate MAPE. RMSE provides a sense of the absolute magnitude of the error.\n\n")

    f.write("Visualized Forecasted Sales Trends: The plot titled 'Holt-Winters Exponential Smoothing Sales Forecast: Historical, Test & Future' provides a visual representation of the analysis. This visualization shows:\n")
    f.write("  - Training Data (Blue): The historical sales data used to train the model, exhibiting a clear upward trend and visible weekly seasonality.\n")
    f.write("  - Actual Test Data (Green): The most recent historical sales data, used to evaluate the model's performance on unseen data.\n")
    f.write("  - In-sample Fit (Purple Dashed): How well the model explains the training data.\n")
    f.write("  - Test Period Prediction (Orange Dot-Dashed): The model's forecast for the test period, showing its ability (or lack thereof) to capture recent trends.\n")
    f.write("  - Future Forecast (Red Dotted): The extrapolated sales trend for the next 12 months, maintaining the identified seasonality and trend from the historical data.\n\n")

    f.write("The plot indicates a general upward trend in sales across the historical period. The future forecast suggests a continuation of this upward trajectory, maintaining the weekly patterns. The model appears to capture the weekly fluctuations, but the deviation between predicted and actual sales in the test set highlights areas for improvement.\n\n")

    f.write("Analysis of Business Implications:\n")
    f.write("Expected Sales Growth/Decline Over the Forecasted Period:\n")
    f.write("  - Total forecasted revenue for the next 12 months: Approximately R$ 628,476.10.\n")
    f.write("  - The forecast predicts a slight decreasing trend from the initial forecast day (R$ 2,408.06) to the final forecast day (R$ 1,745.05) within the 12-month future period. However, considering the overall historical growth, this might indicate a plateauing or a need for higher-level aggregated forecasts (e.g., monthly) for clearer long-term trend visibility.\n")
    f.write("  - Forecasted Peak Sales Month: 2018-09 with Revenue: R$ 59,584.28.\n")
    f.write("  - Forecasted Trough Sales Month: 2019-02 with Revenue: R$ 40,713.84.\n\n")

    f.write("Strategic Recommendations based on Forecast:\n")
    f.write("  - Inventory & Logistics Planning:\n")
    f.write("    - Proactive Stocking: Given the strong seasonal patterns (weekly) and the identified monthly peaks, the business should prepare for higher inventory levels and scale logistics operations (warehousing, shipping capacity) in anticipation of these periods. Special attention should be paid to the forecasted peak month (September 2018) to prevent stockouts and ensure timely deliveries.\n")
    f.write("    - Optimize for Troughs: For forecasted periods of lower sales (e.g., February 2019), inventory can be lean, and logistics resources might be reallocated, helping reduce holding costs.\n")
    f.write("  - Marketing & Promotional Strategies:\n")
    f.write("    - Peak Season Maximization: Intensify marketing campaigns, flash sales, and product promotions leading up to and during the forecasted peak sales months to capitalize on consumer demand.\n")
    f.write("    - Off-Peak Stimulation: During forecasted periods of slow sales, implement targeted discounts, personalized promotions, or bundles to stimulate demand and mitigate potential revenue dips. This is a crucial time for customer re-engagement.\n")
    f.write("  - Resource Allocation:\n")
    f.write("    - Adjust staffing levels in customer service, sales, and fulfillment centers in line with expected sales fluctuations. For example, increase temporary staff for peak periods and optimize schedules during troughs to manage operational costs.\n")
    f.write("  - Strategic Product Launches:\n")
    f.write("    - Consider launching new products or running major promotional events just before forecasted peak seasons to maximize their visibility and impact, leveraging the predicted higher sales activity.\n")
    f.write("  - Continuous Monitoring & Model Refinement:\n")
    f.write("    - Iterative Improvement: The high MAPE highlights the need for continuous improvement. Regularly compare actual sales data against these predictions.\n")
    f.write("    - Error Analysis: Investigate specific instances where predictions significantly deviate from actuals in the test set. Were there unmodeled holidays, major marketing campaigns, or external economic shocks?\n")
    f.write("    - Model Re-evaluation: If prediction accuracy remains unsatisfactory, consider:\n")
    f.write("      - Tuning Parameters: Experiment with different seasonal_periods (e.g., daily, monthly if the data span allows) or seasonality_mode (multiplicative vs. additive) for Holt-Winters.\n")
    f.write("      - Alternative Models: Explore other time series models like SARIMA (Seasonal ARIMA) for more complex patterns, or external libraries like Facebook Prophet, which can explicitly incorporate holiday effects.\n")
    f.write("      - Exogenous Variables: If external factors (e.g., marketing spend, competitor activity, major national holidays) are quantifiable, integrate them into more advanced models to improve accuracy.\n")
    f.write("      - Data Granularity: For highly volatile daily data, it might be beneficial to forecast at a weekly or monthly level for more stable and strategic insights, then disaggregate for operational planning.\n\n")

    f.write("This sales forecast serves as a valuable tool for strategic planning, enabling the business to proactively manage inventory, optimize marketing efforts, and allocate resources more efficiently. However, its high error metrics emphasize the need for ongoing validation and potential model enhancement.\n")
