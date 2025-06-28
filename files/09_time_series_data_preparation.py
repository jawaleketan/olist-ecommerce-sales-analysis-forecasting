# =============================== Importing libraries =============================== #

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Load the cleaned dataset
df_cleaned = pd.read_csv('olist_merged_data_cleaned.csv')

# =============================== Prepare Time Series Data =============================== #

# =============================== Extract & Format Date Information =============================== #

# Extract & Format Date Information
# Ensure 'order_purchase_timestamp' is datetime
df_cleaned['order_purchase_timestamp'] = pd.to_datetime(df_cleaned['order_purchase_timestamp'])
df_cleaned['order_purchase_date'] = df_cleaned['order_purchase_timestamp'].dt.date # Extract just the date part
df_cleaned['order_purchase_year'] = df_cleaned['order_purchase_timestamp'].dt.year
df_cleaned['order_purchase_month'] = df_cleaned['order_purchase_timestamp'].dt.month
df_cleaned['order_purchase_day_of_week'] = df_cleaned['order_purchase_timestamp'].dt.dayofweek # Monday=0, Sunday=6
df_cleaned['order_purchase_day_of_month'] = df_cleaned['order_purchase_timestamp'].dt.day

print("DataFrame with new date components (head):")
print(df_cleaned[['order_purchase_timestamp', 'order_purchase_date', 'order_purchase_year', 'order_purchase_month', 'order_purchase_day_of_week', 'order_purchase_day_of_month']].head().to_markdown(index=False, numalign="left", stralign="left"))

# =============================== Aggregate Sales Data by Date =============================== #

# Aggregate total sales revenue by the full date (not just date components)
daily_sales = df_cleaned.groupby('order_purchase_date').agg(
    total_sales=('item_total_revenue', 'sum')
).reset_index()

# Convert 'order_purchase_date' back to datetime objects for time series operations
daily_sales['order_purchase_date'] = pd.to_datetime(daily_sales['order_purchase_date'])

print("\nDaily Sales (head):")
print(daily_sales.head().to_markdown(index=False, numalign="left", stralign="left"))

# =============================== Handle Missing or Incomplete Data =============================== #

# Create a full date range from the min to max date in the dataset
min_date = daily_sales['order_purchase_date'].min()
max_date = daily_sales['order_purchase_date'].max()
full_date_range = pd.date_range(start=min_date, end=max_date, freq='D')

# Set 'order_purchase_date' as index for reindexing
daily_sales = daily_sales.set_index('order_purchase_date')

# Reindex the daily_sales DataFrame with the full date range and fill missing values with 0
daily_sales = daily_sales.reindex(full_date_range, fill_value=0)

# Reset index to make 'Date' a column again, then rename for clarity
daily_sales = daily_sales.reset_index()
daily_sales.rename(columns={'index': 'order_date'}, inplace=True)

print("\nDaily Sales after handling missing dates (head and tail):")
print(daily_sales.head().to_markdown(index=False, numalign="left", stralign="left"))
print(daily_sales.tail().to_markdown(index=False, numalign="left", stralign="left"))

# =============================== Check for Data Trends & Seasonality =============================== #

# Check for Data Trends & Seasonality (Visualize)
plt.figure(figsize=(15, 7))
sns.lineplot(x='order_date', y='total_sales', data=daily_sales)
plt.title('Daily Total Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Total Sales ($)')
plt.grid(True)
plt.tight_layout()
plt.savefig('daily_total_sales_timeseries.png')
plt.show()
plt.close()

# Aggregate to weekly and monthly for different perspectives on trends/seasonality
weekly_sales = daily_sales.resample('W', on='order_date').agg(total_sales=('total_sales', 'sum'))
monthly_sales = daily_sales.resample('M', on='order_date').agg(total_sales=('total_sales', 'sum'))

plt.figure(figsize=(15, 7))
sns.lineplot(x=weekly_sales.index, y='total_sales', data=weekly_sales)
plt.title('Weekly Total Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Total Sales ($)')
plt.grid(True)
plt.tight_layout()
plt.savefig('weekly_total_sales_timeseries.png')
plt.show()
plt.close()

plt.figure(figsize=(15, 7))
sns.lineplot(x=monthly_sales.index, y='total_sales', data=monthly_sales)
plt.title('Monthly Total Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Total Sales ($)')
plt.grid(True)
plt.tight_layout()
plt.savefig('monthly_total_sales_timeseries.png')
plt.show()
plt.close()

# =============================== Ensure Data is Sorted and Indexed Properly =============================== #

# This step is performed after filling missing dates, setting 'order_date' as index for the final TS data.
daily_sales_ts = daily_sales.set_index('order_date')
print("\nDaily Sales with Date as Index (head for saving):")
print(daily_sales_ts.head().to_markdown(numalign="left", stralign="left"))

# =============================== Save the Processed Dataset for Analysis =============================== #

daily_sales_ts.to_csv('daily_sales_time_series.csv')
print("\n'daily_sales_time_series.csv' saved successfully.")

daily_sales_ts.head()