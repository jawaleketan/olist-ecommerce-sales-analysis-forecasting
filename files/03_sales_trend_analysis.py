# =============================== Importing libraries =============================== #

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# =============================== Analyze Monthly and Yearly Sales Trends =============================== #

# Load the cleaned dataset
df_cleaned = pd.read_csv('olist_merged_data_cleaned.csv')

# Ensure the order_purchase_timestamp is datetime type
df_cleaned['order_purchase_timestamp'] = pd.to_datetime(df_cleaned['order_purchase_timestamp'])

# Aggregate Sales Data by Month and Year

# Create a 'YYYY-MM' format for easier grouping and plotting of monthly trends
df_cleaned['order_purchase_year_month'] = df_cleaned['order_purchase_timestamp'].dt.to_period('M')

# Aggregate by month and year
monthly_sales = df_cleaned.groupby('order_purchase_year_month')['payment_value'].sum().reset_index()
monthly_sales['order_purchase_year_month'] = monthly_sales['order_purchase_year_month'].dt.to_timestamp() # Convert back to timestamp for plotting

# Aggregate by year
yearly_sales = df_cleaned.groupby('order_purchase_year')['payment_value'].sum().reset_index()

print("Monthly Sales (Top 5):")
print(monthly_sales.head().to_markdown(index=False, numalign="left", stralign="left"))

print("\nYearly Sales:")
print(yearly_sales.to_markdown(index=False, numalign="left", stralign="left"))

# Identify best and worst performing months
best_month = monthly_sales.loc[monthly_sales['payment_value'].idxmax()]
worst_month = monthly_sales.loc[monthly_sales['payment_value'].idxmin()]

print(f"\nBest Performing Month: {best_month['order_purchase_year_month'].strftime('%Y-%m')} with Revenue: ${best_month['payment_value']:.2f}")
print(f"Worst Performing Month: {worst_month['order_purchase_year_month'].strftime('%Y-%m')} with Revenue: ${worst_month['payment_value']:.2f}")

# =============================== Visualize Monthly Sales Trends =============================== #

# Visualize Monthly Sales Trends
plt.figure(figsize=(14, 7))
sns.lineplot(x='order_purchase_year_month', y='payment_value', data=monthly_sales)
plt.title('Monthly Sales Trend (Total Revenue)')
plt.xlabel('Date')
plt.ylabel('Total Revenue ($)')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('monthly_sales_trend.png')
plt.show()
plt.close()

# =============================== Analyze Yearly Sales Performance =============================== #

# Analyze Yearly Sales Performance
plt.figure(figsize=(10, 6))
sns.barplot(x='order_purchase_year', y='payment_value', data=yearly_sales, hue='order_purchase_year', palette='viridis')
plt.title('Yearly Sales Performance (Total Revenue)')
plt.xlabel('Year')
plt.ylabel('Total Revenue ($)')
plt.grid(axis='y')
plt.tight_layout()
plt.savefig('yearly_sales_performance.png')
plt.show()
plt.close()

# =============================== Compare Peak and Low Sales Periods =============================== #

# Compare Peak and Low Sales Periods
# From the above aggregations, we have identified best and worst months/years.
# Let's also check for cancellations
cancelled_orders_monthly = df_cleaned[df_cleaned['order_status'] == 'canceled'].groupby('order_purchase_year_month')['order_id'].nunique().reset_index()
cancelled_orders_monthly['order_purchase_year_month'] = cancelled_orders_monthly['order_purchase_year_month'].dt.to_timestamp()
cancelled_orders_monthly.rename(columns={'order_id': 'number_of_cancellations'}, inplace=True)

print("\nMonthly Cancellations (Top 5):")
print(cancelled_orders_monthly.head().to_markdown(index=False, numalign="left", stralign="left"))

# Merge monthly sales with monthly cancellations for combined analysis
monthly_sales_with_cancellations = pd.merge(monthly_sales, cancelled_orders_monthly, on='order_purchase_year_month', how='left')
monthly_sales_with_cancellations['number_of_cancellations'] = monthly_sales_with_cancellations['number_of_cancellations'].fillna(0) # Fill NaN with 0 for months with no cancellations

print("\nMonthly Sales with Cancellations (Top 5):")
print(monthly_sales_with_cancellations.head().to_markdown(index=False, numalign="left", stralign="left"))

# Visualize Monthly Sales vs Cancellations
fig, ax1 = plt.subplots(figsize=(14, 7))

color = 'tab:blue'
ax1.set_xlabel('Date')
ax1.set_ylabel('Total Revenue ($)', color=color)
sns.lineplot(x='order_purchase_year_month', y='payment_value', data=monthly_sales_with_cancellations, ax=ax1, color=color, label='Total Revenue')
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_title('Monthly Sales and Cancellations')
ax1.grid(True)
ax1.set_xticklabels(monthly_sales_with_cancellations['order_purchase_year_month'].dt.strftime('%Y-%m'), rotation=45)

ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Number of Cancellations', color=color)
sns.lineplot(x='order_purchase_year_month', y='number_of_cancellations', data=monthly_sales_with_cancellations, ax=ax2, color=color, linestyle='--', label='Number of Cancellations')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.savefig('monthly_sales_cancellations_trend.png')
plt.show()
plt.close()

# Find month with highest cancellations
if not cancelled_orders_monthly.empty:
    worst_cancellation_month = cancelled_orders_monthly.loc[cancelled_orders_monthly['number_of_cancellations'].idxmax()]
    print(f"\nMonth with Highest Cancellations: {worst_cancellation_month['order_purchase_year_month'].strftime('%Y-%m')} with {worst_cancellation_month['number_of_cancellations']} cancellations")
else:
    print("\nNo cancellation data available to determine the month with highest cancellations.")


# =============================== Key Findings =============================== #

with open('sales_trends_summary.txt', 'w') as f:
    f.write("Overall Trend: The e-commerce platform experienced rapid growth in 2017, with sales significantly increasing from 2016. Growth continued in 2018, but a decelerating trend was observed in the latter part of the year.\n")
    f.write("Seasonality: There's a clear seasonal peak in sales around November (likely due to holiday shopping or promotional events), especially prominent in 2017.\n")
    f.write("Areas of Improvement:\n")
    f.write("  - Boosting Sales During Low-Performing Months: Strategies for months with historically lower sales (e.g., early 2017, and potentially post-peak periods) could include targeted discounts, flash sales, or marketing campaigns to stimulate demand.\n")
    f.write("  - Addressing the 2018 Downtrend: Further investigation is needed to understand the reasons for the slowdown in 2018, which could involve analyzing external economic factors, competitor activity, or changes in customer behavior.\n")
    f.write("  - Cancellation Management: While cancellations are generally low, understanding the reasons behind cancellations in months like October 2016 could help in refining order fulfillment processes.\n")

