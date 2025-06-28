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


# =============================== Identify Business Opportunities ======================================# 

# =============================== Analyze High-Growth Product Categories ======================================# 

# Convert purchase timestamp to datetime for trend analysis
df_cleaned['order_purchase_timestamp'] = pd.to_datetime(df_cleaned['order_purchase_timestamp'])

# Extract year and month for time-based grouping
df_cleaned['order_year_month'] = df_cleaned['order_purchase_timestamp'].dt.to_period('M')

# Group by product category and month, summing total revenue
category_trend = (
    df_cleaned.groupby(['order_year_month', 'product_category_name_english'])['item_total_revenue']
    .sum()
    .reset_index()
)

# Pivot to see trends more clearly (product categories as columns)
category_trend_pivot = category_trend.pivot(index='order_year_month', columns='product_category_name_english', values='item_total_revenue')

# Fill NaNs with 0 for analysis
category_trend_pivot = category_trend_pivot.fillna(0)

# Calculate the overall growth per category (latest - earliest)
category_growth = category_trend_pivot.iloc[-1] - category_trend_pivot.iloc[0]
category_growth = category_growth.sort_values(ascending=False)

# Show top 10 high-growth categories
print("Top 10 high-growth categories: \n", category_growth.head(10))

# =============================== Identify Regional Sales Growth Areas ======================================# 

# Aggregate total revenue by state and city
state_sales = df_cleaned.groupby('customer_state')['item_total_revenue'].sum().sort_values(ascending=False)
city_sales = df_cleaned.groupby('customer_city')['item_total_revenue'].sum().sort_values(ascending=False)

# Get top 10 states and cities by revenue
top_states = state_sales.head(10)
top_cities = city_sales.head(10)

print("Top States: \n", top_states)
print("Top Cities: \n", top_cities)

# =============================== Understand Customer Behavior for Upselling & Cross-Selling ======================================# 

# Prepare data for RFM segmentation

# Latest date in dataset for recency calculation
latest_date = df_cleaned['order_purchase_timestamp'].max()

# Aggregate RFM metrics by unique customer
rfm = df_cleaned.groupby('customer_unique_id').agg({
    'order_purchase_timestamp': lambda x: (latest_date - x.max()).days,  # Recency
    'order_id': 'nunique',  # Frequency
    'item_total_revenue': 'sum'  # Monetary
}).reset_index()

# Rename columns
rfm.columns = ['customer_id', 'recency', 'frequency', 'monetary']

# Assign RFM segments (simple scoring: 1=low, 4=high)
rfm['r_score'] = pd.qcut(rfm['recency'], 4, labels=[4, 3, 2, 1]).astype(int)
rfm['f_score'] = pd.qcut(rfm['frequency'].rank(method='first'), 4, labels=[1, 2, 3, 4]).astype(int)
rfm['m_score'] = pd.qcut(rfm['monetary'], 4, labels=[1, 2, 3, 4]).astype(int)

# Combine RFM scores
rfm['rfm_segment'] = rfm['r_score'].astype(str) + rfm['f_score'].astype(str) + rfm['m_score'].astype(str)

# Identify top high-value customers (e.g., segment 444)
top_rfm_customers = rfm[rfm['rfm_segment'] == '444']

print("Top High-value Customers: \n", top_rfm_customers.head())

# Product Bundling Analysis

# Group all products in the same order
order_products = df_cleaned.groupby('order_id')['product_id'].apply(list)

# Create pairs
from collections import Counter
from itertools import combinations

pair_counter = Counter()

for products in order_products:
    unique_pairs = set(combinations(sorted(products), 2))
    pair_counter.update(unique_pairs)

# Most common product pairs
common_pairs = pd.DataFrame(pair_counter.most_common(10), columns=['pair', 'count'])

print("Most Common Product Pairs: \n", common_pairs)

# =============================== Optimize Pricing & Promotional Strategies ======================================# 

# Correlation between price and review score
correlation = df_cleaned[['price', 'review_score']].corr()

# Filter orders with review_score ≤ 3 and high price
price_sensitive = df_cleaned[(df_cleaned['review_score'] <= 3) & (df_cleaned['price'] > df_cleaned['price'].median())]

monthly_sales = df_cleaned.groupby('order_purchase_month')['item_total_revenue'].sum()
category_monthly = df_cleaned.groupby(['product_category_name_english', 'order_purchase_month'])['item_total_revenue'].sum().unstack()

# Average price by month
avg_price = df_cleaned.groupby('order_purchase_month')['price'].mean()
monthly_orders = df_cleaned.groupby('order_purchase_month')['order_id'].nunique()

# Find inverse relationship months
promo_months = (avg_price.diff() < 0) & (monthly_orders.diff() > 0)

print("Promotion Month: \n", promo_months)

# ===============================  Improve Customer Retention & Reduce Churn ======================================#

# At-risk customers: no purchase in last 90+ days
at_risk_customers = rfm[rfm['recency'] > 90]
lost_customers = rfm[rfm['recency'] > 180]

# Loyal customers: high frequency and recency
loyal_customers = rfm[(rfm['frequency'] > 3) & (rfm['recency'] < 30)]

# Low-score reviews with comments
negative_feedback = df_cleaned[(df_cleaned['review_score'] <= 3) & (df_cleaned['review_comment_message'].notna())]

print("Low Score Reviews with Comments: \n", negative_feedback)

# ===============================  Explore New Market Segments or Sales Channels ======================================#

# Top cities for high-value categories like 'computers_accessories' or 'health_beauty'
tech_buyers = df_cleaned[df_cleaned['product_category_name_english'] == 'computers_accessories']
tech_buyers_city = tech_buyers.groupby('customer_city')['order_id'].count().sort_values(ascending=False)

# Customers placing high-frequency or high-value orders
potential_b2b = rfm[(rfm['frequency'] > 10) | (rfm['monetary'] > 5000)]

print("Potential B2B Market: \n", potential_b2b)

# ===============================  Document Key Business Opportunities ======================================#

with open('business_opportunities_summary.txt', 'w') as f:
    f.write("Top Business Opportunities Summary\n\n")

    f.write("Expand Inventory in High-Growth Product Categories\n")
    f.write("Insight: Categories like Health & Beauty, Home Decor, and Tech Accessories have shown strong, consistent revenue growth.\n")
    f.write("Recommendation:\n")
    f.write("  - Scale up inventory in top categories.\n")
    f.write("  - Launch new product lines in emerging verticals (e.g., smart home, skincare).\n")
    f.write("Next Steps:\n")
    f.write("  - Collaborate with top-performing sellers.\n")
    f.write("  - Add filters like 'Trending Now' or 'New Launches' to feature them.\n\n")

    f.write("Target High-Sales Regions with Localized Marketing\n")
    f.write("Insight: States like São Paulo, Rio de Janeiro, and cities like Belo Horizonte are revenue hubs.\n")
    f.write("Recommendation:\n")
    f.write("  - Run regional marketing campaigns and festive promotions.\n")
    f.write("  - Ensure fast delivery options and stock availability in these areas.\n")
    f.write("Next Steps:\n")
    f.write("  - Invest in local warehouses or courier partners.\n")
    f.write("  - Launch city-based discount codes (e.g., 'SP10' for São Paulo).\n\n")

    f.write("Upsell to High-Value Customers Using RFM Segmentation\n")
    f.write("Insight: Segment 444 customers are highly loyal and high spenders.\n")
    f.write("Recommendation:\n")
    f.write("  - Promote premium bundles or membership perks (free shipping, early access).\n")
    f.write("  - Target RFM 441 and 144 segments with upsell or retention nudges.\n")
    f.write("Next Steps:\n")
    f.write("  - Launch a personalized product recommender.\n")
    f.write("  - A/B test bundled offers with frequent pairings (e.g., laptop + headphones).\n\n")

    f.write("Optimize Pricing & Promotions Based on Demand Cycles\n")
    f.write("Insight: Certain months show sales surges—likely due to seasonal trends or past promotions.\n")
    f.write("Recommendation:\n")
    f.write("  - Use dynamic pricing: raise prices during demand spikes, offer off-season deals.\n")
    f.write("  - Prioritize flash sales or email campaigns in proven successful months.\n")
    f.write("Next Steps:\n")
    f.write("  - Automate pricing adjustments using trend rules.\n")
    f.write("  - Build a promo calendar for each major category.\n\n")

    f.write("Retain At-Risk Customers with Loyalty & Win-Back Programs\n")
    f.write("Insight: Customers with high recency (>90 days) and low frequency/monetary scores are at churn risk.\n")
    f.write("Recommendation:\n")
    f.write("  - Design win-back emails and one-time reactivation discounts.\n")
    f.write("  - Reward loyal users with a points-based loyalty system.\n")
    f.write("Next Steps:\n")
    f.write("  - Monitor cohort retention trends monthly.\n")
    f.write("  - Integrate feedback from low-score reviews to improve product/service.\n")
