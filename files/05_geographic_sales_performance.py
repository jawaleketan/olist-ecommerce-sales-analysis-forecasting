# =============================== Importing libraries =============================== #

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Load the cleaned dataset
df_cleaned = pd.read_csv('olist_merged_data_cleaned.csv')

# ===============================  Identify Geographic Data in the Dataset =============================== #

print("Geographic Data in the Dataset: customer_city, customer_state are available.")

# ===============================  Group Sales Data by Location =============================== #

# Group by customer_state
state_sales = df_cleaned.groupby('customer_state').agg(
    total_revenue=('payment_value', 'sum'),
    total_orders=('order_id', 'nunique') # Count unique order IDs for total orders
).reset_index()

# Group by customer_city
city_sales = df_cleaned.groupby('customer_city').agg(
    total_revenue=('payment_value', 'sum'),
    total_orders=('order_id', 'nunique')
).reset_index()

# Rank locations
state_sales_ranked = state_sales.sort_values(by='total_revenue', ascending=False)
city_sales_ranked = city_sales.sort_values(by='total_revenue', ascending=False)

print("Top 5 High-Performing States by Revenue:")
print(state_sales_ranked.head(5).to_markdown(index=False, numalign="left", stralign="left"))

print("\nLow-Performing States by Revenue (Bottom 5):")
print(state_sales_ranked.tail(5).to_markdown(index=False, numalign="left", stralign="left"))

print("\nTop 5 High-Performing Cities by Revenue:")
print(city_sales_ranked.head(5).to_markdown(index=False, numalign="left", stralign="left"))

print("\nLow-Performing Cities by Revenue (Bottom 5):")
print(city_sales_ranked.tail(5).to_markdown(index=False, numalign="left", stralign="left"))

# ===============================  Identify Trends Across Regions =============================== #

# Customer Retention/Repeat Purchases by State:
# Calculate unique customers per state and repeat purchase rate (if possible)
customer_state_unique = df_cleaned.groupby('customer_state')['customer_unique_id'].nunique().reset_index()
customer_state_unique.rename(columns={'customer_unique_id': 'unique_customers'}, inplace=True)

# To check repeat purchases, we need to identify customers with more than one order
repeat_customers = df_cleaned.groupby('customer_unique_id')['order_id'].nunique()
repeat_customers = repeat_customers[repeat_customers > 1].reset_index()
repeat_customer_ids = repeat_customers['customer_unique_id'].tolist()

repeat_customer_data = df_cleaned[df_cleaned['customer_unique_id'].isin(repeat_customer_ids)]
repeat_purchase_by_state = repeat_customer_data.groupby('customer_state')['customer_unique_id'].nunique().reset_index()
repeat_purchase_by_state.rename(columns={'customer_unique_id': 'repeat_customers'}, inplace=True)

state_customer_summary = pd.merge(customer_state_unique, repeat_purchase_by_state, on='customer_state', how='left')
state_customer_summary['repeat_customers'] = state_customer_summary['repeat_customers'].fillna(0)
state_customer_summary['repeat_customer_rate'] = (state_customer_summary['repeat_customers'] / state_customer_summary['unique_customers']) * 100
state_customer_summary.sort_values(by='repeat_customer_rate', ascending=False, inplace=True)

print("\nStates by Repeat Customer Rate (Top 5):")
print(state_customer_summary.head(5).to_markdown(index=False, numalign="left", stralign="left"))

# Identify states where certain product categories perform better
# Get top 3 categories from previous task
top_3_categories_names = ['health_beauty', 'watches_gifts', 'bed_bath_table']

category_state_sales = df_cleaned[df_cleaned['product_category_name_english'].isin(top_3_categories_names)].groupby(['customer_state', 'product_category_name_english']).agg(
    category_revenue=('item_total_revenue', 'sum')
).reset_index()

# Find the top category for each state, among the top 3 overall categories
idx = category_state_sales.groupby('customer_state')['category_revenue'].idxmax()
top_category_per_state = category_state_sales.loc[idx].sort_values(by='category_revenue', ascending=False)

print("\nTop Category (among overall top 3) by Revenue for Each State (Top 5 States):")
print(top_category_per_state.head(5).to_markdown(index=False, numalign="left", stralign="left"))

# ===============================  Visualize the Data on a Heatmap =============================== #

# Bar chart for Top 10 States by Revenue
plt.figure(figsize=(12, 7))
sns.barplot(x='total_revenue', y='customer_state', data=state_sales_ranked.head(10), palette='coolwarm')
plt.title('Top 10 States by Total Sales Revenue')
plt.xlabel('Total Revenue ($)')
plt.ylabel('State')
plt.tight_layout()
plt.savefig('top_10_states_revenue.png')
plt.show()
plt.close()

# Bar chart for Top 10 Cities by Revenue
plt.figure(figsize=(12, 7))
sns.barplot(x='total_revenue', y='customer_city', data=city_sales_ranked.head(10), palette='viridis')
plt.title('Top 10 Cities by Total Sales Revenue')
plt.xlabel('Total Revenue ($)')
plt.ylabel('City')
plt.tight_layout()
plt.savefig('top_10_cities_revenue.png')
plt.show()
plt.close()

# ===============================  Key Insights & Business Recommendations =============================== #

with open('regional_analysis_summary.txt', 'w') as f:
    f.write("Top-Performing Locations:\n")
    f.write("The state of Sao Paulo (SP) and its capital city Sao Paulo are the undisputed leaders in sales revenue and order volume, followed by Rio de Janeiro (RJ) and Minas Gerais (MG) and their respective major cities. These urban centers represent the core customer base and sales drivers for the business.\n\n")

    f.write("Underperforming Locations:\n")
    f.write("A large number of cities have very low sales, often with only a single order. These represent areas with minimal market penetration.\n\n")

    f.write("Logistics Improvements based on Regional Demand:\n")
    f.write("  - Optimize for SP/RJ/MG: Given the high concentration of sales in SP, RJ, and MG, logistics infrastructure (warehouses, delivery hubs, last-mile delivery services) should be heavily optimized in these regions to ensure fast, efficient, and cost-effective deliveries.\n")
    f.write("  - Tiered Delivery Service: Consider implementing tiered delivery services, offering faster (potentially premium) options in high-density, high-revenue areas, and more economical (standard) options for lower-volume regions.\n")
    f.write("  - Route Optimization: Invest in advanced route optimization technologies for deliveries within the top-performing cities to maximize efficiency and reduce delivery times.\n\n")

    f.write("Targeted Marketing Campaigns for Specific Regions to Boost Sales:\n")
    f.write("  - Maintain Dominance in Top Regions: Continue strong marketing efforts in SP, RJ, and MG to retain market share and potentially cross-sell/upsell to the existing large customer base. Leverage data on popular categories within these states to run targeted campaigns.\n")
    f.write("  - Strategic Expansion to Mid-Tier States/Cities: Focus marketing and logistical efforts on states like RS, PR, and SC, and their respective larger cities, which show moderate sales. These could be regions with high growth potential with focused investment.\n")
    f.write("  - Awareness Campaigns in Low-Penetration Areas: For the numerous low-performing cities, consider highly localized, cost-effective digital marketing campaigns to build brand awareness and drive initial purchases. Partnerships with local influencers or community groups could be effective.\n")
    f.write("  - Category-Specific Regional Marketing: While not overwhelmingly distinct in this analysis, if further data reveals specific category preferences in certain states (e.g., more sports leisure in a coastal state), tailor marketing content and promotions accordingly.\n")
