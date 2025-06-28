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

# =============================== Group Sales Data by Product and Category =============================== #

# Group by product_id
product_sales = df_cleaned.groupby('product_id').agg(
    total_quantity_sold=('order_item_id', 'count'), # Count items sold
    total_revenue=('item_total_revenue', 'sum') # Sum of item_total_revenue
).reset_index()

# Group by product_category_name_english
category_sales = df_cleaned.groupby('product_category_name_english').agg(
    total_quantity_sold=('order_item_id', 'count'),
    total_revenue=('item_total_revenue', 'sum')
).reset_index()

print("Product Sales (Top 5 by Revenue):")
print(product_sales.sort_values(by='total_revenue', ascending=False).head().to_markdown(index=False, numalign="left", stralign="left"))

print("\nCategory Sales (Top 5 by Revenue):")
print(category_sales.sort_values(by='total_revenue', ascending=False).head().to_markdown(index=False, numalign="left", stralign="left"))

# =============================== Find the Best-Selling Products =============================== #

top_10_products_revenue = product_sales.sort_values(by='total_revenue', ascending=False).head(10)
print("\nTop 10 Best-Selling Products by Revenue:")
print(top_10_products_revenue.to_markdown(index=False, numalign="left", stralign="left"))

# Check if certain products sell better in specific seasons (e.g., winter wear during December).
# To do this, we need to join back with product_id and month information.
product_monthly_sales = df_cleaned.groupby(['product_id', 'order_purchase_month']).agg(
    monthly_revenue=('item_total_revenue', 'sum')
).reset_index()

# Find the month with highest sales for each of the top 10 products
top_products_seasonal = pd.merge(
    top_10_products_revenue[['product_id']],
    product_monthly_sales,
    on='product_id',
    how='inner'
)
top_products_seasonal_peak_month = top_products_seasonal.loc[top_products_seasonal.groupby('product_id')['monthly_revenue'].idxmax()]

print("\nPeak Sales Month for Top Products:")
print(top_products_seasonal_peak_month.to_markdown(index=False, numalign="left", stralign="left"))

# =============================== Analyze Best-Selling Categories =============================== #

top_categories_revenue = category_sales.sort_values(by='total_revenue', ascending=False).head(10)
print("\nTop 10 Best-Selling Categories by Revenue:")
print(top_categories_revenue.to_markdown(index=False, numalign="left", stralign="left"))

# Identify which category contributes the most revenue
most_revenue_category = top_categories_revenue.iloc[0]
print(f"\nCategory contributing the most revenue: {most_revenue_category['product_category_name_english']} with Revenue: ${most_revenue_category['total_revenue']:.2f}")

# =============================== Visualize the Results =============================== #

# Bar chart for Top 10 Best-Selling Products by Revenue
plt.figure(figsize=(12, 7))
sns.barplot(x='total_revenue', y='product_id', data=top_10_products_revenue, hue = 'total_revenue', palette='coolwarm')
plt.title('Top 10 Best-Selling Products by Revenue')
plt.xlabel('Total Revenue ($)')
plt.ylabel('Product ID')
plt.tight_layout()
plt.savefig('top_10_products_revenue.png')
plt.show()
plt.close()

# Bar chart for Top 10 Best-Selling Categories by Revenue
plt.figure(figsize=(12, 7))
sns.barplot(x='total_revenue', y='product_category_name_english', data=top_categories_revenue, hue = 'total_revenue', palette='viridis')
plt.title('Top 10 Best-Selling Categories by Revenue')
plt.xlabel('Total Revenue ($)')
plt.ylabel('Product Category')
plt.tight_layout()
plt.savefig('top_10_categories_revenue.png')
plt.show()
plt.close()

# Pie chart for Sales Distribution Across Top 5 Categories
top_5_categories = category_sales.sort_values(by='total_revenue', ascending=False).head(5)
plt.figure(figsize=(10, 10))
plt.pie(top_5_categories['total_revenue'], labels=top_5_categories['product_category_name_english'], autopct='%1.1f%%', startangle=90, pctdistance=0.85) # Removed cmap
plt.title('Sales Distribution Across Top 5 Categories')
plt.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
plt.tight_layout()
plt.savefig('top_5_categories_sales_distribution.png')
plt.show()
plt.close()

# Check if certain categories perform better during specific months.
category_monthly_sales = df_cleaned.groupby(['product_category_name_english', 'order_purchase_month']).agg(
    monthly_revenue=('item_total_revenue', 'sum')
).reset_index()

# Focus on the top 5 categories for monthly trend visualization
top_5_category_names = top_5_categories['product_category_name_english'].tolist()
filtered_category_monthly_sales = category_monthly_sales[category_monthly_sales['product_category_name_english'].isin(top_5_category_names)]

plt.figure(figsize=(14, 8))
sns.lineplot(x='order_purchase_month', y='monthly_revenue', hue='product_category_name_english', data=filtered_category_monthly_sales, marker='o')
plt.title('Monthly Revenue Trends for Top 5 Categories')
plt.xlabel('Month')
plt.ylabel('Monthly Revenue ($)')
plt.xticks(ticks=range(1, 13), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.grid(True)
plt.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('monthly_revenue_top_categories.png')
plt.show()
plt.close()

# =============================== Key Findings =============================== #

with open('product_analysis_summary.txt', 'w') as f:
    f.write("Top 5 Products by Revenue:\n")
    f.write("422879e10f46567339264dad8f8993ec\n")
    f.write("99a4788f14d9465d46c76a17d692d689\n")
    f.write("389d119b48cf79933d3e09829940fa6c\n")
    f.write("3dd2a17168ec895c781a9191c1e95ad7\n")
    f.write("53759a2ec86d6ff9dbf326530cddcd61\n\n")

    f.write("Top 3 Categories by Revenue:\n")
    f.write("health_beauty\n")
    f.write("watches_gifts\n")
    f.write("bed_bath_table\n\n")

    f.write("Seasonal Trends in Product Demand:\n")
    f.write("Many of the top products show a peak in sales during August, while the overall sales trend for the platform typically peaks in November. This suggests that while November is a high-volume month across the board, individual product popularity might fluctuate. Categories like health_beauty and watches_gifts also show monthly fluctuations, indicating potential seasonal demand for these categories.\n\n")

    f.write("Suggested Inventory Management Strategies for Top-Performing Items:\n")
    f.write("  - Proactive Stocking: For products and categories that consistently perform well and show clear seasonal peaks (e.g., August for many top products, November for overall platform sales), implement proactive inventory stocking strategies to ensure sufficient supply to meet demand during peak seasons.\n")
    f.write("  - Demand Forecasting: Utilize historical sales data for these top products and categories to refine demand forecasting models, especially considering the observed monthly trends, to optimize inventory levels and minimize stockouts or overstocking.\n")
    f.write("  - Supplier Relationship Management: Strengthen relationships with suppliers of top-selling products and categories to ensure reliable and timely replenishment, especially during high-demand periods.\n")
    f.write("  - Diversification for Low-Sales Months: For categories or products that experience significant dips in sales during certain months, consider diversifying the product offering or running targeted promotions to stimulate sales during these slower periods.\n")
