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

# =============================== Choose a Reference Date =============================== #

df_cleaned['order_purchase_timestamp'] = pd.to_datetime(df_cleaned['order_purchase_timestamp'])
reference_date = df_cleaned['order_purchase_timestamp'].max() + timedelta(days=1)
print(reference_date)

# =============================== Group Data by Customer ID =============================== #

rfm_df = df_cleaned.groupby('customer_unique_id').agg(
    recency=('order_purchase_timestamp', lambda date: (reference_date - date.max()).days),
    frequency=('order_id', 'nunique'), # Number of unique orders
    monetary=('payment_value', 'sum') # Total amount spent
).reset_index()

print("RFM Data (Top 5):")
print(rfm_df.head().to_markdown(index=False, numalign="left", stralign="left"))

# =============================== Normalize and Rank RFM Scores =============================== #

# Normalize and Rank RFM Scores (1-5)
# Recency: Lower days (more recent) is better, so assign higher score to lower recency values.
rfm_df['R_Score'] = pd.qcut(rfm_df['recency'], q=5, labels=[5, 4, 3, 2, 1])

# Frequency: Higher frequency is better.
bins = pd.qcut(rfm_df['frequency'], q=5, duplicates='drop')
rfm_df['F_Score'] = bins
rfm_df['F_Score'] = rfm_df['F_Score'].cat.rename_categories([1, 2, 3, 4, 5][:len(bins.cat.categories)])

# Monetary: Higher monetary value is better.
bins = pd.qcut(rfm_df['monetary'], q=5, duplicates='drop')
rfm_df['M_Score'] = bins
rfm_df['M_Score'] = rfm_df['M_Score'].cat.rename_categories([1, 2, 3, 4, 5][:len(bins.cat.categories)])

# Convert scores to integer type for concatenation
rfm_df['R_Score'] = rfm_df['R_Score'].astype(int)
rfm_df['F_Score'] = rfm_df['F_Score'].astype(int)
rfm_df['M_Score'] = rfm_df['M_Score'].astype(int)

# Create RFM Segment
rfm_df['RFM_Segment'] = rfm_df['R_Score'].astype(str) + rfm_df['F_Score'].astype(str) + rfm_df['M_Score'].astype(str)

print("\nRFM Data with Scores (Top 5):")
print(rfm_df.head().to_markdown(index=False, numalign="left", stralign="left"))

# =============================== Interpret the RFM Segments =============================== #

# 5. Interpret the RFM Segments
# Example segmentation logic (can be adjusted based on business needs)
def rfm_segmentation(df):
    if df['R_Score'] == 5 and df['F_Score'] == 5 and df['M_Score'] == 5:
        return 'Champions' # Best customers, most recent, frequent, and high spenders
    elif df['R_Score'] == 5 and df['F_Score'] == 5:
        return 'Loyal Customers' # Recent, frequent, but not necessarily high spenders
    elif df['R_Score'] == 5 and df['M_Score'] == 5:
        return 'New High-Value' # Recent, high spenders, might become frequent
    elif df['R_Score'] == 5 and df['F_Score'] >= 3:
        return 'Promising' # Recent, somewhat frequent, potential to be loyal
    elif df['R_Score'] >= 4 and df['F_Score'] >= 4:
        return 'Potential Loyalists' # Good Recency & Frequency
    elif df['R_Score'] <= 2 and df['F_Score'] >= 4 and df['M_Score'] >= 4:
        return 'At-Risk' # High F & M, but not recent
    elif df['R_Score'] <= 2 and df['F_Score'] <= 2 and df['M_Score'] <= 2:
        return 'Lost Customers' # Low R, F, M
    elif df['R_Score'] <= 2:
        return 'Hibernating' # Low Recency, but might have bought before
    else:
        return 'Others' # Catch-all for less clear segments

rfm_df['Customer_Segment'] = rfm_df.apply(rfm_segmentation, axis=1)

print("\nRFM Data with Customer Segments (Top 5):")
print(rfm_df.head().to_markdown(index=False, numalign="left", stralign="left"))

# Summarize the number of customers in each RFM segment
segment_counts = rfm_df['Customer_Segment'].value_counts().reset_index()
segment_counts.columns = ['Customer_Segment', 'Number_of_Customers']
print("\nNumber of Customers in Each RFM Segment:")
print(segment_counts.to_markdown(index=False, numalign="left", stralign="left"))

# Save RFM data to CSV for potential future use
rfm_df.to_csv('rfm_metrics.csv', index=False)

# Visualizing RFM segments
plt.figure(figsize=(12, 7))
sns.barplot(x='Number_of_Customers', y='Customer_Segment', data=segment_counts.sort_values(by='Number_of_Customers', ascending=False), palette='coolwarm')
plt.title('Number of Customers in Each RFM Segment')
plt.xlabel('Number of Customers')
plt.ylabel('Customer Segment')
plt.tight_layout()
plt.savefig('rfm_segment_counts.png')
plt.show()
plt.close()

# =============================== Key Insights =============================== #

with open('customer_segmentation_summary.txt', 'w') as f:
    f.write("Key Insights:\n")
    f.write("The largest segment is 'Others', indicating a diverse customer base that doesn't fit neatly into specific high-value or at-risk categories.\n")
    f.write("A significant portion of the customer base falls into 'Lost Customers' and 'At Risk' segments, highlighting a challenge in customer retention.\n")
    f.write("The 'Champions' and 'Loyal Customers' segments, though smaller in number, represent the most valuable customer groups.\n\n")

    f.write("Customer Groups Needing Re-engagement Strategies:\n")
    f.write("  - At Risk: These customers were once active but haven't purchased recently. They need immediate re-engagement to prevent them from becoming 'Lost Customers'.\n")
    f.write("  - Lost Customers: These customers have not purchased in a long time. While challenging, some may be reactivated with compelling offers.\n\n")

    f.write("Suggested Marketing Strategies:\n")
    f.write("  - High-Value Customers (Champions, Loyal Customers, Big Spenders):\n")
    f.write("    - Champions (R5, F5, M5): Reward programs, VIP exclusive access, early access to new products, personalized thank-you messages. Goal: Retain and maximize their lifetime value.\n")
    f.write("    - Loyal Customers (R5, F5): Loyalty bonuses, personalized product recommendations, special discounts on their preferred categories. Goal: Encourage continued frequent purchases.\n")
    f.write("    - Big Spenders (R5, M5): High-value product recommendations, premium service, personalized offers on higher-priced items. Goal: Reinforce their spending habits and encourage further high-value purchases.\n")
    f.write("  - At-Risk Customers (R<=2, F>=4):\n")
    f.write("    - Re-engagement Campaigns: Send personalized emails or notifications with special discounts on previously viewed or purchased items.\n")
    f.write("    - Feedback Surveys: Understand reasons for reduced activity to address pain points.\n")
    f.write("    - Limited-Time Offers: Create urgency with expiring discounts or free shipping.\n")
    f.write("  - Lost Customers (R<=2, F<=2):\n")
    f.write("    - Win-Back Campaigns: Offer significant incentives to encourage a return purchase (e.g., a large discount on their first order back).\n")
    f.write("    - New Product Announcements: Inform them about significant new arrivals that might pique their interest.\n")
    f.write("    - Re-subscription Prompts: For those who have disengaged, a simple prompt to re-engage with the brand.\n")
    f.write("  - New Customers (R5, F>=3):\n")
    f.write("    - Onboarding Programs: Welcome series, guides on product usage, encourage reviews, and offer incentives for a second purchase.\n")
    f.write("    - Educate on Loyalty Program: Introduce them to the benefits of becoming a loyal customer.\n")
