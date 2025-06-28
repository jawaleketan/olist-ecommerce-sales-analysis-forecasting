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

# =============================== Define Customer Segments Based on RFM Scores =============================== #

df_cleaned['order_purchase_timestamp'] = pd.to_datetime(df_cleaned['order_purchase_timestamp'])
reference_date = df_cleaned['order_purchase_timestamp'].max() + timedelta(days=1)

rfm_df = df_cleaned.groupby('customer_unique_id').agg(
    last_purchase_date=('order_purchase_timestamp', 'max'),
    frequency=('order_id', 'nunique'),
    monetary_value=('payment_value', 'sum')
).reset_index()

rfm_df['recency'] = (reference_date - rfm_df['last_purchase_date']).dt.days

rfm_df['R_score'] = pd.qcut(rfm_df['recency'], 5, labels=[5, 4, 3, 2, 1])
rfm_df['F_score'] = pd.qcut(rfm_df['frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
rfm_df['M_score'] = pd.qcut(rfm_df['monetary_value'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])

rfm_df['RFM_score'] = rfm_df['R_score'].astype(str) + rfm_df['F_score'].astype(str) + rfm_df['M_score'].astype(str)

def rfm_segment(df):
    if df['R_score'] == 5 and df['F_score'] == 5 and df['M_score'] == 5:
        return 'Champions'
    elif df['R_score'] == 5 and df['F_score'] == 5:
        return 'Loyal Customers'
    elif df['R_score'] == 5 and df['M_score'] == 5:
        return 'Big Spenders'
    elif df['R_score'] == 5 and df['F_score'] >= 3:
        return 'New Customers'
    elif df['R_score'] >= 4 and df['F_score'] >= 4:
        return 'Promising'
    elif df['R_score'] <= 2 and df['F_score'] >= 4:
        return 'At Risk'
    elif df['R_score'] <= 2 and df['F_score'] <= 2:
        return 'Lost Customers'
    elif df['R_score'] >= 4 and df['M_score'] >= 4:
        return 'Potential Loyalists'
    else:
        return 'Others'

rfm_df['RFM_segment'] = rfm_df.apply(rfm_segment, axis=1)

# =============================== Analyze Customer Behavior =============================== #

# Calculate average RFM values for each segment
segment_behavior = rfm_df.groupby('RFM_segment').agg(
    Avg_Recency=('recency', 'mean'),
    Avg_Frequency=('frequency', 'mean'),
    Avg_Monetary_Value=('monetary_value', 'mean'),
    Num_Customers=('customer_unique_id', 'count')
).reset_index()

print("\nAverage RFM Values and Customer Count per Segment:")
print(segment_behavior.sort_values(by='Num_Customers', ascending=False).to_markdown(index=False, numalign="left", stralign="left"))

# Join RFM segments back to df_cleaned to analyze geographic patterns for high-value customers
df_with_rfm_segment = pd.merge(df_cleaned, rfm_df[['customer_unique_id', 'RFM_segment']], on='customer_unique_id', how='left')

# Analyze geographical distribution of Champions/Loyal/Big Spenders (High-Value Customers)
high_value_segments = ['Champions', 'Loyal Customers', 'Big Spenders']
high_value_customers_geo = df_with_rfm_segment[df_with_rfm_segment['RFM_segment'].isin(high_value_segments)]

# Group by state for high-value customers
high_value_state_distribution = high_value_customers_geo.groupby('customer_state')['customer_unique_id'].nunique().reset_index()
high_value_state_distribution.rename(columns={'customer_unique_id': 'Unique_High_Value_Customers'}, inplace=True)

print("\nDistribution of High-Value Customers by State (Top 10):")
print(high_value_state_distribution.sort_values(by='Unique_High_Value_Customers', ascending=False).head(10).to_markdown(index=False, numalign="left", stralign="left"))

# =============================== Visualize the Segments =============================== #

# Bar chart for Number of Customers in Each RFM Segment (already done in previous step, but re-run for completeness)
rfm_segment_counts = rfm_df['RFM_segment'].value_counts().reset_index()
rfm_segment_counts.columns = ['RFM_segment', 'Number_of_Customers']

plt.figure(figsize=(12, 7))
sns.barplot(x='Number_of_Customers', y='RFM_segment', data=rfm_segment_counts.sort_values(by='Number_of_Customers', ascending=False), palette='magma')
plt.title('Distribution of Customers Across RFM Segments')
plt.xlabel('Number of Customers')
plt.ylabel('RFM Segment')
plt.tight_layout()
plt.savefig('rfm_segment_distribution.png')
plt.show()
plt.close()

# Heatmap of average RFM scores per segment
# Create a pivot table for the heatmap
heatmap_data = segment_behavior.set_index('RFM_segment')[['Avg_Recency', 'Avg_Frequency', 'Avg_Monetary_Value']]
plt.figure(figsize=(10, 6))
sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="YlGnBu", linewidths=.5)
plt.title('Average RFM Metrics by Customer Segment')
plt.ylabel('RFM Segment')
plt.tight_layout()
plt.savefig('rfm_segment_heatmap.png')
plt.show()
plt.close()

# Scatter plot of Recency vs. Monetary for 'Champions' vs 'Lost Customers'
plt.figure(figsize=(10, 8))
sns.scatterplot(x='recency', y='monetary_value', hue='RFM_segment',
                data=rfm_df[rfm_df['RFM_segment'].isin(['Champions', 'Lost Customers'])],
                alpha=0.6, s=50)
plt.title('Recency vs. Monetary Value for Champions and Lost Customers')
plt.xlabel('Recency (Days since last purchase)')
plt.ylabel('Monetary Value ($)')
plt.legend(title='RFM Segment')
plt.grid(True)
plt.tight_layout()
plt.savefig('rfm_recency_monetary_scatter.png')
plt.show()
plt.close()

# =============================== Provide Marketing and Engagement Recommendations =============================== #

with open('marketing_engagement_recommendations.txt', 'w') as f:
    f.write("Provide Marketing and Engagement Recommendations\n")
    f.write("Document Key Insights\n\n")

    f.write("Key Insights:\n")
    f.write("The business has a solid base of 'Champions' and 'Loyal Customers' who are crucial for sustained revenue.\n")
    f.write("A large segment of 'At Risk' and 'Lost Customers' presents a significant opportunity for recovery through targeted re-engagement.\n")
    f.write("'New Customers' are a vital segment for future growth, requiring careful nurturing to transition them into loyal buyers.\n")
    f.write("Geographic insights show concentrated high-value customer bases in specific states, allowing for regionally tailored marketing efforts.\n\n")

    f.write("Targeted Marketing Strategy:\n")
    f.write("Focus on retention for high-value segments through exclusive benefits.\n")
    f.write("Prioritize re-engagement for 'At Risk' and 'Lost Customers' with personalized incentives.\n")
    f.write("Implement strong nurturing programs for 'New Customers' to foster loyalty and increase their lifetime value.\n")
    f.write("Leverage geographic data to tailor marketing campaigns to regions with a high concentration of valuable customer segments.\n\n")

    f.write("Next Steps to Improve Customer Engagement and Retention:\n")
    f.write("  - A/B Test Marketing Campaigns: Continuously test different offers, messaging, and channels for each RFM segment to optimize engagement.\n")
    f.write("  - Personalization at Scale: Invest in tools and strategies for hyper-personalization, especially for 'Champions' and 'Loyal Customers'.\n")
    f.write("  - Customer Feedback Loop: Implement mechanisms to consistently collect and act on customer feedback, particularly from 'At Risk' and 'Lost Customers', to address issues and improve the overall customer experience.\n")
    f.write("  - Predictive Analytics: Explore using predictive models to identify customers at risk of churning before they become 'At Risk', allowing for proactive intervention.\n")
    f.write("  - Segment-Specific Product Development: Use insights from 'Champions' and 'Big Spenders' to inform future product development and inventory management.\n")
    f.write("  - Referral Programs: Encourage 'Champions' and 'Loyal Customers' to refer new customers, leveraging their positive experience.\n")