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

rfm_segment_counts = rfm_df['RFM_segment'].value_counts().reset_index()
rfm_segment_counts.columns = ['RFM_segment', 'Number_of_Customers']

# =============================== Visualize Recency vs. Frequency vs. Monetary (RFM) Scores =============================== #

# Visualize Recency vs. Frequency vs. Monetary (RFM) Scores
# Scatter plot: Recency vs. Frequency colored by Monetary Score
plt.figure(figsize=(12, 8))
sns.scatterplot(x='recency', y='frequency', hue='M_score', data=rfm_df, palette='viridis', s=100, alpha=0.6)
plt.title('RFM: Recency vs. Frequency (Colored by Monetary Score)')
plt.xlabel('Recency (Days since last purchase)')
plt.ylabel('Frequency (Number of Purchases)')
plt.legend(title='Monetary Score', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.savefig('rfm_recency_frequency_monetary_scatter.png')
plt.show()
plt.close()

# Scatter plot: Frequency vs. Monetary Value colored by Recency Score
plt.figure(figsize=(12, 8))
sns.scatterplot(x='frequency', y='monetary_value', hue='R_score', data=rfm_df, palette='plasma', s=100, alpha=0.6)
plt.title('RFM: Frequency vs. Monetary Value (Colored by Recency Score)')
plt.xlabel('Frequency (Number of Purchases)')
plt.ylabel('Monetary Value ($)')
plt.legend(title='Recency Score', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.savefig('rfm_frequency_monetary_recency_scatter.png')
plt.show()
plt.close()

# =============================== Heatmap for RFM Distribution =============================== #

# Heatmap for RFM Distribution
# Heatmap of customer counts by R_score and F_score
rf_counts = rfm_df.groupby(['R_score', 'F_score']).size().unstack(fill_value=0)
rf_counts = rf_counts.sort_index(ascending=False) # Sort R_score for better visualization (5 at top)

plt.figure(figsize=(10, 8))
sns.heatmap(rf_counts, annot=True, fmt='d', cmap='YlGnBu', linewidths=.5, linecolor='black')
plt.title('Customer Count by Recency and Frequency Scores')
plt.xlabel('Frequency Score')
plt.ylabel('Recency Score')
plt.tight_layout()
plt.savefig('rf_score_heatmap.png')
plt.show()
plt.close()

# Heatmap of customer counts by F_score and M_score
fm_counts = rfm_df.groupby(['F_score', 'M_score']).size().unstack(fill_value=0)

plt.figure(figsize=(10, 8))
sns.heatmap(fm_counts, annot=True, fmt='d', cmap='YlGnBu', linewidths=.5, linecolor='black')
plt.title('Customer Count by Frequency and Monetary Scores')
plt.xlabel('Monetary Score')
plt.ylabel('Frequency Score')
plt.tight_layout()
plt.savefig('fm_score_heatmap.png')
plt.show()
plt.close()

# =============================== Pie Charts for Segment Proportions =============================== #

# Pie Charts for Segment Proportions
plt.figure(figsize=(10, 10))
plt.pie(rfm_segment_counts['Number_of_Customers'], labels=rfm_segment_counts['RFM_segment'],
        autopct='%1.1f%%', startangle=90, pctdistance=0.85)
plt.title('Proportion of Customers in Each RFM Segment')
plt.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
plt.tight_layout()
plt.savefig('rfm_segment_proportion_pie_chart.png')
plt.show()

# =============================== Document and Interpret the Visualizations =============================== #

with open('visualization_interpretation.txt', 'w') as f:
    f.write("Document and Interpret the Visualizations\n\n")

    f.write("High-Value Customers (Champions, Loyal Customers, Big Spenders):\n")
    f.write("  - Insights: Visually located at the 'sweet spot' of RFM plots (low recency, high frequency/monetary value). They are a smaller but crucial portion of the customer base.\n")
    f.write("  - Recommendations: Focus on retention through exclusive loyalty programs, personalized premium offers, and proactive engagement to maintain their high value. Their high R, F, and M scores on the heatmap and scatter plots indicate their importance.\n\n")

    f.write("New Customers:\n")
    f.write("  - Insights: Generally recent, but with low frequency and monetary value.\n")
    f.write("  - Recommendations: Implement robust onboarding programs and nurture campaigns (e.g., welcome series, second purchase incentives) to encourage repeat purchases and increase their lifetime value. The scatter plots help identify them as recent but still early in their journey.\n\n")

    f.write("Potential Loyalists:\n")
    f.write("  - Insights: Good recency, frequency, and monetary value, but not yet top tier.\n")
    f.write("  - Recommendations: Encourage deeper engagement through personalized recommendations, loyalty program education, and exclusive offers to transition them into 'Loyal Customers' or 'Champions'.\n\n")

    f.write("At-Risk Customers:\n")
    f.write("  - Insights: High recency (haven't purchased recently) but historically good frequency/monetary value. Clearly visible on the right side of the Recency vs. Frequency scatter plot.\n")
    f.write("  - Recommendations: Immediate re-engagement campaigns with targeted discounts, personalized product recommendations, and feedback surveys to prevent churn. The heatmap will show these as higher R, and lower F density.\n\n")

    f.write("Lost Customers:\n")
    f.write("  - Insights: High recency and low frequency/monetary values. Located in the top-right of the Recency vs. Frequency plot. Largest segment with high recency.\n")
    f.write("  - Recommendations: Implement win-back campaigns with significant incentives, or consider re-activating them with new product launches. While challenging, some may respond to strong offers. The heatmap will show a high density in the higher R-score and lower F-score cells.\n\n")

    f.write("Others:\n")
    f.write("  - Insights: A diverse group not fitting neatly into other categories.\n")
    f.write("  - Recommendations: Further analysis might be needed to identify sub-segments within this group. Generic marketing might be less effective; consider broad awareness campaigns or identifying common behaviors for more targeted approaches.\n")
