# =============================== Importing libraries =============================== #

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# =============================== Loading dataset =============================== #

df = pd.read_csv("olist_merged_data.csv")

# =============================== Handle Missing Values =============================== #

print("--- Before handling missing values ---")
print(df.isnull().sum().to_markdown(numalign="left", stralign="left"))
print(f"\nShape after handling missing values: {df.shape}")

# Create a copy to work on
df_cleaned = df.copy()

# Drop rows where payment information is missing (critical for revenue calculation)
payment_cols = ['payment_sequential', 'payment_type', 'payment_installments', 'payment_value']
df_cleaned.dropna(subset=payment_cols, inplace=True)

# Convert all timestamp columns to datetime objects before handling missing values
datetime_cols = [
    'shipping_limit_date', 'order_purchase_timestamp', 'order_approved_at',
    'order_delivered_carrier_date', 'order_delivered_customer_date',
    'order_estimated_delivery_date'
]
for col in datetime_cols:
    df_cleaned[col] = pd.to_datetime(df_cleaned[col], errors='coerce')

# Drop rows where critical date columns are missing
df_cleaned.dropna(subset=['order_approved_at', 'order_delivered_carrier_date', 'order_delivered_customer_date'], inplace=True)

# Fill missing product-related numerical columns with their median
product_numeric_cols = [
    'product_name_lenght', 'product_description_lenght', 'product_photos_qty',
    'product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm'
]
for col in product_numeric_cols:
    df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())

# Fill missing categorical product columns with 'unknown'
df_cleaned['product_category_name'] = df_cleaned['product_category_name'].fillna('unknown')
df_cleaned['product_category_name_english'] = df_cleaned['product_category_name_english'].fillna('unknown')

# Fill missing review_score with the median
df_cleaned['review_score'] = df_cleaned['review_score'].fillna(df_cleaned['review_score'].median())

# Fill missing customer and seller latitude/longitude with their median
geo_cols = ['customer_lat', 'customer_lng', 'seller_lat', 'seller_lng']
for col in geo_cols:
    df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())

# Fill missing review comment titles and messages with 'no_comment'
df_cleaned['review_comment_title'] = df_cleaned['review_comment_title'].fillna('no_comment')
df_cleaned['review_comment_title'] = df_cleaned['review_comment_message'].fillna('no_comment')

print("--- After handling missing values ---")
print(df_cleaned.isnull().sum().to_markdown(numalign="left", stralign="left"))
print(f"\nShape after handling missing values: {df_cleaned.shape}")

# =============================== Remove Duplicate Records =============================== #

num_duplicates_before = df_cleaned.duplicated().sum()
df_cleaned.drop_duplicates(inplace=True)
num_duplicates_after = df_cleaned.duplicated().sum()

print(f"Number of duplicate rows removed: {num_duplicates_before - num_duplicates_after}")
print(f"Shape after removing duplicates: {df_cleaned.shape}")

# =============================== Correct Data Types =============================== #

print("--- Data Types after initial conversion (from Step 1) ---")
df_cleaned.info()

# =============================== Fix Data Inconsistencies =============================== #

# Standardize text-based columns (e.g., product categories)
df_cleaned['product_category_name_english'] = df_cleaned['product_category_name_english'].str.lower().str.strip()
df_cleaned['product_category_name'] = df_cleaned['product_category_name'].str.lower().str.strip()
df_cleaned['customer_city'] = df_cleaned['customer_city'].str.lower().str.strip()
df_cleaned['customer_state'] = df_cleaned['customer_state'].str.lower().str.strip()
df_cleaned['seller_city'] = df_cleaned['seller_city'].str.lower().str.strip()
df_cleaned['seller_state'] = df_cleaned['seller_state'].str.lower().str.strip()
df_cleaned['payment_type'] = df_cleaned['payment_type'].str.lower().str.strip()
df_cleaned['order_status'] = df_cleaned['order_status'].str.lower().str.strip()

# Correct negative or unrealistic values (e.g., negative prices or order quantities)
# Check for negative values in 'price', 'freight_value', 'payment_value', 'payment_installments', 'review_score'
numeric_check_cols = ['price', 'freight_value', 'payment_value', 'payment_installments', 'review_score']
for col in numeric_check_cols:
    if (df_cleaned[col] < 0).any():
        print(f"\nFound negative values in {col}. Setting them to absolute value.")
        df_cleaned[col] = df_cleaned[col].abs()

# Check product dimensions and weights for non-positive values
product_dims_weights = ['product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm']
for col in product_dims_weights:
    if (df_cleaned[col] <= 0).any():
        print(f"\nFound non-positive values in {col}. Setting them to a small positive value (1.0).")
        df_cleaned[col] = df_cleaned[col].replace(0, 1).replace(np.nan, 1) # If 0, replace with 1, or if NaN replace with 1, assuming that a product must have some dimension/weight.

print("\n--- After fixing data inconsistencies ---")
print("Sample of standardized product categories:")
print(df_cleaned[['product_category_name', 'product_category_name_english']].head().to_markdown(index=False, numalign="left", stralign="left"))

# ===============================  Handle Outliers =============================== #

# Using IQR method for 'price', 'freight_value', and 'payment_value'
outlier_cols = ['price', 'freight_value', 'payment_value']
for col in outlier_cols:
    Q1 = df_cleaned[col].quantile(0.25)
    Q3 = df_cleaned[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Cap outliers
    df_cleaned[col] = np.where(df_cleaned[col] < lower_bound, lower_bound, df_cleaned[col])
    df_cleaned[col] = np.where(df_cleaned[col] > upper_bound, upper_bound, df_cleaned[col])
    print(f"\nOutliers in '{col}' capped using IQR method. Lower Bound: {lower_bound:.2f}, Upper Bound: {upper_bound:.2f}")

# ===============================  Create New Columns (if necessary) =============================== #

# Extract insights from order_purchase_timestamp
df_cleaned['order_purchase_year'] = df_cleaned['order_purchase_timestamp'].dt.year
df_cleaned['order_purchase_month'] = df_cleaned['order_purchase_timestamp'].dt.month
df_cleaned['order_purchase_day'] = df_cleaned['order_purchase_timestamp'].dt.day
df_cleaned['order_purchase_day_of_week'] = df_cleaned['order_purchase_timestamp'].dt.dayofweek # Monday=0, Sunday=6
df_cleaned['order_purchase_hour'] = df_cleaned['order_purchase_timestamp'].dt.hour

# Calculate total revenue per order item (price + freight_value) for individual item analysis
df_cleaned['item_total_revenue'] = df_cleaned['price'] + df_cleaned['freight_value']

print("\n--- After creating new columns ---")
print(df_cleaned[['order_purchase_timestamp', 'order_purchase_year', 'order_purchase_month', 'order_purchase_day', 'order_purchase_day_of_week', 'order_purchase_hour', 'item_total_revenue', 'payment_value']].head().to_markdown(index=False, numalign="left", stralign="left"))

# ===============================  Save the Cleaned Dataset =============================== #

cleaned_file_path = 'olist_merged_data_cleaned.csv'
df_cleaned.to_csv(cleaned_file_path, index=False)
print(f"\nCleaned dataset saved to {cleaned_file_path}")