# =============================== Importing libraries =============================== #

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# =============================== Loading dataset =============================== #

def load_dataset(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print(f"Failed to load dataset: {e}")

orders_data = load_dataset('olist_orders_dataset.csv')
order_items_data = load_dataset('olist_order_items_dataset.csv')
products_data = load_dataset('olist_products_dataset.csv')
product_category_name_translation_data = load_dataset('product_category_name_translation.csv')
customers_data = load_dataset('olist_customers_dataset.csv')
sellers_data = load_dataset('olist_sellers_dataset.csv')
order_payments_data = load_dataset('olist_order_payments_dataset.csv')
order_reviews_data = load_dataset('olist_order_reviews_dataset.csv')
geolocation_data = load_dataset('olist_geolocation_dataset.csv')

# =============================== Merge the datasets =============================== #

# Merge order items with products
merged_data = pd.merge(order_items_data, products_data, on='product_id', how='left')

# Merge with product category name translation
merged_data = pd.merge(merged_data, product_category_name_translation_data, on='product_category_name', how='left')

# Merge with orders
merged_data = pd.merge(merged_data, orders_data, on='order_id', how='left')

# Merge with customers
merged_data = pd.merge(merged_data, customers_data, on='customer_id', how='left')

# Merge customer geolocation data
customer_geolocation_data = geolocation_data[geolocation_data['geolocation_zip_code_prefix'].isin(customers_data['customer_zip_code_prefix'])]
customer_geolocation_data = customer_geolocation_data.groupby('geolocation_zip_code_prefix').agg({
    'geolocation_lat': 'mean',
    'geolocation_lng': 'mean'
}).reset_index()
customer_geolocation_data = customer_geolocation_data.rename(columns={
    'geolocation_zip_code_prefix': 'customer_zip_code_prefix',
    'geolocation_lat': 'customer_lat',
    'geolocation_lng': 'customer_lng'
})
merged_data = pd.merge(merged_data, customer_geolocation_data, on='customer_zip_code_prefix', how='left')

# Merge with sellers
merged_data = pd.merge(merged_data, sellers_data, on='seller_id', how='left')

# Merge seller geolocation data
seller_geolocation_data = geolocation_data[geolocation_data['geolocation_zip_code_prefix'].isin(sellers_data['seller_zip_code_prefix'])]
seller_geolocation_data = seller_geolocation_data.groupby('geolocation_zip_code_prefix').agg({
    'geolocation_lat': 'mean',
    'geolocation_lng': 'mean'
}).reset_index()
seller_geolocation_data = seller_geolocation_data.rename(columns={
    'geolocation_zip_code_prefix': 'seller_zip_code_prefix',
    'geolocation_lat': 'seller_lat',
    'geolocation_lng': 'seller_lng'
})
merged_data = pd.merge(merged_data, seller_geolocation_data, on='seller_zip_code_prefix', how='left')

# Merge with order payments
order_payments_data = order_payments_data.groupby('order_id').agg({
    'payment_sequential': 'count',
    'payment_type': lambda x: ', '.join(x),
    'payment_installments': 'sum',
    'payment_value': 'sum'
}).reset_index()
merged_data = pd.merge(merged_data, order_payments_data, on='order_id', how='left')

# Merge with order reviews
order_reviews_data = order_reviews_data.groupby('order_id').agg({
    'review_score': 'mean',
    'review_comment_title': lambda x: ', '.join(x.astype(str)),
    'review_comment_message': lambda x: ', '.join(x.astype(str))
}).reset_index()
merged_data = pd.merge(merged_data, order_reviews_data, on='order_id', how='left')

# Save the merged data to a CSV file
merged_data.to_csv('olist_merged_data.csv', index=False)

# =============================== Explore the Dataset Structure =============================== #

# Display the first few rows
print("First 5 rows of the dataset:", merged_data.head())

# Explore the dataset structure
print("\nDataset Information:", merged_data.info())

# Check the total number of rows and columns
print(f"Number of Rows: {merged_data.shape[0]}, Number of Columns: {merged_data.shape[1]}")

# Identify different data types
print("Data Types:\n", merged_data.dtypes)

# Check for missing values
print("\nMissing values in each column:")
print(merged_data.isnull().sum().to_markdown(numalign="left", stralign="left"))

# Check for duplicate rows
print(f"\nNumber of duplicate rows: {merged_data.duplicated().sum()}")

# =============================== Document observations =============================== #

with open('data_exploration_summary.txt', 'w') as f:
    f.write("Dataset has {} rows and {} columns.\n".format(merged_data.shape[0], merged_data.shape[1]))
    f.write("Many columns have missing values. All date/time columns are currently stored as object (string) data types. These will need to be converted to proper datetime objects for further analysis.\n")
    f.write("Duplicates found: {}. \n".format(merged_data.duplicated().sum()))
    f.write("Key columns for analysis: \n"
            "1) Sales and Revenue: price, freight_value, payment_value, order_item_id. \n"
            "2) Customer Segmentation: customer_id, customer_unique_id, order_purchase_timestamp \n"
            "(for purchase frequency and recency), payment_value (for monetary value), \n"
            "customer_city, customer_state, review_score. \n"
            "3) Geographical Information: customer_city, customer_state, seller_city, seller_state.\n")