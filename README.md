# E-Commerce Sales Analysis and Forecasting

## Overview

- This project analyzes an e-commerce dataset to uncover sales trends, customer behavior, and future sales predictions.
- Using data from the Olist E-Commerce Public Dataset, we performed exploratory data analysis (EDA), RFM (Recency, Frequency, Monetary) segmentation, time series forecasting, and identify business opportunities.
- The goal is to provide actionable insights for improving sales, customer retention, and business growth.

## Key technologies used:

- `Programming Language:` Python
- `Libraries:` Pandas, NumPy, Matplotlib, Seaborn, statsmodels.tsa.api, sklearn.metrics
- `Dataset:` [Olist E-Commerce dataset from Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)

This repository contains the code, visualizations, and reports generated during the analysis.

## Project Structure

The repository is organized as follows:

- `data/:` Contains the raw and processed datasets (e.g., olist_orders_dataset.csv, daily_sales.csv).
- `files/:` Python code files:
  - 01_download_explore_dataset.py: Download and explore the dataset.
  - 02_data_cleaning_preprocessing.py: Handle missing values, duplicates, and data types.
  - 03_sales_trend_analysis.py: Analyze Monthly and yearly sales trends, compare Peak and Low Sales Periods.
  - 04_best_selling_products_cat.py: Identify best-selling products and categories with visualizations. 
  - 05_geographic_sales_performance.py: Geographic sales performance with visualizations.
  - 06_customer_segmentation_rfm.py: Calculate RFM metrics and scores.
  - 07_segment_customers_rfm_scores.py: Segment customers based on RFM scores.
  - 08_visualize_customer_segments.py: Visualization of Customer Segments.
  - 09_time_series_data_preparation.py: Time series data preparations.
  - 10_sales_forecasting_model.py: Prepares time series data, trains a Holt-Winters forecasting model.
  - 11_predict_future_sales.py: Future sales predictions.
  - 12_identify_business_opportunities.py: Analyze high-growth product categories and identify regional sales growth areas.
- `reports/:`
  - `pdf/:`
    - Indian E-commece Market Research.html
    - olist_ecommerce_analysis_report.pdf
    - [olist_streamlit_sales_forecasting_dashboard](https://ecommerce-sales-forecasting-2025.streamlit.app)
  - `summaries/:` text files with observations, finding and suggestions.
  - `figures/:` png visualization charts files

Thank you for visiting!
  
