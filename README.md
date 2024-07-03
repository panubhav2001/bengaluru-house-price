# Bengaluru House Price Prediction

This project aims to predict house prices in Bengaluru using various machine learning models. The process involves data preprocessing, outlier removal, feature engineering, model training, and evaluation.

## Table of Contents

- [Project Overview](#project-overview)
- [Data Preprocessing](#data-preprocessing)
- [Outlier Removal](#outlier-removal)
- [Feature Engineering](#feature-engineering)
- [Model Building and Evaluation](#model-building-and-evaluation)
- [Model Selection and Saving](#model-selection-and-saving)
- [Dependencies](#dependencies)
- [How to Run](#how-to-run)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

The goal of this project is to build a predictive model for house prices in Bengaluru. The dataset used is `Bengaluru_House_Data.csv`. The project involves several steps including data cleaning, feature engineering, outlier removal, and applying machine learning models.

## Data Preprocessing

1. **Reading the Data**: Loaded the dataset using Pandas.
2. **Initial Exploration**: Inspected the data for null values and data types.
3. **Handling Missing Values**:
   - Filled missing values in categorical columns ('location', 'size') with the mode.
   - Filled missing values in numerical columns ('bath', 'balcony') with the median.
4. **Feature Selection and Engineering**:
   - Dropped irrelevant columns ('area_type', 'society', 'availability').
   - Extracted the number of bedrooms ('bhk') from the 'size' column.
   - Cleaned the 'total_sqft' column by handling ranges and non-numeric values.

## Outlier Removal

- **Price per Square Foot**: Created a new feature 'price_per_sqft'.
- **Outlier Removal**:
  - Removed data points with price per square foot values beyond one standard deviation from the mean for each location.
  - Further refined the dataset by removing properties with unusually low prices compared to properties with one less bedroom in the same location.

## Feature Engineering

- **Feature Scaling and Encoding**: Applied `StandardScaler` for numerical features and `OneHotEncoder` for the 'location' feature.

## Model Building and Evaluation

- **Train-Test Split**: Split the data into training and testing sets (80-20 split).
- **Models Applied**:
  - Linear Regression
  - Random Forest Regressor
  - AdaBoost Regressor
  - Gradient Boosting Regressor
- **Evaluation Metrics**:
  - R-square
  - Mean Absolute Percentage Error (MAPE)

## Model Selection and Saving

- **Best Model**: Random Forest Regressor was selected based on performance metrics.
- **Model Persistence**: Saved the trained Random Forest model pipeline using `pickle`.

## Dependencies

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- pickle

## How to Run

1. Clone the repository.
2. Ensure you have all dependencies installed.
3. Run main.py script.
4. Use the saved model for predictions.

```bash
git clone https://github.com/yourusername/bengaluru-house-price-prediction.git
cd bengaluru-house-price-prediction
pip install -r requirements.txt
python main.py
