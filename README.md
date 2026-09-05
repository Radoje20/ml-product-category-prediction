# Product Category Prediction

This project predicts a product's category based on its product title, using a dataset of 30,000+ products.

## Project structure
- `data/IMLP6_TASK_03-products.csv` — the dataset
- `notebooks/product_category_analysis.ipynb` — full EDA, cleaning, feature engineering, and model comparison
- `train_model.py` — trains the final model and saves it to `model/category_model.pkl`
- `predict_category.py` — loads the saved model and lets you interactively test product titles
- `model/category_model.pkl` — the trained model file

## How to run

1. Install dependencies:
   pip install pandas scikit-learn joblib

2. Train the model:
   python train_model.py

3. Test the model interactively:
   python predict_category.py

## Approach
- Cleaned malformed column names (stray spaces/underscores) and missing values, standardized category labels
- Engineered features from the product title: character length, word count, presence of digits
- Compared Logistic Regression, Random Forest, and SVM — chose Random Forest based on its overall accuracy and F1 performance across categories
- Final model trained on the full dataset and saved with joblib