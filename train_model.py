import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load data
df = pd.read_csv("data/IMLP6_TASK_03-products.csv")

# Clean column names
df.columns = df.columns.str.strip().str.lstrip('_')

# Clean data
df = df.dropna(subset=['Product Title', 'Category Label'])
df['Category Label'] = df['Category Label'].astype(str).str.strip()
df = df.drop(columns=['product ID', 'Merchant ID', 'Product Code'], errors='ignore')

# Feature engineering
df['title_length'] = df['Product Title'].astype(str).str.len()
df['title_word_count'] = df['Product Title'].astype(str).str.split().str.len()
df['has_digit'] = df['Product Title'].astype(str).str.contains(r'\d', regex=True).astype(int)

# Features and label
X = df[["Product Title", "title_length", "title_word_count", "has_digit"]]
y = df["Category Label"]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("title", TfidfVectorizer(max_features=5000), "Product Title"),
        ("numeric", MinMaxScaler(), ["title_length", "title_word_count", "has_digit"])
    ]
)

# Pipeline with the chosen model
pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=100, max_depth=30))
])

# Train on the full dataset
pipeline.fit(X, y)

# Save the model
joblib.dump(pipeline, "model/category_model.pkl")

print("Model trained and saved as 'model/category_model.pkl'")  