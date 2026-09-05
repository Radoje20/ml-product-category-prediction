import joblib
import pandas as pd

# Load the saved model
model = joblib.load("model/category_model.pkl")

print("Model loaded successfully!")
print("Type 'exit' at any point to stop.\n")

while True:
    title = input("Enter product title: ")
    if title.lower() == "exit":
        print("Exiting...")
        break

    title_length = len(title)
    title_word_count = len(title.split())
    has_digit = int(any(char.isdigit() for char in title))

    user_input = pd.DataFrame([{
        "Product Title": title,
        "title_length": title_length,
        "title_word_count": title_word_count,
        "has_digit": has_digit
    }])

    prediction = model.predict(user_input)[0]
    print(f"Predicted category: {prediction}\n" + "-" * 40)