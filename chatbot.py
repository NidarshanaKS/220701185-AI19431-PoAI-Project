import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Load and prepare the training dataset
train_data = pd.read_csv('training_data.csv')

# Function to clean text
def clean_text(text):
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text

# Clean the 'description' column
train_data['description'] = train_data['description'].apply(clean_text)

# Vectorization
vectorizer = TfidfVectorizer(max_features=500)
X = vectorizer.fit_transform(train_data['description'])
y = train_data['category']

# Train the model
model = LogisticRegression()
model.fit(X, y)

# Save the model and vectorizer
joblib.dump(model, 'category_classifier.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

# Load the schemes dataset
schemes_data = pd.read_csv('schemes_data.csv')

def load_model():
    model = joblib.load('category_classifier.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    return model, vectorizer

def get_schemes(category):
    # Filter schemes based on the category
    schemes = schemes_data[schemes_data['category'] == category]['scheme'].tolist()
    return schemes

def chatbot():
    model, vectorizer = load_model()
    
    print("Welcome! Please describe your situation:")
    user_input = input("User: ")
    
    # Clean and vectorize user input
    user_input_cleaned = clean_text(user_input)
    user_input_vectorized = vectorizer.transform([user_input_cleaned])
    
    # Predict the category
    predicted_category = model.predict(user_input_vectorized)[0]
    print(f"\nChatbot: You belong to the '{predicted_category}' category.")
    
    # Get schemes for the predicted category
    schemes = get_schemes(predicted_category)
    print(f"\nHere are the schemes available for the '{predicted_category}' category:")
    for scheme in schemes:
        print(f"- {scheme}")

# Start the chatbot
chatbot()
