from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS if needed

# Load the trained model and vectorizer
model = joblib.load('category_classifier.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Load schemes data
schemes_data = pd.read_csv('schemes_data.csv')

def clean_text(text):
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text

def get_schemes(category):
    schemes = schemes_data[schemes_data['category'] == category]['scheme'].tolist()
    return schemes

# Serve the HTML file for the root URL
@app.route('/')
def serve_index():
    return send_from_directory('', 'index.html')

@app.route('/chatbot', methods=['POST'])
def chatbot():
    data = request.json
    user_input = data['input']
    
    # Clean and vectorize user input
    user_input_cleaned = clean_text(user_input)
    user_input_vectorized = vectorizer.transform([user_input_cleaned])
    
    # Predict the category
    predicted_category = model.predict(user_input_vectorized)[0]
    
    # Get schemes for the predicted category
    schemes = get_schemes(predicted_category)
    
    # Prepare the response
    response = {
        'category': predicted_category,
        'schemes': schemes
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
