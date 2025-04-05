from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import pickle

app = Flask(__name__)

# Load dataset and train model
data = pd.read_csv("langdetdata.csv")
x = np.array(data["Text"])
y = np.array(data["language"])

cv = CountVectorizer()
X = cv.fit_transform(x)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

# Save model and vectorizer to avoid retraining
pickle.dump(model, open("language_model.pkl", "wb"))
pickle.dump(cv, open("vectorizer.pkl", "wb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    sentence = data.get("sentence", "")
    
    if not sentence:
        return jsonify({"error": "No input provided"}), 400
    
    # Load trained model and vectorizer
    model = pickle.load(open("language_model.pkl", "rb"))
    cv = pickle.load(open("vectorizer.pkl", "rb"))
    
    # Transform input and predict
    transformed_input = cv.transform([sentence]).toarray()
    prediction = model.predict(transformed_input)[0]
    
    return jsonify({"language": prediction})

if __name__ == '__main__':
    app.run(debug=True)
