# Language Detection Web Application

This project is a web-based language detection system built using **Flask**, **Scikit-learn**, and **JavaScript**. It utilizes a **Naive Bayes classifier** trained on textual data to predict the language of a given sentence. The application features an interactive and responsive frontend with a modern UI design.

---

## Project Overview

The goal of this project is to detect the language of input text in real-time using a machine learning model. It includes:

- Data preprocessing and feature extraction using `CountVectorizer`
- Training a `Multinomial Naive Bayes` model
- Persisting the model and vectorizer with `pickle` for efficient reuse
- Building a user interface with HTML, CSS, and JavaScript
- Integrating backend and frontend using Flask

---

## Directory Structure

``` language-detector/ 
├── app.py # Flask application
├── langdetdata.csv # Dataset containing text and corresponding languages
├── language_model.pkl # Trained Naive Bayes model
├── vectorizer.pkl # CountVectorizer instance
├── templates/
│ └── index.html # Frontend HTML page
```
---

## Technologies Used

- Python 3
- Flask for backend web framework
- scikit-learn for machine learning
- pandas, numpy for data manipulation
- HTML, CSS, JavaScript for frontend development
- pickle for model serialization

---

## Getting Started

### Prerequisites

Make sure you have Python 3 and `pip` installed on your system.

### Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/language-detector.git
cd language-detector
```

2. Install required packages:

```bash
pip install -r requirements.txt
```
If requirements.txt is not available, install manually:
```bash
pip install flask pandas numpy scikit-learn
```

3. Run the Flask server:

```bash
python app.py
```

4. Open your browser and navigate to http://127.0.0.1:5000/.

## How It Works

1. The dataset (langdetdata.csv) is used to train a model that classifies sentences by language.
2. CountVectorizer transforms the text into numerical features.
3. A MultinomialNB model is trained and saved using pickle.
4. The user inputs a sentence in the frontend.
5. The backend receives the input, vectorizes it, predicts the language, and returns the result to be displayed on the UI.

