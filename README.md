# SMS Spam Classifier

## Project Overview

The SMS Spam Classifier is a beginner-friendly machine learning project that identifies whether a text message is **Spam** or **Not Spam (Ham)**. It uses Natural Language Processing (NLP) techniques and a Naive Bayes machine learning algorithm to classify messages.

The project provides a simple web interface built with Streamlit, where users can enter an SMS message and receive a prediction.

## Features

- Classifies SMS messages as Spam or Not Spam.
- Cleans and preprocesses the dataset.
- Converts text messages into numerical features using TF-IDF.
- Trains a Naive Bayes machine learning model.
- Evaluates the model using test data.
- Provides an interactive web interface using Streamlit.
- Saves the trained model for future predictions.

## Technologies Used

- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorization
- Multinomial Naive Bayes
- Streamlit
- Pickle

## Project Structure

```text
SMS calssifier/
├── data/
│   └── spam.csv
├── models/
│   ├── spam_model.pkl
│   └── tfidf_vectorizer.pkl
├── venv/
├── train_model.py
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

## How It Works

1. **Data Collection:** Loads SMS messages from the dataset.
2. **Data Preprocessing:** Selects the required columns, removes duplicate and missing records, and converts labels into numerical values.
3. **Feature Extraction:** Uses TF-IDF to convert text messages into numerical features.
4. **Model Training:** Trains a Multinomial Naive Bayes classifier using the training data.
5. **Model Evaluation:** Evaluates the model's accuracy using the test data.
6. **Prediction:** Loads the saved model and vectorizer to classify messages entered by the user.
7. **Web Interface:** Displays the prediction using Streamlit.

## Installation and Setup

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

Navigate into the project folder:

```bash
cd "SMS calssifier"
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Train the Model

```bash
python train_model.py
```

This creates the trained model and TF-IDF vectorizer files inside the `models/` directory.

### 5. Run the Application

```bash
python -m streamlit run app.py
```

Open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

## Example

**Input message:**

```text
Congratulations! You won a free prize. Claim now!
```

**Expected output:** Spam

**Input message:**

```text
Hey, are you coming to college tomorrow?
```

**Expected output:** Not Spam

*Note: These are example predictions. Actual results depend on the trained model.*

## Machine Learning Concepts

- **NLP:** Techniques used to process and analyze human language.
- **TF-IDF:** Represents words numerically based on their importance in messages.
- **Naive Bayes:** A probabilistic machine learning algorithm used for classification.
- **Train-Test Split:** Divides the dataset into training and testing portions.
- **Accuracy:** Measures the proportion of correct predictions on the test dataset.

## Future Improvements

- Add precision, recall, and F1-score evaluation.
- Improve text preprocessing.
- Enhance the web interface.
- Deploy the application online.
- Test the model on more diverse SMS messages.

## Learning Outcomes

- Learned basic data preprocessing with Pandas.
- Understood text feature extraction using TF-IDF.
- Trained and evaluated a machine learning model.
- Learned to save and load models using Pickle.
- Built an interactive machine learning application using Streamlit.

## Author

**B.Tech Computer Science and Engineering Student**

## License

This project is intended for educational and learning purposes.
