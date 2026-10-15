
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pickle 
import os


df = pd.read_csv("data/spam.csv", encoding="latin-1")

df = df[["v1", "v2"]]
df.columns = ["label", "message"]

df = df.drop_duplicates(keep="first")
df = df.dropna()

df["label"] = df["label"].map({
    "ham": 0,
    "spam": 1
})

print("First five rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nLabel distribution:")
print(df["label"].value_counts())

print("\nMissing values:")
print(df.isnull().sum())

X = df["message"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining messages:", len(X_train))
print("Testing messages:", len(X_test))

print("\nTraining label distribution:")
print(y_train.value_counts())

print("\nTesting label distribution:")
print(y_test.value_counts())

tfidf = TfidfVectorizer(max_features=3000)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print("\nTF-IDF conversion completed!")
print("Training feature shape:", X_train_tfidf.shape)
print("Testing feature shape:", X_test_tfidf.shape)


model = MultinomialNB()

model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel training completed!")
print("Model accuracy:", round(accuracy * 100, 2), "%")



os.makedirs("models", exist_ok=True)

pickle.dump(model, open("models/spam_model.pkl", "wb"))
pickle.dump(tfidf, open("models/tfidf_vectorizer.pkl", "wb"))

print("Model saved successfully!")
