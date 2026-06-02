import pandas as pd
import nltk
import string

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

nltk.download('stopwords')

# Load dataset
data = pd.read_csv(
    'dataset/spam.csv',
    sep='\t',
    names=['label', 'message']
)

# Preprocessing
def preprocess(text):
    text = text.lower()

    text = ''.join(
        char for char in text
        if char not in string.punctuation
    )

    words = text.split()

    words = [
        word for word in words
        if word not in stopwords.words('english')
    ]

    return ' '.join(words)

data['message'] = data['message'].apply(preprocess)

# TF-IDF
vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(data['message'])

y = data['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = MultinomialNB()

model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

print("Accuracy:")
print(accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d')

plt.title("Spam Detection Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()