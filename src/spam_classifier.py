import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

# Load sms dataset in the classifier file
messages = pd.read_csv(
    "data/sms+spam+collection/SMSSpamCollection",
    sep="\t",
    names=["label", "message"]
)

# X: Messages, y: Labels
X = messages["message"]
y = messages["label"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

tfidf = TfidfVectorizer()

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

model = MultinomialNB()

model.fit(X_train_tfidf, y_train)
predictions = model.predict(X_test_tfidf)
print(classification_report(y_test, predictions))

ConfusionMatrixDisplay.from_predictions(y_test, predictions)

plt.title("Naive Bayes Confusion Matrix")
plt.tight_layout()
plt.savefig("results/naive_bayes_confusion_matrix.png")
plt.close()