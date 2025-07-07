# Multinominal Naive Bayes

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("en-3-classes.csv")
# df = pd.read_csv("en-tickets.csv")


df['text'] = df['subject'].fillna('') + ' ' + df['body'].fillna('')
df = df.dropna(subset=['type'])

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["type"], test_size=0.3, random_state=42
)

tfidf = TfidfVectorizer()
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)

acc = accuracy_score(y_test, y_pred)

print("=== Raport klasyfikacji ===")
print(classification_report(y_test, y_pred))
print('Acc:', acc)

cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel("Predykcja")
plt.ylabel("Prawda")
plt.title("Macierz pomyłek")
plt.show()