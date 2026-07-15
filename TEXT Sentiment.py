import pandas as pd
import numpy as np
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
df=pd.read_csv("customer sentiment -text.csv")
df
df.head()
df.info()
df.shape
df.columns
df.isnull().sum()
df=df.dropna()
df.duplicated().sum()
df=df.drop_duplicates()
df.shape
df.isnull().sum()
df["Sentiment"].value_counts()
df['Review'] = df['Review'].astype(str)
df['Review'] = df['Review'].str.lower()
df['Review'].head()
df["Review"] = df['Review'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x))
df['Review'].head()
df['Review'] = df['Review'].apply(word_tokenize)
stop_words = set(stopwords.words('english'))
df['Review'] = df['Review'].apply(
    lambda words: [word for word in words if word not in stop_words]
)
df['Review'].head()
df['Review'] = df['Review'].apply(lambda x: " ".join(x))
df['Review'].head()
x=df['Review']
y=df['Sentiment']
tfidf = TfidVectorizer()
x = tfidf.fit_transform(df['Review'])
print(x.shape)
le = LabelEncoder()
y = le.fit_transform(df['Sentiment'])
print(y)
x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2, random_state=42)
LogisticRegression(max_iter=1000, class_weight='balanced',)
model.fit(x_train, y_train)
y_pred=model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
