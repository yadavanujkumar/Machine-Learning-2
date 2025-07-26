import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Load dataset
df = pd.read_csv(r'e:\anuj\sarforsh-sentiment-analysis\src\data\sarfarosh_movie_reviews.csv')

# Basic preprocessing
df['Review'] = df['Review'].astype(str).str.lower()

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['Review'], df['Review Sentiment'], test_size=0.2, random_state=42
)

# Vectorize text
vectorizer = CountVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X_train_vec, y_train)

# Evaluate
y_pred = model.predict(X_test_vec)
print(classification_report(y_test, y_pred))