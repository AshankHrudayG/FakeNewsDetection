import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Download stopwords
nltk.download('stopwords')

# Load dataset
df = pd.read_csv("news.csv")

# Convert labels to numeric format (Fake=1, Real=0)
df['label'] = df['label'].map({'fake': 1, 'real': 0})  # Ensure correct mapping

# Check if labels are correctly assigned
print(df['label'].value_counts())

# Function to clean text data
def clean_text(text):
    text = text.lower()
    text = "".join([char for char in text if char not in string.punctuation])
    text = " ".join([word for word in text.split() if word not in stopwords.words('english')])
    return text

# Apply text cleaning
df['text'] = df['text'].apply(clean_text)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1,2))
X = vectorizer.fit_transform(df['text']).toarray()
y = df['label']  # Labels (Fake=1, Real=0)

# Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest Classifier with better parameters
model = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))

# Save the trained model and vectorizer
joblib.dump(model, 'fake_news_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

print("Model and Vectorizer saved successfully!")
