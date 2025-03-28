from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load trained model and vectorizer
model = joblib.load('fake_news_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    news_text = request.form['news']
    
    # Ensure text is properly vectorized before prediction
    news_vector = vectorizer.transform([news_text]).toarray()
    prediction = model.predict(news_vector)[0]  # Extract single prediction value

    # Ensure correct output (1=Fake, 0=Real)
    result = "Fake News" if prediction == 1 else "Real News"
    
    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
