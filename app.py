from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

# Load the model and vectorizer
model = pickle.load(open("spam_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    email_content = [request.form['email_content']]

    
    # Vectorize the input
    email_vector = vectorizer.transform(email_content)
    
    # Make prediction
    prediction = model.predict(email_vector)[0]
    
    result = "Spam" if prediction == 1 else "Not Spam"
    
    return render_template("index.html", prediction=result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)

