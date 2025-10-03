from flask import Flask, request, jsonify, send_from_directory
import pickle

# Load models and objects
with open('logReg.pkl', 'rb') as f:
    logReg = pickle.load(f)
with open('nBayes.pkl', 'rb') as f:
    nBayes = pickle.load(f)
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
with open('encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

# Define the prediction function
def predict_category(model, text, vectorizer, encoder):
    text_tfidf = vectorizer.transform([text])
    pred_encoded = model.predict(text_tfidf)
    return encoder.inverse_transform(pred_encoded)[0]

# Initialize Flask
app = Flask(__name__)

@app.route('/home')
def home_page():
    return send_from_directory('.', 'index.html')

@app.route('/')
def home():
    return "Welcome to the BBC News Text Classification API!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    if 'text' not in data or 'model' not in data:
        return jsonify({'error': 'Provide both "text" and "model" fields'}), 400

    text = data['text']
    model_name = data['model'].lower()

    if model_name == 'logistic':
        model = logReg
    elif model_name == 'naive_bayes':
        model = nBayes
    else:
        return jsonify({'error': 'Model must be "logistic" or "naive_bayes"'}), 400

    prediction = predict_category(model, text, vectorizer, encoder)

    return jsonify({
        'text': text,
        'model': model_name,
        'predicted_category': prediction
    })

if __name__ == '__main__':
    app.run(debug=True)
