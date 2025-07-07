from flask import Flask, request, jsonify
from keras.models import load_model
import re
import nltk
from langdetect import detect
import pickle
from keras.preprocessing.sequence import pad_sequences
import re
from nltk.corpus import stopwords
import nltk
import time

app = Flask(__name__)
nltk.download('stopwords')

def detect_lang(content=""):
    return detect(content)

def multi_instance_detect_lang(data):
    res = []
    for entry in data:
        lng = detect(entry['message'])
        if lng != 'en':
            res.append(entry['id'])
    return None if len(res) == 0 else res

def preprocess_text(text, lang="english"):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    stop_words = set(stopwords.words(lang))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

def get_tokenizer(lang="en"):
    with open(lang+"_lstm_tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return tokenizer

def get_labels(lang="en"):
    with open(lang+"_lstm_labels.pkl", "rb") as f:
        le = pickle.load(f)
    return le

EN_MODEL = load_model("en_lstm.h5")
EN_TOKENIZER = get_tokenizer("en")
EN_LABELS = get_labels("en")
TOKENIZER_MAX_LEN = 1000
def predict_en(text):
    text = preprocess_text(text)
    seq = EN_TOKENIZER.texts_to_sequences([text])

    padded_seq = pad_sequences(seq, maxlen=TOKENIZER_MAX_LEN, padding='post', truncating='post')
    pred_probs = EN_MODEL.predict(padded_seq)
    pred_class_index = pred_probs.argmax(axis=-1)[0]
    predicted_class = EN_LABELS.inverse_transform([pred_class_index])[0]
    class_probabilities = {label: float(prob) for label, prob in zip(EN_LABELS.classes_, pred_probs[0])}
    
    print("Predicted class:", predicted_class)
    print("Probs:", pred_probs[0])
    return class_probabilities

def predict_batch_en(texts_with_subjects):
    # Preprocess all texts
    texts = [obj["subject"] + ' ' + obj["message"] for obj in texts_with_subjects]
    processed_texts = [preprocess_text(text) for text in texts]
    
    # Convert texts to sequences and pad them
    sequences = EN_TOKENIZER.texts_to_sequences(processed_texts)
    padded_sequences = pad_sequences(sequences, maxlen=TOKENIZER_MAX_LEN, padding='post', truncating='post')
    
    # Perform batch prediction
    batch_probs = EN_MODEL.predict(padded_sequences)
    
    # Prepare results
    results = []
    it = 0
    for probs in batch_probs:
        class_probabilities = {label: float(prob) for label, prob in zip(EN_LABELS.classes_, probs)}
        results.append(
            {
                "id": texts_with_subjects[it]['id'],
                "result": class_probabilities
            })
        it+=1
    
    return results

####################################################################################

@app.route('/process-single', methods=['POST'])
def process_single():
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({'error': 'Missing "message" field in JSON'}), 400
    elif 'id' not in data:
        return jsonify({'error': 'Missing "id" field in JSON'}), 400
    subject = data['subject']
    message = data['message']
    id = data['id']
    lang = detect_lang(message)
    start_time = time.time()
    res = predict_en(subject + ' ' + message)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Single prediction time: {execution_time:.4f}")
    error_message = None if lang == "en" else "Error! A non-English language was detected."
    return jsonify({
        "id": id,
        "result": res,
        "error": error_message
        })

@app.route('/process-multiple', methods=['POST'])
def process_multiple():
    multi_data = request.get_json()
    if not multi_data or 'data' not in multi_data:
        return jsonify({'error': 'Missing "data" field in JSON'}), 400
    messages_list = multi_data['data']
    non_english_entries = multi_instance_detect_lang(messages_list)
    start_time = time.time()
    res = predict_batch_en(messages_list)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Multi [{str(len(messages_list))}] prediction time: {execution_time:.4f}")
    error_message = None if non_english_entries is None else f"Error! Some entries: {non_english_entries} have languge different than english."
    return jsonify({
        "result": res,
        "error": error_message
        })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)