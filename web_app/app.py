from flask import Flask, request, jsonify
from keras.models import load_model
import re
import nltk
from langdetect import detect
import pickle
from keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

def detect_lang(content=""):
    return detect(content)

def preprocess_text(text, lang="english"):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    stop_words = set(nltk.corpus.stopwords.words(lang))
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
DE_MODEL = load_model("de_lstm.h5")
DE_TOKENIZER = get_tokenizer("de")
DE_LABELS = get_labels("de")
TOKENIZER_MAX_LEN = 2000
def predict_en(text):
    text = preprocess_text(text)

    print("preprocessed text: ")
    print(text)

    print("TOKENIZER")
    # print(EN_TOKENIZER.word_index)

    print("text len:",len(text))

    seq = EN_TOKENIZER.texts_to_sequences([text])
    print("seq len: ", len(seq))
    # print("sequence?")
    print(seq)
    padded_seq = pad_sequences(seq, maxlen=TOKENIZER_MAX_LEN, padding='post', truncating='post')
    pred_probs = EN_MODEL.predict(padded_seq)

    print(pred_probs)


    pred_class_index = pred_probs.argmax(axis=-1)[0]
    predicted_class = EN_LABELS.inverse_transform([pred_class_index])[0]
    print("Predykowana klasa:", predicted_class)
    class_probabilities = {label: float(prob) for label, prob in zip(EN_LABELS.classes_, pred_probs[0])}
    print(class_probabilities)
    print("Wektor prawdopodobieństw:", pred_probs[0])
    return class_probabilities

def predict_de(text):
    return "ok"

####################################################################################

@app.route('/echo', methods=['POST'])
def echo():
    data = request.get_json()
    
    if not data or 'message' not in data:
        return jsonify({'error': 'Missing "message" field in JSON'}), 400

    subject = data['subject']
    message = data['message']
    lang = detect_lang(message)
    print(lang)
    res = predict_en(subject + ' '+ message)
    print(res)

    return jsonify({"result": res})

if __name__ == '__main__':
    app.run(debug=True)