import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from transformers import BertTokenizer, TFBertModel, TFBertForSequenceClassification
from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.callbacks import EarlyStopping, Callback
import tensorflow as tf
from tensorflow.python.platform import build_info as tf_build_info
from transformers import create_optimizer
import re
from nltk.corpus import stopwords
import nltk
from sklearn.utils.class_weight import compute_class_weight

nltk.download('stopwords')

lang, lang_long = "en", "english"

def preprocess_text(text, lang="english"):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    stop_words = set(stopwords.words(lang))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# 1. Wczytanie danych
df = pd.read_csv('en-tickets.csv')
df['text'] = df['subject'].fillna('') + ' ' + df['body'].fillna('')
df['text'] = df['text'].astype(str).apply(lambda x: preprocess_text(x, lang=lang_long))
df = df.dropna(subset=['type'])


# 2. Kodowanie etykiet
le = LabelEncoder()
labels = le.fit_transform(df['type'])
labels_cat = to_categorical(labels)

# 3. Tokenizacja z użyciem BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 4. Tokenizacja tekstu
encodings = tokenizer(
    list(df['text']),
    truncation=True,
    padding=True,
    max_length=212,
    return_tensors='tf'
)

input_ids = encodings['input_ids'].numpy()
attention_mask = encodings['attention_mask'].numpy()

X_train_input_ids, X_temp_input_ids, \
X_train_attention_mask, X_temp_attention_mask, \
y_train, y_temp = train_test_split(
    input_ids,
    attention_mask,
    labels_cat,
    test_size=0.3,
    random_state=42
)

X_val_input_ids, X_test_input_ids, \
X_val_attention_mask, X_test_attention_mask, \
y_val, y_test = train_test_split(
    X_temp_input_ids,
    X_temp_attention_mask,
    y_temp,
    test_size=0.5,
    random_state=42
)

# 6. Budowa modelu (POPRAWIONA)
from transformers import TFBertModel

# Użyj podstawowego modelu BERT bez warstwy klasyfikacyjnej
bert_model = TFBertModel.from_pretrained('bert-base-uncased')

# Wejścia
input_ids = Input(shape=(212,), dtype=tf.int32, name='input_ids')
attention_mask = Input(shape=(212,), dtype=tf.int32, name='attention_mask')

# Wyjście BERTa (używamy pooler_output - reprezentacja [CLS])
bert_output = bert_model(input_ids, attention_mask=attention_mask)
pooled_output = bert_output.pooler_output

# Dodaj własne warstwy klasyfikacyjne
x = Dropout(0.1)(pooled_output)
x = Dense(256, activation='relu')(x)  # Warstwa pośrednia
x = Dropout(0.2)(x)
outputs = Dense(4, activation='softmax')(x)  # Warstwa wyjściowa

model = Model(inputs=[input_ids, attention_mask], outputs=outputs)

# Optymalizator (mniejszy learning rate dla BERTa)
num_train_steps = len(X_train_input_ids) // 32 * 10  # 10 epok
my_optimizer, _ = create_optimizer(
    init_lr=2e-5,  # Typowy LR dla fine-tuningu BERT
    num_train_steps=num_train_steps,
    num_warmup_steps=num_train_steps // 10
)

model.compile(
    optimizer=my_optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']  # Poprawna metryka dla klasyfikacji
)

# 7. Trening (z dodanym monitoringiem)
early_stop = EarlyStopping(
    monitor='val_accuracy',  # Monitoruj dokładność walidacyjną
    patience=3,
    restore_best_weights=True,
    mode='max'
)

class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
class_weight_dict = dict(enumerate(class_weights))


history = model.fit(
    [X_train_input_ids, X_train_attention_mask],
    y_train,
    validation_data=(
        [X_val_input_ids, X_val_attention_mask],
        y_val
    ),
    epochs=6,
    batch_size=8,
    callbacks=[early_stop],
    class_weight=class_weight_dict
)

# Ocena na zbiorze testowym
loss, acc = model.evaluate(
    [X_test_input_ids, X_test_attention_mask],
    y_test
)
print(f"Test accuracy: {acc:.4f}")

import pickle

model.save("models/"+lang+"_bert.h5")

with open("models/"+lang+"_bert_tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

with open("models/"+lang+"_bert_labels.pkl", "wb") as f:
    pickle.dump(le, f)