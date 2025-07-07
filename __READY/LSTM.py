import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from keras.callbacks import EarlyStopping, ModelCheckpoint
import pickle


# lang, lang_long = "de", "german"
lang, lang_long = "en", "english"

df = pd.read_csv(lang+'-3-classes.csv')

import re
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')

def preprocess_text(text, lang="english"):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    stop_words = set(stopwords.words(lang))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text




# 2. Łączenie kolumn subject i body w jeden tekst
df['text'] = df['subject'].fillna('') + ' ' + df['body'].fillna('')
print(df['text'])
print(type(df['text']))
df['text'] = df['text'].astype(str).apply(lambda x: preprocess_text(x, lang=lang_long))


# 3. Usuwamy wiersze z brakującym typem
df = df.dropna(subset=['type'])

# 4. Tokenizacja tekstu
max_words = 10000
max_len = 1000

tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(df['text'])
sequences = tokenizer.texts_to_sequences(df['text'])
padded = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')

# 5. Kodowanie etykiet
le = LabelEncoder()
labels = le.fit_transform(df['type'])
labels_cat = to_categorical(labels)

# 6. Podział na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(padded, labels_cat, test_size=0.15, random_state=42)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=4,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    filepath="models/best_" + lang + "_lstm.h5",
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=False
)

# 7. Budowa modelu LSTM
model = Sequential()
model.add(Embedding(max_words, 256, input_length=max_len))
model.add(Bidirectional(LSTM(128, return_sequences=False)))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dense(labels_cat.shape[1], activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 8. Trening
history = model.fit(
    X_train, y_train, validation_split=0.2,
    epochs=50,
    batch_size=64,
    verbose=1,
    callbacks=[early_stop, checkpoint]
    )

# 9. Ewaluacja
loss, acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {acc:.2f}")



model.save("models/"+lang+"_lstm.h5")

with open("models/"+lang+"_lstm_tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

with open("models/"+lang+"_lstm_labels.pkl", "wb") as f:
    pickle.dump(le, f)