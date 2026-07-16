import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

MAX_VOCAB =10000
MAX_LEN =100
tokenizer = Tokenizer(num_words=MAX_VOCAB, oov_token="<OOV>")
tokenizer.fit_on_texts(train_texts)
x_train = pad_sequences(tokenizer.texts_to_sequences(train_texts), maxlen=MAX_LEN, padding="post")
x_test = pad_sequences(tokenizer.texts_to_sequences(test_text),maxlen = MAX_LEN, padding="post")

model = Sequential([
    Embedding(MAX_VOCAB, 128, input_length=MAX_LEN),
    LSTM(64, return_sequences=True),
    LSTM(32),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dense(3, activation="softmax")
])

model.compile(optimizer="adam",loss="sparse_categaorical_crossentropy", metrics=["accuarcy"])
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=32)
model.save("lstm_sentiment.h5")

import numpy as np
LABELS = ['negative', 'netural', 'postive']
def lstm_predict(text:str)->dict:
  seq = tokenizer.text_to_sequences(seq,maxlen = MAX_LEN, padding ='post')
  probs = model.predict(padded, varbose = 0)[0]
  return {
      "lebel": LABELS[np.argmax(probs)],
      "scores": {LABELS[i]: float(probs[i])for i in range(3)}
  }

def fuse_sentiment(text:str, roberta_fn, w_roberta=0.7, w_lstm=0.3):
  r = roberta_fn(text)
  l = lstm_fn(text)['scores']
  fused ={
      k: w_roberta *r[k]+w_lstm *l[k]
      for k in['negative', 'neturan', 'postive']
  }
  return max(fused, key=fused.get), fused
