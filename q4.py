import collections
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import GRU, Dense, TimeDistributed, Dropout
import pandas as pd

data = pd.read_csv("/Users/cameronwalcott/Desktop/ITP499/Exam 2/english-spanish-dataset.csv")
data = data.drop(["Unnamed: 0"], axis=1)
data = data.iloc[:50000,:]
english_sentences = data["english"]
spanish_sentences = data["spanish"]

def tokenize(x):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(x)
    return tokenizer.texts_to_sequences(x), tokenizer

text_sentences = [
    'The quick brown fox jumps over the lazy dog .',
    'By Jove, my quick study of lexicography won a prize .',
    'This is a short sentence .']

text_tokenized, text_tokenizer = tokenize(text_sentences)

def pad(x, length=None):
    return pad_sequences(x, maxlen=length, padding='post')

test_pad = pad(text_tokenized)

def preprocess(x, y):
    preprocess_x, x_tk = tokenize(x)
    preprocess_y, y_tk = tokenize(y)

    preprocess_x = pad(preprocess_x)
    preprocess_y = pad(preprocess_y)

    preprocess_y = preprocess_y.reshape(*preprocess_y.shape, 1)
    return preprocess_x, preprocess_y, x_tk, y_tk


pp_english_sentences, pp_spanish_sentences, english_tokenizer, \
    spanish_tokenizer = preprocess(english_sentences, spanish_sentences)




max_english_sequence_length = pp_english_sentences.shape[1]
english_vocab_size = len(english_tokenizer.word_index)
max_spanish_sequence_length = pp_spanish_sentences.shape[1]
spanish_vocab_size = len(spanish_tokenizer.word_index)



def logits_to_text(logits, tokenizer):
    index_to_words = {id: word for word, id in tokenizer.word_index.items()}
    index_to_words[0] = '<PAD>'

    return ' '.join([index_to_words[prediction] for prediction in np.argmax(logits, 1)] )

def simple_model(input_shape, output_sequence, english_vocab_size, spanish_vocab_size):
    model = Sequential()
    model.add(GRU(128, input_shape=input_shape[1:], return_sequences=True))
    model.add(Dropout(0.5))
    model.add(GRU(128, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(256, activation='relu')))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(spanish_vocab_size + 1, activation='softmax')))

    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer="adam",
                  metrics=['accuracy'])
    return model

# print(pp_english_sentences.shape)
tmp_x = pad(pp_english_sentences, max_spanish_sequence_length)
# print(tmp_x.shape)
#
# print(pp_spanish_sentences.shape)
tmp_x = tmp_x.reshape((-1, pp_spanish_sentences.shape[-2], 1))
print(tmp_x)

model = simple_model(
            tmp_x.shape,
            max_spanish_sequence_length,
            english_vocab_size,
            spanish_vocab_size)

model.summary()


history = model.fit(tmp_x, pp_spanish_sentences, batch_size=300, epochs=5,
          validation_split=0.2)



plt.figure(1)
plt.title("Train Loss")
plt.plot(history.history['loss'])
plt.ylabel("Loss")
plt.xlabel("epoch")

plt.figure(2)
plt.title("Validation Loss")
plt.ylabel("Loss")
plt.xlabel("epoch")
plt.plot(history.history['val_loss'])

plt.figure(3)
plt.title("Train Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("epoch")
plt.plot(history.history['accuracy'])

plt.figure(4)
plt.title("Validation Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("epoch")
plt.plot(history.history['val_accuracy'])
plt.show()


user = input("Enter a sentence to translate: ")
