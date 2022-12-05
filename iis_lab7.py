import keras
import numpy as np
from keras.models import load_model
from keras.layers import LSTM, Dropout, Dense
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

with open('d/Dostoevskiy', encoding='utf-8', mode='r') as file:
    text = file.read().lower()

chars = sorted(list(set(text)))
# print('Total chars:', len(chars))
# char_indices = dict((c, i) for i, c in enumerate(chars))
# indices_char = dict((i, c) for i, c in enumerate(chars))

# chars = 72
char_indices = {'\t': 0, '\n': 1, ' ': 2, '!': 3, '(': 4, ')': 5, ',': 6, '-': 7, '.': 8, '0': 9, '1': 10, '2': 11, '3': 12, '4': 13, '5': 14, '6': 15, '7': 16, '8': 17, '9': 18, ':': 19, ';': 20, '?': 21, '[': 22, ']': 23, 'c': 24, 'h': 25, 'i': 26, 'v': 27, 'x': 28, 'y': 29, '\xa0': 30, '«': 31, '»': 32, 'а': 33, 'б': 34, 'в': 35, 'г': 36, 'д': 37, 'е': 38, 'ж': 39, 'з': 40, 'и': 41, 'й': 42, 'к': 43, 'л': 44, 'м': 45, 'н': 46, 'о': 47, 'п': 48, 'р': 49, 'с': 50, 'т': 51, 'у': 52, 'ф': 53, 'х': 54, 'ц': 55, 'ч': 56, 'ш': 57, 'щ': 58, 'ъ': 59, 'ы': 60, 'ь': 61, 'э': 62, 'ю': 63, 'я': 64, 'ё': 65, '–': 66, '—': 67, '“': 68, '„': 69, '…': 70, '№': 71}
indices_char = {0: '\t', 1: '\n', 2: ' ', 3: '!', 4: '(', 5: ')', 6: ',', 7: '-', 8: '.', 9: '0', 10: '1', 11: '2', 12: '3', 13: '4', 14: '5', 15: '6', 16: '7', 17: '8', 18: '9', 19: ':', 20: ';', 21: '?', 22: '[', 23: ']', 24: 'c', 25: 'h', 26: 'i', 27: 'v', 28: 'x', 29: 'y', 30: '\xa0', 31: '«', 32: '»', 33: 'а', 34: 'б', 35: 'в', 36: 'г', 37: 'д', 38: 'е', 39: 'ж', 40: 'з', 41: 'и', 42: 'й', 43: 'к', 44: 'л', 45: 'м', 46: 'н', 47: 'о', 48: 'п', 49: 'р', 50: 'с', 51: 'т', 52: 'у', 53: 'ф', 54: 'х', 55: 'ц', 56: 'ч', 57: 'ш', 58: 'щ', 59: 'ъ', 60: 'ы', 61: 'ь', 62: 'э', 63: 'ю', 64: 'я', 65: 'ё', 66: '–', 67: '—', 68: '“', 69: '„', 70: '…', 71: '№'}

print(char_indices)
print(indices_char)

maxlen = 32
step = 3
sequences = []
next_sequence_values = []
for i in range(0, len(text) - maxlen, step):
    sequences.append(text[i: i + maxlen])
    next_sequence_values.append(text[i + maxlen])
print('Number of sequences:', len(sequences))

X = np.zeros((len(sequences), maxlen, len(chars)), dtype=np.bool_)
y = np.zeros((len(sequences), len(chars)), dtype=np.bool_)

for i, sentence in enumerate(sequences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_sequence_values[i]]] = 1


# model = keras.Sequential()
# model.add(LSTM(128, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
# model.add(Dropout(0.2))
# model.add(LSTM(128))
# model.add(Dropout(0.2))
# model.add(Dense(y.shape[1], activation='softmax'))
# model.compile(loss='categorical_crossentropy', optimizer='adam')

model = load_model('my_model.h5')

epochs = 10
batch_size = 128

for epoch in range(epochs):
    model.fit(X, y, batch_size=batch_size, epochs=1)
    print(epoch, 'is finished')

model.save('my_model2.h5')
