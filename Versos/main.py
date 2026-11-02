import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Activation
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import load_model

filepath = tf.keras.utils.get_file(
    'claricelispector.txt',
    'https://raw.githubusercontent.com/anajulia-alves/versos-de-silicio/refs/heads/main/claricelispector.txt'
)

text = open(filepath, 'rb').read().decode(encoding='utf-8').lower()

characters = sorted(set(text))
char_to_index = {c: i for i, c in enumerate(characters)}
index_to_char = {i: c for i, c in enumerate(characters)}

SEQ_LENGTH = 40
STEP_SIZE = 1

sentences = []
next_characters = []

for i in range(0, len(text) - SEQ_LENGTH, STEP_SIZE):
    sentences.append(text[i:i + SEQ_LENGTH])
    next_characters.append(text[i + SEQ_LENGTH])

x = np.zeros((len(sentences), SEQ_LENGTH, len(characters)), dtype=bool)
y = np.zeros((len(sentences), len(characters)), dtype=bool)

for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_to_index[char]] = 1
    y[i, char_to_index[next_characters[i]]] = 1

model = Sequential()
model.add(Input(shape=(SEQ_LENGTH, len(characters))))
model.add(LSTM(128))
model.add(Dense(len(characters)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer=RMSprop(learning_rate=0.001))

model.fit(x, y, batch_size=16, epochs=300)

model.save('textgenerator.keras')


def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds + 1e-8) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def generate_text(length, temperature):
    start_index = random.randint(0, len(text) - SEQ_LENGTH - 1)
    sentence = text[start_index: start_index + SEQ_LENGTH]
    generated_text = sentence

    for _ in range(length):
        x = np.zeros((1, SEQ_LENGTH, len(characters)))
        for t, character in enumerate(sentence):
            x[0, t, char_to_index[character]] = 1

        predictions = model(x, training=False).numpy()[0]
        next_index = sample(predictions, temperature)
        next_character = characters
        next_character = index_to_char[int(next_index)]

        generated_text += next_character
        sentence = sentence[1:] + next_character

    return generated_text


for temp in (0.2, 0.4, 0.6, 0.8):
    print(f'--------{temp}---------')
    print(generate_text(300, temp))