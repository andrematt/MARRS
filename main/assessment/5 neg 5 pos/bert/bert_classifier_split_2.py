import tensorflow_text as text

from keras import backend as K

import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import keras
from keras.layers import Layer
from keras.callbacks import EarlyStopping

import numpy as np
import matplotlib.pyplot as plt
from keras.metrics import AUC
from keras.metrics import Precision
from keras.metrics import Recall

import string

bert_preprocess = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
bert_encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4")
bert_encoder.trainable = True

print(bert_encoder.__class__.__name__)

path = "your_path/data/5 neg 5 pos"

train=pd.read_csv(path + "/fold_2_train.csv", encoding="latin-1", sep=",", index_col=False)
train.head()

test=pd.read_csv(path + "/fold_2_test.csv", encoding="latin-1", sep=",", index_col=False)
test.head()

embedding_dimension = 8

train_label=train["label"].squeeze()
test_label=test["label"].squeeze()

train_element = "The user " +train["user"].squeeze() + " used the context " + train["context_nl"].squeeze() + " with the item " + train["item_nl"].squeeze()
test_element = "The user " +test["user"].squeeze() + " used the context " + test["context_nl"].squeeze() + " with the item " + test["item_nl"].squeeze()

# Logging 
train_element_lower = [x.lower() for x in train_element]
test_element_lower = [x.lower() for x in test_element]

print(train_element_lower[0])

train_element_no_punct = [x.translate(str.maketrans('",.;-?!','       ',)) for x in train_element_lower]
test_element_no_punct = [x.translate(str.maketrans('",.;-?!','       ',)) for x in test_element_lower]

print(len(string.punctuation))

print(train_element_no_punct[0])
print(test_element_no_punct[0])

################

# Bert layers
text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
preprocessed_text = bert_preprocess(text_input)
outputs = bert_encoder(preprocessed_text)


# Neural network layers
net = tf.keras.layers.Dropout(0.2, name="dropout")(outputs['pooled_output'])
net = tf.keras.layers.Dense(64, activation="relu", name="hidden1")(net)
net = tf.keras.layers.Dense(1, activation='sigmoid', name="output")(net)

# Use inputs and outputs to construct a final model
model = tf.keras.Model(inputs=[text_input], outputs = [net])

early_stopping =EarlyStopping(monitor='val_loss', patience=2)

model.compile(tf.keras.optimizers.Adam(learning_rate=2e-5),
              loss='binary_crossentropy',
              metrics=['accuracy', 'AUC', 'Precision', 'Recall'])

history = model.fit(train_element, train_label, validation_data=[test_element, test_label], epochs=40, batch_size = 8, callbacks=[early_stopping]) 

# Convert the history to a pandas DataFrame
history_df = pd.DataFrame(history.history)

# Save the history to a CSV file
history_df.to_csv('bert_split_2.csv', index=False)
