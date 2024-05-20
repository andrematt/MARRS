import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Text

import tensorflow as tf
import tensorflow_text as text
import tensorflow_hub as hub
import tensorflow.keras

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from keras.metrics import top_k_categorical_accuracy
from keras.metrics import AUC
from keras.metrics import Precision
from keras.metrics import Recall

from keras.callbacks import EarlyStopping

from tensorflow.nn import  softmax_cross_entropy_with_logits
from tensorflow.nn import  sparse_softmax_cross_entropy_with_logits
from tensorflow.nn import  sigmoid_cross_entropy_with_logits

from tensorflow.keras.layers import Input, Dense, concatenate, StringLookup, Embedding, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model

import keras
from keras import backend as K

bert_preprocess = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
bert_encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4")
bert_encoder.trainable = True

path = "path/data/1 neg 0 pos"

df_train=pd.read_csv(path + "/fold_4_train.csv", encoding="latin-1", sep=",", index_col=False)
df_train.head()

df_test=pd.read_csv(path + "/fold_4_test.csv", encoding="latin-1", sep=",", index_col=False)
df_test.head()

embedding_dimension = 8

df_train['colonna2'] = df_train['context'].str.split(';')
df_test['colonna2'] = df_test['context'].str.split(';')

unique_user_names_train = df_train["user"].unique() # rinominali unique_user_names_train se rinserisci il test set separato come prima
unique_user_names_test = df_test["user"].unique()
unique_user_names = np.concatenate((unique_user_names_train, unique_user_names_test))
unique_user_names = np.unique(unique_user_names)

train_element = "The user " +df_train["user"].squeeze() + " used the context " + df_train["context_nl"].squeeze() + " with the item " + df_train["item_nl"].squeeze()
test_element = "The user " + df_test["user"].squeeze() + " used the context " + df_test["context_nl"].squeeze() + " with the item " + df_test["item_nl"].squeeze()


unique_item_names = []
for i in df_train['colonna2']:
    for y in i:
      if y not in unique_item_names:
        unique_item_names.append(y)
for i in df_test['colonna2']:
    for y in i:
      if y not in unique_item_names:
        unique_item_names.append(y)


for i in df_train['item']:  # non dovrebbe servire
  if i not in unique_item_names:
    unique_item_names.append(y)

set_res = set(unique_item_names)
unique_item_names = (list(set_res))

one_hot_dimensions = len(unique_item_names)

my_rows_train = []
for elements_row in df_train['colonna2']:
  new_row = []
  for element in unique_item_names:
    if element in elements_row:
      new_row.append(1)
    else:
      new_row.append(0)
  my_rows_train.append(new_row)
print(len(my_rows_train))

my_rows_test = []
for elements_row in df_test['colonna2']:
  new_row = []
  for element in unique_item_names:
    if element in elements_row:
      new_row.append(1)
    else:
      new_row.append(0)
  my_rows_test.append(new_row)
  
users_train = df_train["user"].to_numpy()
items_train = np.array(my_rows_train)
other_items_train = df_train["item"].to_numpy()

labels_train = df_train["label"].to_numpy()

targets = labels_train

inputs = [users_train, items_train, other_items_train]

num_samples = len(labels_train)

ID_Inp = np.array(range(num_samples))
ID_Out = np.array(range(num_samples))

my_rows_test = []
for elements_row in df_test['colonna2']:
  new_row = []
  for element in unique_item_names:
    if element in elements_row:
      new_row.append(1)
    else:
      new_row.append(0)
  my_rows_test.append(new_row)
print(len(my_rows_test))

df_test["one_hot"] = my_rows_test 

users_test = df_test["user"].to_numpy()
items_test = np.array(my_rows_test)
other_items_test = df_test["item"].to_numpy()
labels_test = df_test["label"].to_numpy()

# Bert layers
text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
preprocessed_text = bert_preprocess(text_input)
outputs = bert_encoder(preprocessed_text)

# Neural network layers
bert_net = tf.keras.layers.Dropout(0.2, name="dropout")(outputs['pooled_output'])
bert_net = tf.keras.layers.Dense(64, activation="relu", name="hidden1")(bert_net)

user_input = tf.keras.layers.Input(dtype=tf.string, shape=(1,))
item_input = tf.keras.layers.Input(dtype=tf.int32, shape=(one_hot_dimensions,), name="one_hot_input")
other_item_input = tf.keras.layers.Input(dtype=tf.string, shape=(1,))

user_mf_1 = StringLookup(vocabulary=unique_user_names)(user_input)
user_mf_2 = Embedding(len(unique_user_names)+1, embedding_dimension*one_hot_dimensions)(user_mf_1)
user_mf_3 = Flatten(name="flatten-user_mf")(user_mf_2)

#item_mf_1 = StringLookup(vocabulary=unique_item_names, name="one_hot_encoding")(item_input)
item_mf_2 = Embedding(len(unique_item_names), embedding_dimension, name="one_hot_embedding_mf")(item_input)
item_mf_3 = Flatten(name="flatten-item_mf")(item_mf_2)

other_item_mf_1 = StringLookup(vocabulary=unique_item_names)(other_item_input)
other_item_mf_2 = Embedding(len(unique_item_names)+1, embedding_dimension*one_hot_dimensions)(other_item_mf_1)
other_item_mf_3 = Flatten(name="other_flatten-item_mf")(other_item_mf_2)

user_mlp_1 = StringLookup(vocabulary=unique_user_names)(user_input)
user_mlp_2 = Embedding(len(unique_user_names)+1, embedding_dimension*one_hot_dimensions)(user_mlp_1)
user_mlp_3 = Flatten(name="flatten-user_mlp")(user_mlp_2)

#item_mlp_1 = StringLookup(vocabulary=unique_item_names)(item_input)
item_mlp_2 = Embedding(len(unique_item_names), embedding_dimension, name="one_hot_embedding_mlp")(item_input)
item_mlp_3 = Flatten(name="flatten-item_mlp")(item_mlp_2)

other_item_mlp_1 = StringLookup(vocabulary=unique_item_names)(other_item_input)
other_item_mlp_2 = Embedding(len(unique_item_names)+1, embedding_dimension*one_hot_dimensions)(other_item_mlp_1)
other_item_mlp_3 = Flatten(name="other_flatten-item_mlp")(other_item_mlp_2)

# concatenate features for MLP
conc = concatenate([user_mlp_3, item_mlp_3, other_item_mlp_3])

# multiply features for CF
dot = tf.multiply(user_mf_3, item_mf_3)
dot = tf.multiply(dot, other_item_mf_3)

# add fully-connected-layers
fc1 = Dense(64, activation='relu')(conc)
fc2 = Dense(32, activation='relu')(fc1)
fc3 = Dense(16, activation='relu')(fc2)

# We merge the two networks together
merged_vector = tf.keras.layers.concatenate([dot, fc3, bert_net]) #Remember to take back to later_1

# 1 neuron output layer
output_layer = Dense(1, activation='sigmoid')(merged_vector)

# Create model and compile it
model = Model(inputs=[(user_input,item_input, other_item_input,[text_input])],outputs=[output_layer])
model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5), metrics=['accuracy', 'AUC', 'Precision', 'Recall'])
print(model.summary())

early_stopping =EarlyStopping(monitor='val_loss', patience=2)


history = model.fit([users_train, items_train, other_items_train, train_element], labels_train, validation_data=[[users_test, items_test, other_items_test, test_element], labels_test], batch_size = 8, epochs=40, verbose=1, callbacks=[early_stopping])

# Convert the history to a pandas DataFrame
history_df = pd.DataFrame(history.history)

# Save the history to a CSV file
history_df.to_csv('ncf_context_bert_split_4.csv', index=False)
