import os
import pandas as pd
#import matplotlib.pyplot as plt
import numpy as np
#from typing import Dict, Text

import tensorflow as tf


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

path = "C:/Users/andre/Programmazione/Python/assessment_ifttt_KERAS_PYTHON_89/OUR DATASET/data/1 neg 1 pos"

df_train=pd.read_csv(path + "/fold_1_train.csv", encoding="latin-1", sep=",", index_col=False)
df_train=pd.read_csv(path + "/fold_1_train.csv", encoding="latin-1", sep=",", index_col=False)
df_train.head()

df_test=pd.read_csv(path + "/fold_1_test.csv", encoding="latin-1", index_col=False)
df_test.head()

embedding_dimension = 8
print(df_train.dtypes)

df_train['colonna2'] = df_train['context'].str.split(';')

unique_user_names = df_train["user"].unique() # rinominali unique_user_names_train se rinserisci il test set separato come prima
print(unique_user_names)


unique_item_names = []
for i in df_train['item']:  # non dovrebbe servire
  if i not in unique_item_names:
    unique_item_names.append(i)
print(len(unique_item_names))

print(unique_item_names)

one_hot_dimensions = len(unique_item_names)  # in realtà sarebbe binary encoding dimensions

user_input = tf.keras.layers.Input(dtype=tf.string, shape=(1,))
other_item_input = tf.keras.layers.Input(dtype=tf.string, shape=(1,))

user_mf_1 = StringLookup(vocabulary=unique_user_names)(user_input)
user_mf_2 = Embedding(len(unique_user_names)+1, embedding_dimension*one_hot_dimensions)(user_mf_1)
user_mf_3 = Flatten(name="flatten-user_mf")(user_mf_2)

other_item_mf_1 = StringLookup(vocabulary=unique_item_names)(other_item_input)
other_item_mf_2 = Embedding(len(unique_item_names)+1, embedding_dimension*one_hot_dimensions)(other_item_mf_1)
other_item_mf_3 = Flatten(name="other_flatten-item_mf")(other_item_mf_2)

user_mlp_1 = StringLookup(vocabulary=unique_user_names)(user_input)
user_mlp_2 = Embedding(len(unique_user_names)+1, embedding_dimension*one_hot_dimensions)(user_mlp_1)
user_mlp_3 = Flatten(name="flatten-user_mlp")(user_mlp_2)

other_item_mlp_1 = StringLookup(vocabulary=unique_item_names)(other_item_input)
other_item_mlp_2 = Embedding(len(unique_item_names)+1, embedding_dimension*one_hot_dimensions)(other_item_mlp_1)
other_item_mlp_3 = Flatten(name="other_flatten-item_mlp")(other_item_mlp_2)

# concatenate features for MLP
#conc = concatenate([user_mlp_3, other_item_mlp_3])

# multiply features for CF
dot = tf.multiply(user_mf_3, other_item_mf_3)
#dot = tf.multiply(dot, other_item_mf_3)

# add fully-connected-layers
'''
fc1 = Dense(64, activation='relu')(conc)
fc2 = Dense(32, activation='relu')(fc1)
fc3 = Dense(16, activation='relu')(fc2)

'''
# We merge the two networks together
#merged_vector = tf.keras.layers.concatenate([dot, fc3])

# 1 neuron output layer
output_layer = Dense(1, activation='sigmoid')(dot)

# Create model and compile it
model = Model(inputs=[(user_input, other_item_input)],outputs=[output_layer])

model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), metrics=['accuracy', 'AUC', 'Precision', 'Recall'])

users_train = df_train["user"].to_numpy()
#items_train = np.array(my_rows_train)
other_items_train = df_train["item"].to_numpy()
labels_train = df_train["label"].to_numpy()
print(model.summary())

users_test = df_test["user"].to_numpy()
#items_train = np.array(my_rows_train)
other_items_test = df_test["item"].to_numpy()
labels_test = df_test["label"].to_numpy()

early_stopping =EarlyStopping(monitor='val_loss', patience=2)

history = model.fit([users_train, other_items_train], labels_train, validation_data=[[users_test, other_items_test], labels_test], epochs=40, batch_size=8, verbose=1, callbacks=[early_stopping])

# Convert the history to a pandas DataFrame
history_df = pd.DataFrame(history.history)

# Save the history to a CSV file
history_df.to_csv('cf_split_1.csv', index=False)
