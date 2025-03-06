import tensorflow_text as text

from keras import backend as K

import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd

import keras
from keras.layers import Layer
from keras.callbacks import EarlyStopping

import numpy as np

from keras.metrics import AUC
from keras.metrics import Precision
from keras.metrics import Recall

import string
import kagglehub
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')


# Load BERT model and preprocessing layers
bert_preprocess = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")

#bert_encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4")
#bert_encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/2")  # GOOD
bert_encoder = hub.KerasLayer(
    "https://www.kaggle.com/models/tensorflow/bert/TensorFlow2/bert-en-uncased-l-2-h-128-a-2/2",
    trainable=True)
#bert_encoder.trainable = True


# Training parameters
batch_size = 8
num_negatives = 1
alpha = 1
epochs = 20

# Define paths
train = pd.read_csv("../data/with_ids_fold_5_train.csv", encoding="latin-1", sep=";", index_col=False)
test = pd.read_csv("../data/with_ids_fold_5_test.csv", encoding="latin-1", sep=";", index_col=False)
#train = pd.read_csv("../data/fold_1_train_mini.csv", encoding="latin-1", sep=";", index_col=False)
#test = pd.read_csv("../data/fold_1_test_mini.csv", encoding="latin-1", sep=";", index_col=False)

# Define the input string
#train_element = "The user " + train["user"].squeeze() + " used the context " + train["context"] + " described by " + train["context_nl"].squeeze() + " with the item " + train["item"]  +" described by " + train["item_nl"].squeeze()
#test_element = "The user " + test["user"].squeeze() + " used the context "  + test["context"] + " described by " + test["context_nl"].squeeze() + " with the item " + test["item"]  +" described by " + test["item_nl"].squeeze()

train_element = "The user " + train["user"] + " used the context " + train["context"] + " with the item " + train["item"]
test_element = "The user " + test["user"] + " used the context "  + test["context"] + " with the item " + test["item"]  

# Data preprocessing
train_element_lower = train_element.str.lower()
test_element_lower = test_element.str.lower()



train_element_no_punct = train_element_lower.str.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
test_element_no_punct = test_element_lower.str.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
#train_element_no_punct = train_element_lower.str.translate(str.maketrans('', '', string.punctuation))
#test_element_no_punct = test_element_lower.str.translate(str.maketrans('', '', string.punctuation))

print("train element")
print(train_element_no_punct[0])


# to np array
train_nparr = np.array(train_element_no_punct)
test_nparr = np.array(test_element_no_punct)

print("test nparr")
print(test_nparr[0])

def data_generator(data_df, batch_size, num_negatives, alpha):
    data = np.array(data_df['formatted_text'])  # Extract precomputed formatted sentences
    while True:
        idx = np.random.choice(len(data_df), size=batch_size, replace=False)
        
        # Positive samples
        pos_data = data[idx]
        pos_labels = np.ones(len(pos_data))
        pos_weights = np.ones(len(pos_data))
        
        # Negative samples
        neg_data = []
        for _ in range(batch_size * num_negatives):
            row_idx = np.random.randint(len(data_df))
            neg_idx = np.random.randint(len(data_df))
            
            # Ensure a different row is chosen for negative example
            while neg_idx == row_idx:
                neg_idx = np.random.randint(len(data_df))
            
            neg_sample = ("The user " + str(data_df.iloc[row_idx]['user']) +
                          " used the context " + str(data_df.iloc[row_idx]['context']) +
                          " with the item " + str(data_df.iloc[neg_idx]['item']))
            '''

            neg_sample = ("The user " + str(data_df.iloc[row_idx]['user']) +
                          " used the context " + str(data_df.iloc[row_idx]['context']) +
                          " described by " + str(data_df.iloc[row_idx]['context_nl']) +
                          " with the item " + str(data_df.iloc[neg_idx]['item']) +
                          " described by " + str(data_df.iloc[neg_idx]['item_nl']))
            '''

            neg_data.append(neg_sample.lower().translate(str.maketrans('', '', string.punctuation)))
        
        neg_data = np.array(neg_data)
        neg_labels = np.zeros(len(neg_data))
        neg_weights = np.full(len(neg_labels), alpha)

        '''
        #Debug 
        print("A POSITIVE SAMPLE")
        print(pos_data[0])
        print(pos_labels[0])
        print(pos_weights[0])

        print("A NEGATIVE SAMPLE")
        print(neg_data[0])
        print(neg_labels[0])
        print(neg_weights[0])
        '''
        
        # Combine positive and negative samples
        combined_data = np.concatenate([pos_data, neg_data])
        combined_labels = np.concatenate([pos_labels, neg_labels])
        combined_weights = np.concatenate([pos_weights, neg_weights])
        
        # Shuffle
        indices = np.arange(len(combined_labels))
        np.random.shuffle(indices)
        
        yield (
            tf.convert_to_tensor(combined_data[indices], dtype=tf.string),
            tf.convert_to_tensor(combined_labels[indices], dtype=tf.float32),
            tf.convert_to_tensor(combined_weights[indices], dtype=tf.float32)
        )

# Precompute formatted text
train['formatted_text'] = ("The user " + train['user'].fillna('Unknown') +
                           " used the context " + train['context'].fillna('Unknown') +
                           " with the item " + train['item'].fillna('Unknown')).str.lower().str.translate(str.maketrans('', '', string.punctuation))

test['formatted_text'] = ("The user " + test['user'].fillna('Unknown') +
                          " used the context " + test['context'].fillna('Unknown') +
                          " with the item " + test['item'].fillna('Unknown')).str.lower().str.translate(str.maketrans('', '', string.punctuation))


# Convert to numpy array
train_nparr = np.array(train['formatted_text'])
test_nparr = np.array(test['formatted_text'])

# Create data generators
train_gen = data_generator(train, batch_size, num_negatives, alpha)
val_gen = data_generator(test, batch_size, num_negatives, alpha)




# Define BERT model
text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
preprocessed_text = bert_preprocess(text_input)
outputs = bert_encoder(preprocessed_text)

net = tf.keras.layers.Dropout(0.2, name="dropout")(outputs['pooled_output'])
net = tf.keras.layers.Dense(256, activation="relu", name="hidden0")(net)
net = tf.keras.layers.Dense(64, activation="relu", name="hidden1")(net)
net = tf.keras.layers.Dense(1, activation='sigmoid', name="output")(net)

model = tf.keras.Model(inputs=[text_input], outputs=[net])

# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
    loss='binary_crossentropy',
    metrics=['accuracy', 'AUC', 'Precision', 'Recall']
)




# Train model
early_stopping = EarlyStopping(monitor='val_loss', patience=2)
history = model.fit(
    train_gen,
    validation_data=val_gen,
    steps_per_epoch=len(train_nparr) // batch_size,
    validation_steps=len(test_nparr) // batch_size,
    epochs=epochs,
    callbacks=[early_stopping]
)


# Convert the history to a pandas DataFrame
history_df = pd.DataFrame(history.history)

# Save the history to a CSV file
history_df.to_csv('ID_bert_split_5.csv', index=False)
