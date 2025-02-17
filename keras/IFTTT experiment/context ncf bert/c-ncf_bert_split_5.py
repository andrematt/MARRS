import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense, Flatten, Multiply, Concatenate
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model

import tensorflow_hub as hub
import tensorflow_text as text
import string
##################### C-NCF BERT


# Hyperparameters
embedding_dim = 8  # Embedding size
batch_size = 8
num_negatives = 1
alpha = 1
epochs = 20
learning_rate=2e-5

# Load BERT model and preprocessing layers
bert_preprocess = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
bert_encoder = hub.KerasLayer(
    "https://www.kaggle.com/models/tensorflow/bert/TensorFlow2/bert-en-uncased-l-2-h-128-a-2/2",
    trainable=True)



# Load data from CSV
train = pd.read_csv("../data/with_ids_fold_5_train.csv", encoding="latin-1", sep=";", index_col=False)
test = pd.read_csv("../data/with_ids_fold_5_test.csv", encoding="latin-1", sep=";", index_col=False)



################### DATA PREPARATION 



# Combine train and test datasets for encoding
combined_users = pd.concat([train["user"], test["user"]], axis=0).astype("category")
combined_items = pd.concat([train["item"], test["item"]], axis=0).astype("category")
combined_contexts = pd.concat([train["context"], test["context"]], axis=0).astype("category")


train_element = "The user " + train["user"] + " used the context " + train["context"] + " with the item " + train["item"]
test_element = "The user " + test["user"] + " used the context "  + test["context"] + " with the item " + test["item"]  

# Data preprocessing
train_element_lower = train_element.str.lower()
test_element_lower = test_element.str.lower()

train_element_no_punct = train_element_lower.str.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
test_element_no_punct = test_element_lower.str.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))

# to np array
train_nparr = np.array(train_element_no_punct)
test_nparr = np.array(test_element_no_punct)


# Encode users and items
train["user_encoded"] = combined_users[:len(train)].cat.codes
test["user_encoded"] = combined_users[len(train):].cat.codes

train["item_encoded"] = combined_items[:len(train)].cat.codes
test["item_encoded"] = combined_items[len(train):].cat.codes

train["context_encoded"] = combined_contexts[:len(train)].cat.codes
test["context_encoded"] = combined_contexts[len(train):].cat.codes

# Convert encoded columns to numpy arrays
users = train["user_encoded"].to_numpy()
items = train["item_encoded"].to_numpy()
contexts = train["context_encoded"].to_numpy()

val_users = test["user_encoded"].to_numpy()
val_items = test["item_encoded"].to_numpy()
val_contexts = test["context_encoded"].to_numpy()

# Update the number of unique users and items
num_users = len(combined_users.cat.categories)
num_items = len(combined_items.cat.categories)
num_contexts = len(combined_contexts.cat.categories)

# Data generator for training

def data_generator(data_df, users, items, contexts, batch_size, num_negatives, alpha):
    data = np.array(data_df['formatted_text'])  # Extract precomputed formatted sentences
    while True:
        # Positive samples
        idx = np.random.choice(len(data_df), size=batch_size, replace=False)
        
        text_batch = data[idx]
        user_batch = users[idx]
        item_batch = items[idx]
        context_batch = contexts[idx]

        labels = np.ones(len(user_batch))
        weights = np.full(len(user_batch), 1)  # Weight for positive samples
        
        '''
        
        print("POSITIVE BATCH")
        print("text")
        print(text_batch)
        print("user")
        print(user_batch)
        print("item")
        print(item_batch)   
        print("context")
        print(context_batch)
        print("labels")
        print(labels)
        print("---------------------------------")
        '''   
        # Negative samples
        neg_data = []
        neg_user_batch = []
        neg_item_batch = []
        neg_context_batch = []
        for _ in range(batch_size * num_negatives):
            row_idx = np.random.randint(len(data_df))
            neg_idx = np.random.randint(len(data_df))
            
            # Ensure a different row is chosen for negative example
            while neg_idx == row_idx:
                neg_idx = np.random.randint(len(data_df))
            
            neg_sample = ("The user " + str(data_df.iloc[row_idx]['user']) +
                          " used the context " + str(data_df.iloc[row_idx]['context']) +
                          " with the item " + str(data_df.iloc[neg_idx]['item']))
                          

            neg_data.append(neg_sample.lower().translate(str.maketrans('', '', string.punctuation)))         
            neg_user_batch.append(users[row_idx])
            neg_item_batch.append(items[row_idx])
            neg_context_batch.append(contexts[neg_idx])

        neg_text_batch = np.array(neg_data)
        neg_user_batch = np.array(neg_user_batch)
        neg_item_batch = np.array(neg_item_batch)
        neg_context_batch = np.array(neg_context_batch)

        neg_labels = np.zeros(len(neg_data))
        neg_weights = np.full(len(neg_labels), alpha)  # Weight for negative samples

        '''
        print("NEGATIVE BATCH")
        print("text")
        print(neg_text_batch)
        print("user")
        print(neg_user_batch)
        print("item")
        print(neg_item_batch)
        print("context")
        print(neg_context_batch)
        print("labels")
        print(neg_labels)
        print("---------------------------------")
        '''

        # Combine positive and negative samples
        combined_data = np.concatenate([text_batch, neg_text_batch])
        combined_user_batch = np.concatenate([user_batch, neg_user_batch])
        combined_item_batch = np.concatenate([item_batch, neg_item_batch])
        combined_context_batch = np.concatenate([context_batch, neg_context_batch])
        combined_labels = np.concatenate([labels, neg_labels])
        combined_weights = np.concatenate([weights, neg_weights])

        '''
        print("COMBINED BATCH")
        print("text")
        print(combined_data)
        print("user")
        print(combined_user_batch)
        print("item")
        print(combined_item_batch)
        print("context")
        print(combined_context_batch)
        print("labels")
        print(combined_labels)
        print("---------------------------------")
        '''

        # Shuffle
        indices = np.arange(len(combined_labels))
        np.random.shuffle(indices)

        '''
        print("SHUFFLED BATCH")
        print("text")
        print(combined_data[indices])
        print("user")
        print(combined_user_batch[indices])
        print("item")
        print(combined_item_batch[indices])
        print("context")
        print(combined_context_batch[indices])
        print("labels")
        print(combined_labels[indices])
        print("---------------------------------")
        '''

        # Yield data as a tuple of tensors with specified types
        yield (
            (
            tf.convert_to_tensor(combined_user_batch[indices], dtype=tf.int32),
            tf.convert_to_tensor(combined_item_batch[indices], dtype=tf.int32),
            tf.convert_to_tensor(combined_context_batch[indices], dtype=tf.int32),
            tf.convert_to_tensor(combined_data[indices], dtype=tf.string)),
            tf.convert_to_tensor(combined_labels[indices], dtype=tf.float32),
            tf.convert_to_tensor(combined_weights[indices], dtype=tf.float32),
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


# Train generator
train_gen = data_generator(train, users, items, contexts, batch_size, num_negatives, alpha)

# Validation generator
val_gen = data_generator(test, val_users, val_items, val_contexts, batch_size,num_negatives, alpha)



############### MODEL DEFINITION


# Neural Collaborative Filtering Model
user_input = Input(shape=(1,), name="user_input")
item_input = Input(shape=(1,), name="item_input")
context_input = Input(shape=(1,), name="context_input")

# Embedding layers
user_mf_embedding = Flatten()(Embedding(num_users, embedding_dim, name="user_mf_embedding")(user_input))
item_mf_embedding = Flatten()(Embedding(num_items, embedding_dim, name="item_mf_embedding")(item_input))
context_mf_embedding = Flatten()(Embedding(num_contexts, embedding_dim, name="context_mf_embedding")(context_input))

user_mlp_embedding = Flatten()(Embedding(num_users, embedding_dim, name="user_mlp_embedding")(user_input))
item_mlp_embedding = Flatten()(Embedding(num_items, embedding_dim, name="item_mlp_embedding")(item_input))
context_mlp_embedding = Flatten()(Embedding(num_contexts, embedding_dim, name="context_mlp_embedding")(context_input))

# Matrix Factorization
mf_vector = Multiply()([user_mf_embedding, item_mf_embedding, context_mf_embedding])

# Multi-Layer Perceptron
mlp_vector = Concatenate()([user_mlp_embedding, item_mlp_embedding, context_mlp_embedding])
mlp_layer1 = Dense(64, activation="relu")(mlp_vector)
mlp_layer2 = Dense(32, activation="relu")(mlp_layer1)
mlp_layer3 = Dense(16, activation="relu")(mlp_layer2)

# Combine MF and MLP parts
#combined_vector = Concatenate()([mf_vector, mlp_layer3])



# Define BERT model
text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
preprocessed_text = bert_preprocess(text_input)
outputs = bert_encoder(preprocessed_text)

bert_net = tf.keras.layers.Dropout(0.2, name="dropout")(outputs['pooled_output'])
bert_net = tf.keras.layers.Dense(256, activation="relu", name="hidden0")(bert_net)
bert_net = tf.keras.layers.Dense(64, activation="relu", name="hidden1")(bert_net)
#bert_net = tf.keras.layers.Dense(1, activation='sigmoid', name="output")(bert_net)


# We merge the two networks together
merged_vector = tf.keras.layers.concatenate([mf_vector, mlp_layer3, bert_net]) 

output = Dense(1, activation="sigmoid", name="output")(merged_vector)



# Build the model
#model = Model(inputs=[(user_input,item_input, context_input,[text_input])], outputs=[output])
model = Model(inputs=[user_input,item_input, context_input,text_input], outputs=[output])
optimizer = Adam(learning_rate=learning_rate)
#model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=['accuracy'])
model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5), metrics=['accuracy', 'AUC', 'Precision', 'Recall'])
# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=2)



# Steps for validation
validation_steps = len(val_users) // batch_size

# Training with data generator
steps_per_epoch = len(users) // batch_size #

#plot_model(model, to_file='c-ncf.png')
# Train the model

history = model.fit(
    train_gen,
    validation_data=val_gen,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    epochs=40,
    verbose=1,
    callbacks=[early_stopping],
)

# Save the history
history_df = pd.DataFrame(history.history)
history_df.to_csv("c-ncf_BERT_split_5.csv", index=False)