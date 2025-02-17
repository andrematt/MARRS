import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense, Flatten, Multiply, Concatenate
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model


##################### CF 


# Hyperparameters
embedding_dim = 8  # Embedding size
learning_rate = 0.01  # Learning rate
m = 5  # Number of negative samples per positive example
batch_size = 256  # Batch size
alpha = 0.2  # Weight for negative examples (1 for positive examples)

# Load data from CSV
df_train = pd.read_csv("./data/fold_5_train.csv", encoding="latin-1", sep=";", index_col=False)
df_test = pd.read_csv("./data/fold_5_test.csv", encoding="latin-1", sep=";", index_col=False)

# Combine train and test datasets for encoding
combined_users = pd.concat([df_train["user"], df_test["user"]], axis=0).astype("category")
combined_items = pd.concat([df_train["item"], df_test["item"]], axis=0).astype("category")

# Encode users and items
df_train["user_encoded"] = combined_users[:len(df_train)].cat.codes
df_test["user_encoded"] = combined_users[len(df_train):].cat.codes

df_train["item_encoded"] = combined_items[:len(df_train)].cat.codes
df_test["item_encoded"] = combined_items[len(df_train):].cat.codes

# Convert encoded columns to numpy arrays
users = df_train["user_encoded"].to_numpy()
items = df_train["item_encoded"].to_numpy()

val_users = df_test["user_encoded"].to_numpy()
val_items = df_test["item_encoded"].to_numpy()

# Update the number of unique users and items
num_users = len(combined_users.cat.categories)
num_items = len(combined_items.cat.categories)

# Data generator for training
def data_generator(users, items, num_items, batch_size, m, alpha):
    while True:
        # Positive samples
        idx = np.random.choice(len(users), size=batch_size, replace=False)
        user_batch = users[idx]
        item_batch = items[idx]
        labels = np.ones(len(user_batch))
        weights = np.full(len(user_batch), 1)  # Weight for positive samples

        # Negative samples
        neg_items = np.random.randint(0, num_items, size=(batch_size, m))
        user_batch_neg = np.repeat(user_batch, m)
        neg_items_flat = neg_items.flatten()
        neg_labels = np.zeros(len(neg_items_flat))
        neg_weights = np.full(len(neg_labels), alpha)  # Weight for negative samples

        # Combine positive and negative samples
        combined_user_batch = np.concatenate([user_batch, user_batch_neg])
        combined_item_batch = np.concatenate([item_batch, neg_items_flat])
        combined_labels = np.concatenate([labels, neg_labels])
        combined_weights = np.concatenate([weights, neg_weights])

        # Shuffle
        indices = np.arange(len(combined_labels))
        np.random.shuffle(indices)

        # Yield data as a tuple of tensors with specified types
        yield (
            (tf.convert_to_tensor(combined_user_batch[indices], dtype=tf.int32),
             tf.convert_to_tensor(combined_item_batch[indices], dtype=tf.int32)),
            tf.convert_to_tensor(combined_labels[indices], dtype=tf.float32),
            tf.convert_to_tensor(combined_weights[indices], dtype=tf.float32),
        )

# Neural Collaborative Filtering Model
user_input = Input(shape=(1,), name="user_input")
item_input = Input(shape=(1,), name="item_input")

# Embedding layers
user_mf_embedding = Flatten()(Embedding(num_users, embedding_dim, name="user_mf_embedding")(user_input))
item_mf_embedding = Flatten()(Embedding(num_items, embedding_dim, name="item_mf_embedding")(item_input))

'''

user_mlp_embedding = Flatten()(Embedding(num_users, embedding_dim, name="user_mlp_embedding")(user_input))
item_mlp_embedding = Flatten()(Embedding(num_items, embedding_dim, name="item_mlp_embedding")(item_input))
'''


# Matrix Factorization
mf_vector = Multiply()([user_mf_embedding, item_mf_embedding])

# Multi-Layer Perceptron
'''

mlp_vector = Concatenate()([user_mlp_embedding, item_mlp_embedding])
mlp_layer1 = Dense(64, activation="relu")(mlp_vector)
mlp_layer2 = Dense(32, activation="relu")(mlp_layer1)
mlp_layer3 = Dense(16, activation="relu")(mlp_layer2)
'''

# Combine MF and MLP parts
#combined_vector = Concatenate()([mf_vector, mlp_layer3])
output = Dense(1, activation="sigmoid", name="output")(mf_vector)

# Build the model
model = Model(inputs=[user_input, item_input], outputs=[output])
optimizer = Adam(learning_rate=learning_rate)
#model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=['accuracy'])
model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), metrics=['accuracy', 'AUC', 'Precision', 'Recall'])
# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=2)

# Validation generator
val_generator = data_generator(val_users, val_items, num_items, batch_size, m, alpha)

# Steps for validation
validation_steps = len(val_users) // batch_size

# Training with data generator
steps_per_epoch = len(users) // batch_size


#plot_model(model, to_file='ncf.png')

# Train the model
history = model.fit(
    data_generator(users, items, num_items, batch_size, m, alpha),
    validation_data=val_generator,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    epochs=40,
    verbose=1,
    callbacks=[early_stopping],
)

# Save the history
history_df = pd.DataFrame(history.history)
history_df.to_csv("./data/standard_cf_split_5.csv", index=False)
