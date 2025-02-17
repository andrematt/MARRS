import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import HashingVectorizer

# Define the number of negative samples to generate per positive example
NEGATIVE_SAMPLES = 5

# Safe label encoding to handle unseen labels
def safe_transform(encoder, values, unknown_value=-1):
    """
    Transforms the values using the encoder. Assigns unknown_value to unseen labels.
    """
    known_classes = set(encoder.classes_)
    return np.array([encoder.transform([v])[0] if v in known_classes else unknown_value for v in values])


#### NOTE: we have to use the "hashing trick" (https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.FeatureHasher.html) to reduce the dimensionality of the context, users and items, otherwise we run out of memory 
# Function to encode context using HashingVectorizer
def encode_context_with_hashing(contexts, n_features=4):
    vectorizer = HashingVectorizer(n_features=n_features, binary=True, alternate_sign=False)
    return vectorizer.fit_transform(contexts).toarray()

# Function to encode users and items with hashing to reduce dimensionality
def encode_with_hashing(values, n_features=16):
    vectorizer = HashingVectorizer(n_features=n_features, binary=True, alternate_sign=False)
    return vectorizer.fit_transform(values).toarray()

# Initialize a list to store the results
results = []

# Start fold processing
i = 5
print("Starting fold", i)

# Load Data from CSV Files for the current fold
train_file = f'data/with_ids_fold_{i}_train.csv'
test_file = f'data/with_ids_fold_{i}_test.csv'

train_data = pd.read_csv(train_file, sep=';')
test_data = pd.read_csv(test_file, sep=';')

# Step 1: Generate negative samples for training data
all_items = train_data['item'].unique()
negative_samples = []

print("HERE 1")
for _, row in train_data.iterrows():
    user = row['user']
    positive_item = row['item']
    context = row['context']

    # Generate negative samples
    negative_items = np.random.choice(
        [item for item in all_items if item != positive_item], 
        size=NEGATIVE_SAMPLES, 
        replace=False
    )

    for neg_item in negative_items:
        negative_samples.append({
            'user': user,
            'item': neg_item,
            'context': context,
            'label': 0  # Negative label
        })

print("HERE 2")
# Append the negative samples to the training data
train_data['label'] = 1  # Positive label for original examples
train_data = pd.concat([train_data, pd.DataFrame(negative_samples)], ignore_index=True)

# Shuffle the training data
train_data = train_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Step 2: Prepare test data
# For evaluation, generate negative samples for the test set
test_negative_samples = []
print("HERE 3")

for _, row in test_data.iterrows():
    user = row['user']
    positive_item = row['item']
    context = row['context']

    # Generate negative samples
    negative_items = np.random.choice(
        [item for item in all_items if item != positive_item], 
        size=NEGATIVE_SAMPLES, 
        replace=False
    )

    for neg_item in negative_items:
        test_negative_samples.append({
            'user': user,
            'item': neg_item,
            'context': context,
            'label': 0  # Negative label
        })

test_data['label'] = 1  # Positive label for original examples
test_data = pd.concat([test_data, pd.DataFrame(test_negative_samples)], ignore_index=True)

# Shuffle the test data
test_data = test_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Step 3: Preprocess the training and test data
print("HERE 4")

# Encode user and item using hashing to reduce dimensionality
user_train = encode_with_hashing(train_data['user'], n_features=128)
item_train = encode_with_hashing(train_data['item'], n_features=128)

user_test = encode_with_hashing(test_data['user'], n_features=128)
item_test = encode_with_hashing(test_data['item'], n_features=128)

# Encode context using HashingVectorizer
print("HERE 5.5")
context_train = encode_context_with_hashing(train_data['context'], n_features=32)
context_test = encode_context_with_hashing(test_data['context'], n_features=32)

print("HERE 6")

# Combine user, item, and context binary features into a single input
X_train = np.hstack([
    user_train,
    item_train,
    context_train
])
y_train = train_data['label'].to_numpy()

print("HERE 7")

X_test = np.hstack([
    user_test,
    item_test,
    context_test
])
print("HERE 8")
y_test = test_data['label'].to_numpy()

print("Starting to train!!")

# Train a Random Forest model (example)
model = RandomForestClassifier(n_estimators = 50,verbose=3,n_jobs=-1)
model.fit(X_train, y_train)

# Extract probabilities for the positive class (class 1)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Convert probabilities to binary predictions with a threshold of 0.5
y_pred = (y_pred_proba > 0.5).astype(int)

# Calculate metrics
roc_auc = roc_auc_score(y_test, y_pred_proba)
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

# Append the results for the current fold
results.append({
    'Fold': i,
    'Precision': precision,
    'Recall': recall,
    'F1 Score': f1,
    'ROC AUC': roc_auc
})

# Convert the results to a DataFrame
results_df = pd.DataFrame(results)

# Save the results to a CSV file
results_df.to_csv(f'IDS_random_forest_fold_{i}.csv', index=False)

print(f'Evaluation metrics saved to IDS_random_forest_fold_{i}.csv')