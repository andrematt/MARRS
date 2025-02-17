import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import os

# Initialize a list to store the results
results = []

# Loop through each fold (from fold_1 to fold_5)
for i in range(1, 6):
    # Load Data from CSV Files for the current fold
    train_file = f'data/5 neg 5 pos/fold_{i}_train.csv'
    test_file = f'data/5 neg 5 pos/fold_{i}_test.csv'
    
    train_data = pd.read_csv(train_file, sep=',')
    test_data = pd.read_csv(test_file, sep=',')
    
    # Step 2: Split the context into lists of items for both train and test sets
    train_data['context_items'] = train_data['context'].apply(lambda x: x.split(';'))
    test_data['context_items'] = test_data['context'].apply(lambda x: x.split(';'))

    # Initialize the MultiLabelBinarizer
    mlb = MultiLabelBinarizer()

    # Fit and transform the context data for the training set
    train_context_binary = mlb.fit_transform(train_data['context_items'])

    # Transform the context data for the test set
    test_context_binary = mlb.transform(test_data['context_items'])

    # Convert the binary arrays to DataFrames
    train_context_binary_df = pd.DataFrame(train_context_binary, columns=mlb.classes_)
    test_context_binary_df = pd.DataFrame(test_context_binary, columns=mlb.classes_)

    # Concatenate the binary context columns to the original dataframes
    train_data = pd.concat([train_data, train_context_binary_df], axis=1)
    test_data = pd.concat([test_data, test_context_binary_df], axis=1)

    # Drop the original context and context_items columns and any other unnecessary columns
    train_data.drop(columns=['context', 'context_items', 'rule_id', 'goal', 'situation', 'user_type', 'context_nl', 'item_nl'], inplace=True)
    test_data.drop(columns=['context', 'context_items', 'rule_id', 'goal', 'situation', 'user_type', 'context_nl', 'item_nl'], inplace=True)

    X_train = train_data.drop(columns=['label'])
    y_train = train_data['label']

    X_test = test_data.drop(columns=['label'])
    y_test = test_data['label']

    # One-hot encode categorical variables in X_train and X_test
    X_train_encoded = pd.get_dummies(X_train)
    X_test_encoded = pd.get_dummies(X_test)

    # Ensure that both train and test sets have the same columns
    X_train_encoded, X_test_encoded = X_train_encoded.align(X_test_encoded, join='left', axis=1, fill_value=0)

    # Train a Random Forest model (example)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_encoded, y_train)

    # Extract probabilities for the positive class (class 1)
    y_pred_proba = model.predict_proba(X_test_encoded)[:, 1]

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
results_df.to_csv('model_evaluation_results.csv', index=False)

print('Evaluation metrics saved to model_evaluation_results.csv')
