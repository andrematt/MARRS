import pandas as pd
import numpy as np

#### LAST STEP: use this script to ensure that automations with the same ID stay in the same set (as we have done in the paper). 
#### If this is not required, use the other script 06_shuffle_and_train_test_split.py 


# Load dataset
df = pd.read_csv("new_dataset_with_ids.csv", delimiter=",", encoding='utf-8')

# Shuffle dataset by unique IDs
unique_ids = df['id'].unique()
np.random.shuffle(unique_ids)

def split_train_test_by_id(df, splits):
    train_dfs, test_dfs = [], []
    
    num_ids = len(unique_ids)
    split_indices = [int(num_ids * s) for s in splits]
    
    prev_index = 0
    for i in range(len(splits)):
        test_ids = unique_ids[prev_index:split_indices[i]]
        train_ids = np.setdiff1d(unique_ids, test_ids)
        
        test_df = df[df['id'].isin(test_ids)].sample(frac=1, random_state=42).reset_index(drop=True)
        train_df = df[df['id'].isin(train_ids)].sample(frac=1, random_state=42).reset_index(drop=True)
        
        train_dfs.append(train_df)
        test_dfs.append(test_df)
        
        prev_index = split_indices[i]
        
    return train_dfs, test_dfs

# Define splits
splits = [0.2, 0.4, 0.6, 0.8, 1.0]

# Perform train-test split ensuring same ID stays in one set
train_dfs, test_dfs = split_train_test_by_id(df, splits)

# Export train-test splits to CSV files
for i in range(len(train_dfs)):
    train_dfs[i].to_csv(f"with_ids_fold_{i+1}_train.csv", sep=';', encoding="utf-8", index=False)
    test_dfs[i].to_csv(f"with_ids_fold_{i+1}_test.csv", sep=';', encoding="utf-8", index=False)
