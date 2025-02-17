import pandas as pd
import numpy as np
import random
#import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# LAST STEP

df = pd.read_csv("new_dataset.csv", delimiter=",", encoding='utf-8')


#Shuffling
final_df_shuffled = df.sample(frac=1)

#train test split
#train, test = train_test_split(final_df_shuffled, test_size=0.1, shuffle=True)
#train.to_csv('data/train_tf_idf.csv', index=False, encoding="latin-1")
#test.to_csv('data/test_tf_idf.csv', index=False, encoding="latin-1")



#final_df_shuffled.to_csv('IFTTT_NEG_SHUFFLED.csv', index=False, sep=";", encoding="utf-8")


# Function to split dataframe into train-test splits
def split_train_test(df, splits):
    train_dfs = []
    test_dfs = []
    for i in range(len(splits)):
        if i == 0:
            test_df = df[:int(len(df) * splits[i])]
            train_df = df[int(len(df) * splits[i]):]
        else:
            test_df = df[int(len(df) * splits[i-1]):int(len(df) * splits[i])]
            train_df = pd.concat([df[:int(len(df) * splits[i-1])], df[int(len(df) * splits[i]):]])
        train_dfs.append(train_df)
        test_dfs.append(test_df)
    return train_dfs, test_dfs

# Defining splits
splits = [0.2, 0.4, 0.6, 0.8, 1.0]

# Splitting dataframe into train-test splits
train_dfs, test_dfs = split_train_test(final_df_shuffled, splits)

# Exporting train-test splits to CSV files
for i in range(len(train_dfs)):
    train_dfs[i].to_csv(f"fold_{i+1}_train.csv",sep =';',encoding="utf-8", index=False)
    test_dfs[i].to_csv(f"fold_{i+1}_test.csv", sep =';', encoding="utf-8", index=False)


###################################################

#train test split
#train, test = train_test_split(df, test_size=0.2, shuffle=True)
#print(test)
#train.to_csv('data/train_tf_idf.csv', index=False, encoding="latin-1")
#test.to_csv('data/test_tf_idf.csv', index=False, encoding="latin-1")

#Shuffling
#final_df_shuffled = df.sample(frac=1)
#print(final_df_shuffled)
#final_df_shuffled.to_csv('data/Book3_synthetic_1NEG_single_elements.csv', index=False, sep=";", encoding="latin-1")

