import csv
import random
import pandas as pd

### Use this script to generate negative samples directly into the dataset
### We have manage the negative examples directly into the training loop, but it also can be useful to have the negative examples beforehand

def read_dataset(input_csv):
    with open(input_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)

def append_random_rows(input_csv, num_appends):
    rows = read_dataset(input_csv)
    #fieldnames = rows[0].keys()

    #with open(output_csv, 'a', newline='', encoding='utf-8') as f:
    #    writer = csv.DictWriter(f, fieldnames=fieldnames)

    for row in rows:
        for _ in range(num_appends):
            random_row = random.choice(rows)
            positive_row = row.copy()
            positive_row['label'] = "1"
            negative_row = row.copy()
            negative_row['item'] = random_row['item']
            negative_row['item_nl'] = random_row['item_nl']
            negative_row['label'] = "0"
            new_rows.append(positive_row)
            new_rows.append(positive_row)
            new_rows.append(negative_row)

                #writer.writerow(new_row)
new_rows = []

# Example usage
input_csv = "new_dataset.csv"
num_appends = 1

append_random_rows(input_csv, num_appends)

final_df = pd.DataFrame(new_rows)
final_df.to_csv('IFTTT_NEG.csv', index=False, sep=";", encoding="utf-8")



