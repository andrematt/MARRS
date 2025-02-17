import json
import csv

'''
Use these scripts to recreate the "sliding window IFTTT" approach starting from the original dataset by Mi and colleagues:
Xianghang Mi, Feng Qian, Ying Zhang, and XiaoFeng Wang. An Empirical Characterization of IFTTT: Ecosystem, Usage, and Performance. 
In Proceedings of IMC ’17. ACM, New York, NY, USA, 7 pages. 
https://doi. org/10.1145/3131365.3131369) 
'''

##### First step: download the dataset from https://www-users.cse.umn.edu/~fengqian/ifttt_measurement/ and put it in the folder "data" 
##### We have used the last version "May 2017"

# Function to convert JSON to CSV
def json_to_csv(json_file, csv_file):
    with open(json_file, 'r') as f:
        data = f.readlines()

    # Remove newline characters
    data = [line.strip() for line in data]

    # Open CSV file in write mode
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["isDoRecipe", "title", "desc", "triggerChannelTitle", "triggerChannelId", "triggerChannelUrl", "triggerTitle", "triggerDesc", "triggerId", "actionChannelTitle", "actionChannelId", "actionChannelUrl", "actionTitle", "actionDesc", "actionId", "favoritesCount", "addCount", "creatorName", "creatorUrl", "url"])
        
        # Write CSV header
        writer.writeheader()
        
        # Iterate over JSON lines and write to CSV
        for line in data:
            row = json.loads(line)
            writer.writerow(row)

# Example usage
json_file = "data/recipes.json"
csv_file = "output.csv"
json_to_csv(json_file, csv_file)
