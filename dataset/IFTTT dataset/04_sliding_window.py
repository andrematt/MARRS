import csv

def create_new_dataset(input_csv, output_csv):
    new_rows = []

    # Open input CSV file in read mode
    with open(input_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        # Iterate through each row
        count = 0
        newrows_count = 0
        for row in reader:
            count += 1
            trigger_type = row['triggerTitle']
            trigger_desc = row['triggerDesc']
            action_type = row['actionTitle']
            action_desc = row['actionDesc']
            creator_name = row['creatorName']
            rule_id = count

            trigger_channel = row['triggerChannelTitle']
            action_channel = row['actionChannelTitle']

            # First row
            new_rows.append({'id':rule_id, 'item': trigger_channel + "-" + trigger_type, 'item_nl': trigger_desc, 'context': 'none', 'context_nl': 'none', 'user': creator_name})
            newrows_count += 1
            
            # Second row
            new_rows.append({'id':rule_id, 'item': action_channel + "-" + action_type, 'item_nl': action_desc, 'context': 'none', 'context_nl': 'none', 'user': creator_name})
            newrows_count += 1
            
            # Third row 
            new_rows.append({'id':rule_id, 'item': trigger_channel + "-" + trigger_type, 'item_nl': trigger_desc, 'context': action_channel + "-" + action_type, 'context_nl': action_desc, 'user': creator_name})
            newrows_count += 1
            
            # Fourth row
            new_rows.append({'id':rule_id, 'item': action_channel + "-" + action_type, 'item_nl': action_desc, 'context': trigger_channel + "-" + trigger_type, 'context_nl': trigger_desc, 'user': creator_name})
            newrows_count += 1

    print("original rows processed: " +str(count))
    print("appended new row: " +str(newrows_count))
    # Write new dataset to output CSV file
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['id', 'item', 'item_nl', 'context', 'context_nl', 'user']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(new_rows)

# Example usage
#input_csv = "filtered_output.csv"
input_csv = "output_with_desc.csv" #for use without filtering      
output_csv = "new_dataset_with_ids.csv"

create_new_dataset(input_csv, output_csv)