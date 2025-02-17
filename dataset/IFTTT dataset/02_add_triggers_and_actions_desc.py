import json
import csv


def read_json_lines(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line.strip()) for line in f.readlines()]

def update_csv_with_desc(csv_file, trigger_list, action_list, output_csv):
    # Read trigger and action lists
    trigger_data = read_json_lines(trigger_list)
    action_data = read_json_lines(action_list)

    # Open CSV file in read mode
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Update CSV with descriptions
    for row in rows:
        trigger_title = row['triggerTitle']
        action_title = row['actionTitle']
        trigger_desc = None
        action_desc = None

        # Find matching trigger
        for trigger in trigger_data:
            if trigger['triggerTitle'] == trigger_title:
                trigger_desc = trigger['triggerDesc']
                break

        # Find matching action
        for action in action_data:
            if action['actionTitle'] == action_title:
                action_desc = action['actionDesc']
                break

        # Update row with descriptions
        row['triggerDesc'] = trigger_desc
        row['actionDesc'] = action_desc

    # Write updated rows to new CSV file
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        fieldnames = reader.fieldnames + ['triggerDesc', 'actionDesc']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
# Example usage
csv_file = "output.csv"
trigger_list_file = "data/triggerList.json"
action_list_file = "data/actionList.json"
output_csv = "output_with_desc.csv"

update_csv_with_desc(csv_file, trigger_list_file, action_list_file, output_csv)