import csv


### Use this script to filter out rows with empty triggerDesc or actionDesc. 
### We have not performed this step, but it can be useful to further experiment with Bert model using also the descriptions of triggers and actions
 
def filter_empty_desc(input_csv, output_csv):
    # Open input CSV file in read mode
    with open(input_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Filter rows with empty triggerDesc or actionDesc
    filtered_rows = [row for row in rows if row['triggerDesc'] != '' and row['actionDesc'] != '']

    # Write filtered rows to new CSV file
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        fieldnames = reader.fieldnames
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(filtered_rows)

# Example usage
input_csv = "output_with_desc.csv"
output_csv = "filtered_output.csv"

filter_empty_desc(input_csv, output_csv)