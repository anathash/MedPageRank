import csv

DEFAULT_STANCE_DICT = {1:1, 2:2,3:3,4:4,5:5}
DEFAULT_STANCE_TO_CLASS = {1:1, 2:1,3:2,4:3,5:3}


def gen_majority_from_feature_file(feature_file, output_dir,  stance_dict = DEFAULT_STANCE_DICT, stance_to_class = DEFAULT_STANCE_TO_CLASS):
    with open(output_dir + 'majority,csv', 'w', encoding='utf-8', newline='') as outputCsv:
        fieldnames = ['query', 'label', 'majority_value','majority_class']
        fieldnames.extend(list(set(stance_dict.values())))
        wr = csv.DictWriter(outputCsv, fieldnames=fieldnames)
        wr.writeheader()
        with open(feature_file, encoding='utf-8', newline='') as features_csv:
            reader = csv.DictReader(features_csv)
            for row in reader:
                output_row = {}
                for i in range(1,6):
                    field = 'stance'+str(i)+'_votes'
                    stance_votes = int(row[field])
                    key = stance_dict(i)
                    output_row[key] = stance_votes
                sorted_votes = sorted(output_row.items(), key=lambda kv: kv[1], reverse=True)
                output_row['majority_value'] = sorted_votes[0][0]
                output_row['majority_class'] = stance_to_class[output_row['majority']]
                output_row['query'] = row['query']
                output_row['label'] = row['label']
                wr.writerow(output_row)



