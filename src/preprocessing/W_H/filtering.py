import csv


def get_concezus(file, conc_percenent, stance_dict):
    with open(file, encoding='utf-8', newline='') as input_csv:
        reader = csv.DictReader(input_csv)
        conc_queries = []
        for row in reader:
            votes = {}
            query = row['query'].strip()
            for i in range(1,6):
                votes_ind = stance_dict[i]
                if votes_ind not in votes:
                    votes[votes_ind] = 0
                votes[votes_ind] += int(row[str(i)])
            votes_sum = sum(list(votes.values()))
            normed_votes = {x:y/votes_sum for x,y in votes.items()}
            for votes_percent in normed_votes.values():
                if votes_percent >= conc_percenent:
                    conc_queries.append(query)
                    continue
    return conc_queries

def get_outlayers(file):
    with open(file, encoding='utf-8', newline='') as input_csv:
        reader = csv.DictReader(input_csv)
        outlayers = []
        for row in reader:
            query = row['query'].strip()
            label = int(row['label'])
            label_votes = int(row[str(label)])
            if label_votes == 0:
                outlayers.append(query)

    return outlayers


def get_outlayers_from_file(file):
    with open(file, encoding='utf-8', newline='') as input_csv:
        reader = csv.DictReader(input_csv)
        outlayers = []
        for row in reader:
            outlayers.append(row['query'])

    return outlayers

def gen_filter_file(output_dir, filename, queries):
    with open(output_dir + filename, 'w', encoding='utf-8', newline='') as postCSv:
        wr = csv.DictWriter(postCSv, fieldnames=['query'])
        wr.writeheader()
        for row in queries:
            wr.writerow({'query':row})


def filter_queries(feature_file, output_file, queries):
    with open(feature_file, encoding='utf-8', newline='') as input_csv:
        reader = csv.DictReader(input_csv)
        fields = reader.fieldnames
        with open(output_file, 'w', encoding='utf-8', newline='') as outcsv:
            wr = csv.DictWriter(outcsv, fieldnames=fields)
            wr.writeheader()
            for row in reader:
                query = row['query'].strip()
                if query not in queries:
                    wr.writerow(row)



def filter_features(feature_file, features_list, output_filename):
    fieldnames = ['query', 'label']
    fieldnames.extend(features_list)
    with open(output_filename, 'w', encoding='utf-8', newline='') as outcsv:
        wr = csv.DictWriter(outcsv, fieldnames=fieldnames)
        wr.writeheader()
        with open(feature_file, encoding='utf-8', newline='') as input_csv:
            reader = csv.DictReader(input_csv)
            for row in reader:
                out_row = {'query': row['query'], 'label':row['label']}
                for feature in features_list:
                    out_row[feature] = row[feature]
                wr.writerow(out_row)

def gen_feature_files():
    feature_file = 'C:\\research\\falseMedicalClaims\\White and Hassan\\model input\\mult_features\\group_features_by_stance_label_shrink_filtered.csv'
    output_file = 'C:\\research\\falseMedicalClaims\\White and Hassan\\model input\\mult_features\\h_index_by_stance_label_shrink_filtered.csv'
    features_list = []
    for i in range(1,6):
        features_list.append('h_index'+str(i)+'_normed_mean')
    filter_features(feature_file=feature_file, features_list=features_list, output_filename= output_file)

def filter_outlayers(filename):
    feature_file = 'C:\\research\\falseMedicalClaims\\White and Hassan\\model input\\mult_features\\' +filename +'.csv'
    output_dir = 'C:\\research\\falseMedicalClaims\\White and Hassan\\model input\\mult_features\\'
    outlayers_file = 'C:\\research\\falseMedicalClaims\\White and Hassan\\model input\\mult_features\\outlayers.csv'
    output_file = 'C:\\research\\falseMedicalClaims\\White and Hassan\\model input\\mult_features\\' + filename + '_nol.csv'
    outlayers = get_outlayers_from_file(outlayers_file)
    filter_queries(feature_file, output_file, outlayers)

def filter_file():
    majority = 'C:\\research\\falseMedicalClaims\\White and Hassan\\model input\\mult_features\\outlayers\\majority.csv'
   # feature_file = 'C:\\research\\falseMedicalClaims\\White and Hassan\\model input\\mult_features\\outlayers\\group_features_by_label_shrink.csv'
    feature_file = 'C:\\research\\falseMedicalClaims\\White and Hassan\\model input\\mult_features\\outlayers\\group_features_by_label_shrink.csv'
    output_dir = 'C:\\research\\falseMedicalClaims\\White and Hassan\\model input\\mult_features\\'
    output_file = 'C:\\research\\falseMedicalClaims\\White and Hassan\\model input\\mult_features\\group_features_by_stance_label_shrink_filtered.csv'
    majority_nol = 'C:\\research\\falseMedicalClaims\\White and Hassan\\model input\\mult_features\\majority_filtered.csv'


    #    gen_outlayers_file(output_dir, outlayers)
    filtered_queries = get_outlayers(majority)
    conc_queries = get_concezus(file=majority, conc_percenent=0.85, stance_dict={1: 1, 2: 1, 3: 3, 4: 5, 5: 5})
    filtered_queries.extend(conc_queries)
    gen_filter_file(output_dir, 'concezus.csv', filtered_queries)
    filter_queries(feature_file, output_file, filtered_queries)
    filter_queries(majority, majority_nol, filtered_queries)


def main():
    filter_outlayers('scaled_group_features_by_stance')
   # gen_feature_files()
if __name__ == '__main__':
    main()
#TODO: norm, return FG to hindex naive scaling