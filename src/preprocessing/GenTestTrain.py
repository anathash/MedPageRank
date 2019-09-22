import csv
import math
import random


def get_test_train(test_portion, queries, examples_folder):
    with open(queries, encoding='utf-8', newline='') as queries_csv:
        reader = csv.DictReader(queries_csv)
        positive = []
        negative = []
        for row in reader:
            label = row['label']
            if label == 1:
                negative.append(row['short query'])
            else:
                positive.append(row['short query'])
        num_queries = len(negative) + len(positive)
        test_num_queries_per_class = int(math.ceil(num_queries*test_portion/2))
        positive_training_examples = random.sample(positive, test_num_queries_per_class)
        negative_training_examples = random.sample(negative, test_num_queries_per_class)



def main():
    get_test_train(0.7, 'C:\\research\\falseMedicalClaims\\examples\\queries.csv',
                   "C:\\research\\falseMedicalClaims\\examples\\model input\\group1\\")

if __name__ == '__main__':
    main()