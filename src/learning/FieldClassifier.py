import csv
import math
import numpy

from sklearn.tree import DecisionTreeClassifier, export_text

from preprocessing import FeaturesGenerator


def get_class(prediction):
    if prediction < 3:
        return 1
    elif prediction < 5:
        return 3
    else:
        return 5



class FieldsClassifier:
    def __init__(self, fields_weights_dict, filename):
        self.fields_weights_dict = fields_weights_dict

        self.mname = "Fields_Classifier"
        for f in fields_weights_dict.keys():
            self.mname += '_'+f
        self.filename = filename


    def gen_predictions(self):
        self.predictions = {}
        with open(self.filename, encoding='utf-8', newline='') as queries_csv:
            reader = csv.DictReader(queries_csv)
            for row in reader:
                vals = {}
                q = row['query']
                self.predictions[q] = {}
                for i in range(1,6):
                    vals[i] = 0
                    for f,w in self.fields_weights_dict.items():
                        vals[i] += float(row[f+str(i)+'_mean'])*w
                sorted_vals = sorted(vals.items(), key=lambda x: x[1], reverse=True)
                prediction = list(sorted_vals)[0][0]
                self.predictions[q][self.mname] = prediction
                label = int(row['label'])
                self.predictions[q]['error'] = math.fabs(prediction - label)
                self.predictions[q]['acc'] = int(get_class(prediction)
                                                 == get_class(label))
        err = [r['error'] for q,r in self.predictions.items()]
        acc = [r['acc'] for q,r in self.predictions.items()]

        print('mae = ' + str(numpy.mean(err)))
        print('acc = ' + str(numpy.mean(acc)))

    def predict(self,x):
        return self.predictions[x]

    def model_name(self):
        return self.mname

def classify_by_field():
    input_folder = 'C:\\research\\falseMedicalClaims\\ECAI\\model input\\'
    cls = 'all_equal_weights'
    input_dir = input_folder + cls + '\\by_group'
    feature_file = "group_features_by_stance"
    fc = FieldsClassifier({'h_index':0.5,'citation_count':0.5},input_dir + '\\' + feature_file + '.csv')
    fc.gen_predictions()
    fc = FieldsClassifier({'h_index':1},input_dir + '\\' + feature_file + '.csv')
    fc.gen_predictions()
    fc = FieldsClassifier({'citation_count':1},input_dir + '\\' + feature_file + '.csv')
    fc.gen_predictions()

def main():
    classify_by_field()

if __name__ == '__main__':
    main()