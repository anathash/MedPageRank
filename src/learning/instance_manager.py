import csv
import math
import os
import random

import pandas as pd


def gen_test_train_set(input_dir, train_percent):
    filenames = os.listdir(input_dir)
    example_files = [x for x in filenames if x.endswith(".csv")]
    train_files_num = int(math.ceil(train_percent * len(example_files)))
    random.shuffle(example_files)
    train_fnames = example_files[:train_files_num]
    train_df = pd.concat([pd.read_csv(f) for f in train_fnames], ignore_index=True)

    test_fnames = example_files[train_files_num:]
    test_df = pd.concat([pd.read_csv(f) for f in test_fnames], ignore_index=True)
    return (train_df, test_df)


class InstanceManager:
    def __init__(self, input_dir, train_percent):
        self.train_df, self.test_df = gen_test_train_set(input_dir, train_percent)

    def get_train_df(self):
        return self.train_df

    def get_test_df(self):
        return self.test_df
