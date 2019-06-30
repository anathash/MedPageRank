import csv
import pandas

import pandas as pd
ISSN_TO_ISSNL = 'C:\\research\\falseMedicalClaims\\issnltables\\20190625.ISSN-to-ISSN-L.TXT'
ISSNL_TO_ISSNL = 'C:\\research\\falseMedicalClaims\\issnltables\\20190625.ISSN-L-to-ISSN1.TXT'
class HIndex:
    def __init__(self, filename):
        self.h_index = {}
        pandas.set_option('display.max_rows', 50)
        pandas.set_option('display.max_columns', 50)
        pandas.set_option('display.width', 1000)  # Clerical work:
        results = pd.read_csv(filename, delimiter=';', error_bad_lines=True)
        for index, row in results.iterrows():
            issn_list = row['Issn'].split(',')
            h_index = row['H index']
            for issn in issn_list:
                formated_issn = self.format_issn(issn)
                self.h_index[formated_issn] = h_index
        #        self.issn_to_issnl = pd.read_csv(ISSN_TO_ISSNL, delimiter='\t', error_bad_lines=True)
        #        self.issn_to_issnl.astype('str')
        #        self.issnl_to_issn = pd.read_csv(ISSNL_TO_ISSNL, delimiter='\t', error_bad_lines=True)
        #        self.issnl_to_issn.set_index('ISSN-L')

    def add_issnl(self):
        linking_table = pd.read_csv(ISSN_TO_ISSNL, delimiter='\t', error_bad_lines=True)
        for index, row in linking_table.iterrows():
            issnl = row['ISSN-L']
            issn = row['ISSN']
            if issn != issnl:
                if issn in self.h_index:
                    if issnl in self.h_index:
                        assert(self.h_index[issnl] == self.h_index[issnl])
                    else:
                        self.h_index[issnl] = self.h_index[issn]
                else:
                    if issnl in self.h_index:
                        self.h_index[issn] = self.h_index[issnl]

    @staticmethod
    def format_issn(issn):
        striped_issn = issn.strip()
        striped_issn = striped_issn.rjust(8, '0')
        issn_hyphen = striped_issn[:4] + '-' + striped_issn[4:]
        return issn_hyphen


    def get_hIndex_by_issnl(self, issnl):
        print('issnl ' + issnl + ' not found. Looking in issn-l table')
        #issnl_df = self.issnl_to_issn.loc[self.issn_to_issnl['ISSN-L'] == issnl]
        issnl_df = self.issnl_to_issn['ISSN-L'].where(self.issn_to_issnl['ISSN-L'] == issnl)
        if issnl_df.empty:
            print('no issnl for  ' + issnl)
            return 0
        print(issnl_df)
        for i in range(1, 68):
            issn = issnl_df.iloc[0]['ISSN'+str(i)]
            if not issn:
                return 0
            if issn in self.h_index:
                print('found hIndex by Issnl')
                return self.h_index[issnl]

        print('issn ' + issn + ' not found by issnl ' + issnl)
        return 0

    def get_H_index1(self, issn):
        if not issn:
            return 0
        if issn == '2213-8595':
            print('stop')
        if issn in self.h_index:
            return self.h_index[issn]
        h_index = self.get_hIndex_by_issn(issn)
        if h_index ==0:
            return  self.get_hIndex_by_issnl(issn)
        return 0

    def get_H_index(self, issn):
        return self.h_index[issn] if issn and issn in self.h_index else 0

    def get_hIndex_by_issn(self, issn):
        print('issn ' + issn + ' not found. Looking in issn-l table')
        issnl_df = self.issn_to_issnl.loc[self.issn_to_issnl['ISSN'] == issn]
        print(issnl_df)
        if issnl_df.empty:
            print('no issnl for  ' + issn)
            return 0
        issnl = issnl_df.iloc[0]['ISSN-L']

        if issnl in self.h_index:
            print('found hIndex by Issnl')
            return self.h_index[issnl]
        print('issn ' + issn + ' not found by issnl ' + issnl)
        return 0


if __name__ == '__main__':
    hindex = HIndex("scimagojr 2018.csv")
