import json
import os
from shutil import copyfile

from Paper import Paper


class PaperCache(object):
    def __init__(self, cache_file_name):
        file_name_no_prefix = cache_file_name[0:len(cache_file_name)-5]
        copy_file_name = file_name_no_prefix + '_old.json'
        if os.path.isfile(copy_file_name):
            os.remove(copy_file_name)
        if os.path.isfile(cache_file_name):
            copyfile(cache_file_name, copy_file_name)
        self.cache = {}
        self.cache_file_name = cache_file_name

        try:
            with open(cache_file_name) as cache_file:
                data = json.loads(cache_file.read())
                for k, v in data.items():
                    self.cache[k] = \
                        Paper(v['pmid'], v['title'], v['journal'], v['authors'], v['pm_cited'], v['h_index'], v['issn'])
        except FileNotFoundError:
            return

    def add_paper(self, pmid, paper):
        self.cache[pmid] = paper

    def get_paper(self, pmid):
        if pmid in self.cache:
            return self.cache[pmid]
        return None

    def save_cache(self):
        with open(self.cache_file_name, 'w') as file:
            data = json.dumps(self.cache, default=lambda o: o.__dict__)
            file.write(data)  # use `json.loads` to do the reverse

