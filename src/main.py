from jsonpickle import json

from HIndex import HIndex
from PaperBuilder import PaperBuilder
from PaperCache import PaperCache
from Paper_Network import P_N
import page_rank_alg as pr
rsp = 0.31


def main():
    paper_cache = PaperCache('cinnamon.json')
    hindex = HIndex("scimagojr 2018.csv")
    papers_network = P_N("C:/reasearch/falseMedicalClaims/turk/cinnamon/Does cinnamon help diabetes_bestMatch.csv", PaperBuilder(hindex, paper_cache))
    papers_network.create_network()
#    with open('input_network.json', 'w') as file:
#        file.write(json.dumps(papers_network.papers_dict))  # use `json.loads` to do the reverse
    edgeWeights, papers_h_index = papers_network.get_network_edges_weights()
    ranks = pr.powerIteration(edgeWeights, papers_h_index, rsp=rsp)
    print(ranks)
    print('--')
    for paper in papers_network.csv_papers:
        rank = ranks.get(paper)
        print(paper + ' : ' + str(rank))
    v=9


if __name__ == '__main__':
    main()