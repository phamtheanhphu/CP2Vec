import pickle
from algorithms.CP2Vec.model import CP2Vec


def main():
    CP2Vec_learnt_data = {}
    with open('./data/hcm_covid_data_preprocessed.pkl', 'rb') as f:
        district_covid_case_data = pickle.load(f)
        for district in district_covid_case_data.keys():
            print('Processing data of district [{:s}]...'.format(district))
            data = district_covid_case_data[district]
            G, node_com_labels, node2vec_embs = data['graph'], data['node_com_labels'], data['node2vec_embs']
            if len(G.edges()) > 0 and len(G.nodes()) > 0:
                model = CP2Vec(G, node2vec_embs, node_com_labels)
                model.process()
                data['CP2Vec_node_embs'] = model.CP2Vec_node_embs
                CP2Vec_learnt_data[district] = data
            else:
                print('Skipping data of district [{:s}], number of edges[{:d}]/nodes[{:d}] is less than 2'.format(
                    district, len(G.edges()), len(G.nodes())))

    with open('./data/hcm_covid_data_final.pkl', 'wb') as f:
        pickle.dump(CP2Vec_learnt_data, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
