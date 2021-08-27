import pickle

from algorithms.CP2Vec.model import CP2Vec


def main():
    with open('./data/us_crime_data_preprocessed.pkl', 'rb') as input_f:
        data = pickle.load(input_f)
        G, node_com_labels, node2vec_embs = data['graph'], data['node_com_labels'], data['node2vec_embs']

        if len(G.edges()) > 0 and len(G.nodes()) > 0:
            model = CP2Vec(G, node2vec_embs, node_com_labels)
            model.process()
            data['CP2Vec_node_embs'] = model.CP2Vec_node_embs

        with open('./data/us_crime_data_final.pkl', 'wb') as output_f:
            pickle.dump(data, output_f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
