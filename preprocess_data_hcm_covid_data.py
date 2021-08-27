import pickle
import networkx as nx
import numpy as np
import haversine as hs
import community as community_louvain
from algorithms.Node2Vec.model import Node2Vec


def learn_node2vec_embs(G):
    node2vec = Node2Vec(G)
    model = node2vec.process()
    return model


def construct_location_graph(data, p=5):
    G = nx.Graph()
    label_acc_count = 0
    label_idx_dict = {}
    labels = []
    X = []
    for record in data:
        coord = record['coordinates']
        X.append([coord.x, coord.y])
        label_val = record['ward']
        if label_val not in label_idx_dict.keys():
            label_idx_dict[label_val] = label_acc_count
            label_acc_count += 1
        labels.append(label_idx_dict[label_val])

    G.add_nodes_from([node_idx for node_idx in range(len(X))])

    print('Calculating threshold value...')
    A = np.zeros((len(X), len(X)))
    for i in range(len(X)):
        for j in range(len(X)):
            # Calculating the haversine distance between two points
            dst = hs.haversine((X[i][0], X[i][1]), (X[j][0], X[j][1]))
            A[i][j] = dst

    avg_dst = A.mean()
    threshold = avg_dst / p

    for i in range(len(X)):
        for j in range(len(X)):
            if i != j:
                if A[i][j] <= threshold:
                    if not G.has_edge(i, j) and not G.has_edge(j, i):
                        G.add_edge(i, j, weight=A[i][j])

    # Learning the communities from a given graph via Louvain
    print('Extracting community information...')
    node_community_dict = community_louvain.best_partition(G.to_undirected())
    node_com_labels = []
    for i in range(len(X)):
        node_com_labels.append(node_community_dict[i])

    # Learning the network node embeddings via Node2Vec model
    print('Learn node embeddings with Node2Vec...')
    node2vec_embs = []
    node2vec_model = learn_node2vec_embs(G)
    for i in range(len(X)):
        node2vec_embs.append(node2vec_model.wv[str(i)])

    return X, G, node_com_labels, node2vec_embs, labels


processed_data = {}

with open('./data/hcm_covid_data.pkl', 'rb') as f:
    district_covid_case_data = pickle.load(f)
    for district in district_covid_case_data.keys():
        data = district_covid_case_data[district]
        if len(data) > 1:
            print('Processing data of district [{:s}]...'.format(district))
            X, G, node_com_labels, node2vec_embs, labels = construct_location_graph(data)
            processed_data[district] = {
                'coords': X,
                'graph': G,
                'node_com_labels': node_com_labels,
                'node2vec_embs': node2vec_embs,
                'labels': labels
            }
        else:
            print('Skipping data of district [{:s}], number of data record is less than 2'.format(district))

with open('./data/hcm_covid_data_preprocessed.pkl', 'wb') as f:
    pickle.dump(processed_data, f, protocol=pickle.HIGHEST_PROTOCOL)
