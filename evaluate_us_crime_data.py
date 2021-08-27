import pickle

import hdbscan
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.cluster import normalized_mutual_info_score

default_eps = 0.005
default_min = 10

with open('./data/us_crime_data_final.pkl', 'rb') as f:
    data = pickle.load(f)

    coords = data['coords']
    ground_truth_labels = data['labels']
    node_com_labels = data['node_com_labels']
    CP2Vec_node_embs = data['CP2Vec_node_embs']

    DBSCAN_model = DBSCAN(eps=default_eps, min_samples=default_min, metric='haversine')
    DBSCAN_model.fit(coords)
    DBSCAN_y_preds = DBSCAN_model.fit_predict(coords)

    HDBSCAN_model = hdbscan.HDBSCAN(min_cluster_size=10)
    HDBSCAN_y_preds = HDBSCAN_model.fit_predict(coords)

    CP2Vec_DBSCAN_model = DBSCAN(eps=default_eps, min_samples=default_min, metric='cosine')
    CP2Vec_DBSCAN_model.fit(CP2Vec_node_embs)
    CP2Vec_DBSCAN_y_preds = CP2Vec_DBSCAN_model.fit_predict(CP2Vec_node_embs)

    CP2Vec_HDBSCAN_model = hdbscan.HDBSCAN(min_cluster_size=default_min)
    CP2Vec_HDBSCAN_y_preds = CP2Vec_HDBSCAN_model.fit_predict(CP2Vec_node_embs)

    DBSCAN_nmi_score = normalized_mutual_info_score(np.asarray(DBSCAN_y_preds),
                                                    np.asarray(ground_truth_labels))

    HDBSCAN_nmi_score = normalized_mutual_info_score(np.asarray(HDBSCAN_y_preds),
                                                     np.asarray(ground_truth_labels))

    CP2Vec_DBSCAN_nmi_score = normalized_mutual_info_score(np.asarray(CP2Vec_DBSCAN_y_preds),
                                                           np.asarray(ground_truth_labels))

    CP2Vec_HDBSCAN_nmi_score = normalized_mutual_info_score(np.asarray(CP2Vec_HDBSCAN_y_preds),
                                                            np.asarray(ground_truth_labels))

    print('NMI accuracy score (DBSCAN): {:.6f}'.format(DBSCAN_nmi_score))
    print('NMI accuracy score (HDBSCAN): {:.6f}'.format(HDBSCAN_nmi_score))
    print('NMI accuracy score (CP2Vec + DBSCAN): {:.6f}'.format(CP2Vec_DBSCAN_nmi_score))
    print('NMI accuracy score (CP2Vec + HDBSCAN): {:.6f}'.format(CP2Vec_HDBSCAN_nmi_score))
