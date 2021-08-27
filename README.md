##CP2Vec

**A community-aware graph neural network representation learning for enhancing location-based clustering task**

This is the source code of *CP2Vec* model which is a geographical location-based representation learning via graph attention network (GAT). The achieved representations of locations are then used to facilitate the density-based spatial clustering algorithms, like as: DBSCAN and HDBSCAN.

Belows are some visualizations of clustering results for COVID-19 hotpots analysis in 13 districts of the Ho Chi Minh city, VN and reported criminal cases in Hartford city, US.


<sup>Illustrations of COVID-19 confirmed cases/isolated area clustering via different techniques in several large/high-population districts of Ho Chi Minh city, Viet Nam</sup>

![alt text](https://github.com/phamtheanhphu/CP2Vec/blob/main/images/hcm_covid_data.png?raw=true)

<sup>Illustrations of police reported criminal case clustering via different techniques in Hartford city, United States</sup>

![alt text](https://github.com/phamtheanhphu/CP2Vec/blob/main/images/us_crime_data.png?raw=true)

<sup>The 3-D visualizations of learnt embedding vectors corresponding with the reported criminal cases in Hartford city, US which are learnt by our CP2Vec model</sup>

![alt text](https://github.com/phamtheanhphu/CP2Vec/blob/main/images/us_crime_pca_tsne.png?raw=true)


### Requirements

- Python >= 3.6
- Skicit-Learn >= 0.24.2
- NetworkX >= 1.1
- Gensim >= 3.8.x
- PyTorch >= 1.7.0
- Python-Louvain >= 0.13
- Haversine > 2.4
- HDBSCAN >= 0.8.27


### Datasets

- COVID-19 confirmed cases/isolated areas data in Ho Chi Minh city, VN: https://bando.tphcm.gov.vn/ogis
- Police Crime Data in ArcGIS hub: https://hub.arcgis.com/datasets/b226933a414046c498764d3b6821826f_2/explore


### Miscellaneous

Please send any question you might have about the code and/or the algorithm to <phamtheanhphu@gmail.com>.
