import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from algorithms.GAT.utils import load_data
from algorithms.GAT.models import GAT


class CP2Vec(object):
    def __init__(self, G, node_features, labels):
        super(CP2Vec, self).__init__()
        self.G = G
        self.node_features = node_features
        self.labels = labels

    def process(self, hidden_size=128, epochs=20):
        self.__init_GAT_model(hidden_size)
        for epoch in range(epochs):
            self.GAT_model.train()
            self.optimizer.zero_grad()
            output = self.GAT_model(self.features, self.adj)
            loss_train = F.nll_loss(output, self.labels)
            loss_train.backward()
            self.optimizer.step()

        self.GAT_model.eval()
        self.GAT_model.out_att.register_forward_hook(self.__last_gcn_layer_hook)
        self.GAT_model(self.features, self.adj)

    def __last_gcn_layer_hook(self, model, input, output):
        self.CP2Vec_node_embs = output.tolist()

        pass

    def __init_GAT_model(self, hidden_size):
        self.__load_data()

        self.GAT_model = GAT(
            hidden_size=hidden_size,
            nfeat=self.features.shape[1],
            nhid=8,
            nclass=int(self.labels.max()) + 1,
            dropout=0.6,
            nheads=8,
            alpha=0.2)

        self.optimizer = optim.Adam(self.GAT_model.parameters(), lr=0.005, weight_decay=5e-4)

    def __load_data(self):
        self.adj, self.features, self.labels = load_data(self.G, self.node_features, self.labels)
        self.adj, self.features, self.labels = Variable(self.adj), Variable(self.features), Variable(self.labels)
