from sklearn import metrics
from munkres import Munkres
import numpy as np


class ClusteringMetrics:
    def __init__(self, true_label, predict_label):
        self.true_label = true_label
        self.pred_label = predict_label

    def clustering_acc(self):
        # best mapping between true_label and predict label
        l1 = list(set(self.true_label))
        num_class1 = len(l1)

        l2 = list(set(self.pred_label))
        num_class2 = len(l2)
        if num_class1 != num_class2:
            print('Class Not equal, Error!!!!')
            return 0

        cost = np.zeros((num_class1, num_class2), dtype=int)
        for i, c1 in enumerate(l1):
            mps = [i1 for i1, e1 in enumerate(self.true_label) if e1 == c1]
            for j, c2 in enumerate(l2):
                mps_d = [i1 for i1 in mps if self.pred_label[i1] == c2]

                cost[i][j] = len(mps_d)

        # match two clustering results by Munkres algorithm
        m = Munkres()
        cost = cost.__neg__().tolist()

        indexes = m.compute(cost)

        # get the match results
        new_predict = np.zeros(len(self.pred_label))
        for i, c in enumerate(l1):
            # corresponding label in l2:
            c2 = l2[indexes[i][1]]

            # ai is the index with label==c2 in the pred_label list
            ai = [ind for ind, elm in enumerate(self.pred_label) if elm == c2]
            new_predict[ai] = c

        acc = metrics.accuracy_score(self.true_label, new_predict)
        f1_weighted = metrics.f1_score(self.true_label, new_predict, average='weighted')
        precision_macro = metrics.precision_score(self.true_label, new_predict, average='macro')
        recall_macro = metrics.recall_score(self.true_label, new_predict, average='macro')
        f1_micro = metrics.f1_score(self.true_label, new_predict, average='micro')
        precision_micro = metrics.precision_score(self.true_label, new_predict, average='micro')
        recall_micro = metrics.recall_score(self.true_label, new_predict, average='micro')
        return acc, f1_weighted, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro

    def evaluation_cluster_model_from_label(self):
        nmi = metrics.normalized_mutual_info_score(self.true_label, self.pred_label)
        ari = metrics.adjusted_rand_score(self.true_label, self.pred_label)
        acc, f1_weighted, _, _, f1_micro, precision_micro, recall_micro= self.clustering_acc()
        return acc, nmi, ari, f1_weighted
