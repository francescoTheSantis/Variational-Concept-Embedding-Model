import torch
from torchmetrics import Metric
from sklearn.metrics import f1_score, accuracy_score
import scipy
import sklearn
import numpy as np
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import homogeneity_score
from tqdm import tqdm 

class Task_Accuracy(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds = torch.argmax(preds, dim=1)
        assert preds.shape == target.shape
        self.correct += torch.sum(preds == target)
        self.total += target.numel()

    def compute(self):
        return self.correct.float() / self.total
    

class Concept_Accuracy(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds = torch.where(preds > 0.5, 1, 0)
        assert preds.shape == target.shape
        self.correct += torch.sum(preds == target)
        self.total += target.shape[0] * target.shape[1]

    def compute(self):
        return self.correct.float() / self.total
    

def f1_acc_metrics(y_true, y_pred):
    # Convert PyTorch tensors to lists if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy().tolist()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy().tolist()
    
    # Calculate the F1 score
    f1 = f1_score(y_true, y_pred, average='macro')
    # Calculate the accuracy
    accuracy = accuracy_score(y_true, y_pred)
    return f1, accuracy


def concept_alignment_score(
    c_vec,
    c_test,
    y_test,
    step=5,
    force_alignment=False,
    alignment=None,
    progress_bar=True,
):
    """
    Computes the concept alignment score between learnt concepts and labels.

    :param c_vec: predicted concept representations (can be concept embeddings)
    :param c_test: concept ground truth labels
    :param y_test: task ground truth labels
    :param step: number of integration steps
    :return: concept alignment AUC, task alignment AUC
    """

    # compute the maximum value for the AUC
    n_clusters = np.linspace(
        2,
        c_vec.shape[0],
        step,
    ).astype(int)
    max_auc = np.trapz(np.ones(len(n_clusters)))

    # for each concept:
    #   1. find clusters
    #   2. compare cluster assignments with ground truth concept/task labels
    concept_auc, task_auc = [], []
    for concept_id in range(c_test.shape[1]):
        concept_homogeneity, task_homogeneity = [], []
        for nc in tqdm(n_clusters):
            kmedoids = KMedoids(n_clusters=nc, random_state=0)
            if c_vec.shape[1] != c_test.shape[1]:
                c_cluster_labels = kmedoids.fit_predict(
                    np.hstack([
                        c_vec[:, concept_id][:, np.newaxis],
                        c_vec[:, c_test.shape[1]:]
                    ])
                )
            elif c_vec.shape[1] == c_test.shape[1] and len(c_vec.shape) == 2:
                c_cluster_labels = kmedoids.fit_predict(
                    c_vec[:, concept_id].reshape(-1, 1)
                )
            else:
                c_cluster_labels = kmedoids.fit_predict(c_vec[:, concept_id, :])

            # compute alignment with ground truth labels
            concept_homogeneity.append(
                homogeneity_score(c_test[:, concept_id], c_cluster_labels)
            )
            task_homogeneity.append(
                homogeneity_score(y_test, c_cluster_labels)
            )
        # compute the area under the curve
        concept_auc.append(np.trapz(np.array(concept_homogeneity)) / max_auc)
        task_auc.append(np.trapz(np.array(task_homogeneity)) / max_auc)
    # return the average alignment across all concepts
    concept_auc = np.mean(concept_auc)
    task_auc = np.mean(task_auc)
    if force_alignment:
        return concept_auc, task_auc, alignment
    return concept_auc, task_auc
