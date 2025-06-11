
import numpy as np
from sklearn.metrics import confusion_matrix


def aca(output, target):

    # Confusion matrix
    cm = confusion_matrix(target, np.argmax(output, -1))
    cm_norm = (cm / np.expand_dims(np.sum(cm, -1), 1))

    # Accuracy per class - and average
    aca = np.round(np.mean(np.diag(cm_norm) * 100), 2)

    return aca


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].float().sum()
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
