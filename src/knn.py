import numpy as np
from collections import Counter
# Should use the `find_k_nearest_neighbors` function below.
def predict_label(examples, features, k, label_key="label"):

    '''
    examples: A dictionary mapping example ids to a dictionary of features.
    features: A list of features to use for prediction.
    k: The number of nearest neighbors to use for prediction.
    label_key: The key in the example dictionary that contains the label.
    '''
    dist = {}
    labels = {}
    for i in examples:
        dist[i] = np.sum([(examples[i]['feature'][j] - features[j])**2 for j in range(len(features))])
        labels[i] = examples[i][label_key]
    dist = {k:v for k,v in sorted(dist.items(),key=lambda x:x[1])}
    knn = list(dist.keys())[0:k]
    l = [labels[k] for k in knn]
    c = dict(Counter(l))
    c = {k:v for k,v in sorted(c.items(),key=lambda x:x[1],reverse = True)}
    return list(c.keys())[0]    


def find_k_nearest_neighbors(examples, features, k):

    '''
    examples: A dictionary mapping example ids to a dictionary of features.
    features: A list of features to use for prediction.
    k: The number of nearest neighbors to use for prediction.
    '''
    
    dist = {}
    for i in examples:
        dist[i] = np.sum([(examples[i]['feature'][j] - features[j])**2 for j in range(len(features))])
    dist = {k:v for k,v in sorted(dist.items(),key=lambda x:x[1])}
    knn = list(dist.keys())[0:k]
    return knn