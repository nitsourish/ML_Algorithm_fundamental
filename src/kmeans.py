import random
import numpy as np

class Centroid:
    def __init__(self, location):
        self.location = location
        self.closest_users = set()


def get_k_means(user_feature_map, num_features_per_user, k):

    '''
    user_feature_map: A dictionary mapping user ids to a list of features.
    num_features_per_user: The number of features per user.
    k: The number of clusters to create.
    '''
    
    random.seed(42)
    # Gets the inital users, to be used as centroids.
    inital_centroid_users = random.sample(sorted(list(user_feature_map.keys())), k)
    d1 = list(user_feature_map.values())
    centroid_feat = np.array([user_feature_map[c] for c in inital_centroid_users])
    for j in range(200):
        assign = []
        for i in user_feature_map:
            c = np.argmin(np.sum(abs(np.array(user_feature_map[i]).reshape((1,num_features_per_user))-centroid_feat),axis=1))
            assign.append(c)   
        center = []
        for i in range(len(inital_centroid_users)):
            ce = np.mean([d1[x] for x in range(len(user_feature_map)) if assign[x]==i],axis=0)
            center.append(list(ce))
            centroid_feat = np.array(center)
    return [list(centroid_feat[i]) for i in range(k)]
