import pytest

from src.regression_tree import RegressionTree
from src.kmeans import get_k_means
from src.knn import predict_label, find_k_nearest_neighbors
from src.NB_multinomial import NB_multinomial
from src.ANN import Neurone
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing

@pytest.fixture

def input_regression_tree():
  df = {'MedInc': 3.8462,
    'HouseAge': 52.0,
    'AveRooms': 6.281853,
    'AveBedrms': 1.081081,
    'Population': 565.0,
    'AveOccup': 2.181467,
    'Latitude': 37.85,
    'Longitude': -122.25}
  return df


def test_regression_tree(input_regression_tree):

  '''
  for this test, we will use the california housing dataset from sklearn
  '''
  data = pd.DataFrame(np.concatenate([fetch_california_housing()['data'],fetch_california_housing()['target'].reshape(-1,1)],axis=1), columns=['MedInc',
  'HouseAge',
  'AveRooms',
  'AveBedrms',
  'Population',
  'AveOccup',
  'Latitude',
  'Longitude',
  'target']).to_dict('records')
  data = data[0:100]
  tree = RegressionTree(data)
  assert tree.predict(input_regression_tree) == 3.422  # This is the correct value for the input value above.

@pytest.fixture
def input_value_kmeans():
  df = {'pid_1':[1.2,1.3],'pid_2':[2.42,1.99]}
  return df


def test_kmeans(input_value_kmeans):
  assert get_k_means(input_value_kmeans, 2, k=1)  == [[1.81, 1.645]]

@pytest.fixture
def input_value_knn():
  d = {'pid_1':{'feature':[1.2,1.3], 'label':0},'pid_4':{'feature':[2.42,1.99], 'label':1},'pid_3':{'feature':[2.4,3.3], 'label':1},'pid_2':{'feature':[1.52,1.36], 'label':0},'pid_5':{'feature':[1.02,1.73],'label':0}}
  return d

def test_knn(input_value_knn):
  feature = [2.4,3.5]
  assert find_k_nearest_neighbors(input_value_knn,feature, k=1)  == ['pid_3']
  assert predict_label(input_value_knn, feature, k=1,label_key='label') == 1

@pytest.fixture
def input_nb_multinomial():
  data = {'sports':[['the','team','played','a','great','game'],['the','game','was','awesome'],['i','love','baseball'],['i','hate','tennis']],
        'not_sports':[['i','love','my','dog'],['my','dog','hates','me'],['the','cat','scratched','me'],['i','hate','cats']]}  
  
  return data

def test_nb_multinomial(input_nb_multinomial):
  nb = NB_multinomial(input_nb_multinomial,1)
  article = ['baseball','is','tough','game']
  assert nb.predict(article)[1] == 'sports'
  assert nb.predict(article)[0] == {'sports': -6.068425588244111, 'not_sports': -7.7458682297922685}  

@pytest.fixture
def input_ann():
  data = [{'feats':[0.1,0.11,0.3],'label':0},{'feats':[0.19,0.51,0.39],'label':1},{'feats':[0.01,0.18,0.23],'label':0},{'feats':[0.31,0.411,0.43],'label':1},{'feats':[0.51,0.611,0.35],'label':1},{'feats':[0.11,0.211,0.035],'label':0}]  
  
  return data

def test_neurone(input_ann):
  N = Neurone(input_ann)
  N.train()
  feature = [[0.72,0.81,0.71],[0.12,0.11,0.41]]
  assert N.predict(feature[0]) == 0.99999
  assert N.predict(feature[1]) == 0.14156
