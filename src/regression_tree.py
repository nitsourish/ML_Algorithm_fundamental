#prepare this as a package
# from .regression_tree import RegressionTree
#
import pandas as pd
import numpy as np
class TreeNode:
    '''
    This class represents a node in the regression tree.
    '''
    def __init__(self, examples):
        self.examples = examples
        self.left = None
        self.right = None
        self.split_point = None

    def split(self):

        '''
        This function should split the current node into two children nodes: self.left and self.right.
        The split should be done on the feature that results in the lowest mean squared error (MSE).
        The split point should be the average of the values of the feature that minimizes the MSE.
        '''
        if len(self.examples) == 1:
            return
        best_split = {'feature':None,'value':None,'split_index':None,'mse':100000}    
        for feat in list(self.examples[0].keys())[:-1]:
            print(feat)
            self.examples.sort(key = lambda example:example[feat])
            for i,_ in enumerate(self.examples[:-1]):
                feat_val = (self.examples[i][feat] + self.examples[i+1][feat])/2
                bst_mse,bst_index = self.mse_split(feat,feat_val)
                if best_split['mse'] > bst_mse:
                    best_split = {'feature':feat,'value':feat_val,'split_index':bst_index,'mse':bst_mse}
        print(best_split)
        self.split_point = best_split
        self.examples.sort(key = lambda example:example[self.split_point ['feature']])
        self.left = TreeNode(self.examples[:self.split_point['split_index']])
        # print(self.left)
        self.left.split()
        self.right = TreeNode(self.examples[self.split_point['split_index']:])
        # print(self.right)
        self.right.split()
    
    def mse_split(self,feat,feat_val):

        '''
        This function should return the MSE of a split on the given feature at the given value.
        '''
        
        left_bpds = [example['target'] for example in self.examples if example[feat] <= feat_val]                 
        split_id = len(left_bpds)
        right_bpds = [example['target'] for example in self.examples if example[feat] > feat_val]
        if not len(left_bpds) or not len(right_bpds):
            return 10,1
        left_mean,right_mean = np.mean(left_bpds),np.mean(right_bpds)
        left_mse = np.sum([(left_bpds[j]-left_mean)**2 for j in range(len(left_bpds))])/len(left_bpds)
        right_mse = np.sum([(right_bpds[j]-right_mean)**2 for j in range(len(right_bpds))])/len(right_bpds)
        total_mse = (len(left_bpds) * left_mse + len(right_bpds) * right_mse)/(len(right_bpds) + len(left_bpds))
        return total_mse,split_id

class RegressionTree:
    def __init__(self, examples):

        self.root = TreeNode(examples)
        self.train()

    def train(self):
        '''
        This function should train the regression tree using the given training examples.
        '''
        self.root.split()

    def predict(self, example):
        '''
        This function should return the prediction given by the regression tree for a single example.
        '''
        node = self.root
        while node.left and node.right:
            if example[node.split_point['feature']] <= node.split_point['value']:
                node = node.left
            else:
                node = node.right    
        val = sum([leaf['target'] for leaf in node.examples])/len(node.examples)
        return val