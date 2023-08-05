from collections import defaultdict
import numpy as np

class NB_multinomial:
    '''
    Naive Bayes classifier for multinomial models
    '''
    def __init__(self,data,alpha):
        self.alpha = 1
        self.data = data
        self.train()
    def train(self):
        '''
        Train the model
        mechanism:
        1. calculate priors
        2. calculate word liklihoods
        3. calculate word liklihoods per tag
     
        '''
        
        self.priors = {}
        total = np.sum([len(self.data[i]) for i in self.data])
        for tag in self.data:
            self.priors[tag] = len(self.data[tag])/total
        self.count_word_per_tag = defaultdict(lambda:{tag:0 for tag in self.data})
        self.total_word_count_tag = defaultdict(int)
        for tag in self.data:
            for article in self.data[tag]:
                for word in article:
                    self.count_word_per_tag[word][tag] += 1
                    self.total_word_count_tag[tag] +=1
        self.word_liklihood_tag = defaultdict(lambda:{tag:0.5 for tag in self.data})
        for word, tag_map in self.count_word_per_tag.items():
            for tag in tag_map:
                self.word_liklihood_tag[word][tag] = (self.count_word_per_tag[word][tag] + 1 * self.alpha)/(self.total_word_count_tag[tag] + 2 * self.alpha)
        return self
    
    def predict(self,article):

        '''
        Predict the tag of an article
        mechanism:
        1. calculate the numerator of the bayes formula
        2. return the tag with the highest numerator
        3. return the numerator for each tag
        '''
        
        prediction = {}
        for tag in self.data:
            numerator = np.log(self.priors[tag])
            for word in article:
                numerator += np.log(self.word_liklihood_tag.get(0.5,self.word_liklihood_tag[word][tag]))
            prediction[tag] = numerator
        return prediction, max(prediction, key=prediction.get)  