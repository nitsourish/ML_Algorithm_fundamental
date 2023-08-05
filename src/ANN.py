import numpy as np

class Neurone:
    '''
    This class implements a single neurone.
    '''
    def __init__(self,examples,n_feats=3):
        
        self.examples = examples
        self.n_feats = n_feats
        np.random.seed(100)
        self.W = np.random.normal(0,1,self.n_feats+1)
        self.train()
    
    def train(self,learning_rate=0.1, batch_size=10, epochs=5000):

        '''
        This function trains the neurone.
        mechanism:
        1) for each epoch:
        a) 'mini-batch gradient descent'
        b) for each batch:
        i) compute the gradient
        ii) update the weights
        2) return the weights

        '''
        no_batch = len(self.examples) % batch_size
        for i in range(len(self.examples)):
            self.examples[i]['feats'].append(1)

        for i in range(epochs):
            # Error = []
            Error = 0
            for batch in range(no_batch):
                grad = np.zeros(self.n_feats + 1,float) 
                minibatch = self.examples[batch * batch_size:(batch+1) * batch_size]  
                for e in minibatch:
                    raw = np.sum([e['feats'][i] * self.W[i] for i in range(self.n_feats + 1)])
                    pred = 1/(1 + np.exp(-1 * raw))
                    error = (pred - e['label'])
                    Error += np.abs(error) 
                    for j in range(4):
                        grad[j] += error * e['feats'][j]
                
                grad = grad/batch_size
                self.W = self.W - learning_rate * grad
            Error = Error/len(self.examples)
            if i % 100 == 0:
                print(np.mean(Error))
        return self.W
    
    def predict(self, features):

        '''
        This function predicts the label for a given set of features.
        '''
        
        features.append(1)
        val = np.sum(np.array(features).reshape(1,len(self.W)) * self.W)
        return np.round(1/(1+np.exp(-(val))),5)