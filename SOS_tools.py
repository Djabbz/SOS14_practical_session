import numpy as np
import pandas as pd
import pylab as plt

from collections import namedtuple

class HiggsData:
    
    def __init__(self, file_name):
        self.data = pd.read_csv(file_name)
        self.num_instances = self.data.shape[0]
        self.fraction = 0.5
        self.random_indices = None
        
        # Add a new column to keep an integer representation of the label
        self.data['label_idx'] = self.data.Label.replace('b', 0).replace('s', 1).astype(np.uint8)
        self.data_columns = self.data.columns.drop(['Label', 'label_idx', 'EventId', 'Weight'])

    # delegate to the DataFrame
    def __getattr__(self, attrname):
        try:
            return getattr(self.wrapped, attrname)
        except:
            if attrname[:8] == 'cleaned_':
                return getattr(self.data.replace(-999., np.nan), attrname[8:])
            else:
                return getattr(self.data, attrname)
                

    # Plotting functions
    def attributes_hist(self):
        self.data[self.data_columns].hist(bins=100, normed=True, figsize=(14,30), layout=(10, 3), weights=self.data.Weight) 
        
    def weights_hist(self):
        self.data.hist(column='Weight', bins=50, alpha=1, normed=True, by=self.data.Label) 
        

    # Get the different dataset parts
    def get_attributes(self):
        return self.data[self.data_columns]
    
    def get_weights(self):
        return self.data.Weight
    
    def get_labels(self):
        return self.data.label_idx
    
    # Splitting the dataset into train and valid

    def set_valid_fraction(self, fraction):
        self.fraction = fraction
        self._compute_random_indices()

    def _compute_random_indices(self):
        self.random_indices = np.random.choice(self.data.index, int(self.fraction * self.num_instances), replace=False)

    def get_data_fold(self, fold): 
        """
        Returns a named tuple (X, Y, Weights) 
        after computing the new weights
        """
        
        assert fold in ['train', 'valid'], '%s is not a correct fold name. Use either train or valid.'

        if self.random_indices == None: self._compute_random_indices()

        # X = data[data_columns] # this creates a copy of the data
        # W = data[['Weight']] # force the creation of a compy 
        # Y = data.label_idx # return a view, not copies

        indices = self.random_indices
        if fold == 'train':
            indices = self.data.index.drop(indices)

        X = self.data[self.data_columns].ix[indices]
        W = self.data.Weight.ix[indices]
        Y = self.data.label_idx.ix[indices]

        # Recalculating the weight
        for l_idx in (0, 1):
            W[Y == l_idx] = 0.5 * W / W[Y == l_idx].sum()

        return namedtuple('Return', 'X Y Weights')(X, Y, W)


    def generate_noisy_subset(self, number, noise):
        pass
        


    def get_subset(self, number, noise=None):
        indices = np.random.choice(self.data.index, number, replace=False)
        
        X = self.data[self.data_columns].ix[indices] 
        # X += np.random.normal(0., 0.01, X.shape)
        W = self.data.Weight.ix[indices]
        Y = self.data.label_idx.ix[indices]

        if noise:
            X += np.random.normal(0., X.replace(-999., np.nan).std() * noise, X.shape)

        return namedtuple('Return', 'X Y Weights')(X, Y, W)
        return self.data.ix[np.random.choice(self.data.index, number, replace=False)]


def step_wise_performance(predictor, X, Y, metric=None):
    assert len(X) == len(Y)

    num_iterations = predictor.n_estimators
    if metric == None:
        from sklearn.metrics import zero_one_loss
        metric = zero_one_loss

    stage_wise_perf = np.zeros(num_iterations)
    if hasattr(predictor, 'staged_predict'):
        # stage_wise_perf = list(predictor.staged_score(X, Y))
        for i, pred in enumerate(predictor.staged_predict(X)):
            stage_wise_perf[i] = metric(Y, pred)

    else:
        prediction = np.zeros(X.shape[0])
        for i, t in enumerate(predictor.estimators_):
            prediction += t.predict(X)
            stage_wise_perf[i] = metric(Y, np.round( prediction / (i+1)))

    return stage_wise_perf 

def Calculate_AMS(s, b):
    assert s >= 0
    assert b >= 0
    b_regul = 10.
    return np.sqrt(2 * ((s + b + b_regul) * np.log(1 + s / (b + b_regul)) - s))