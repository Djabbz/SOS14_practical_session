import numpy as np
import pandas as pd

class HiggsData:
    
    def __init__(self, file_name):
        self.data = pd.read_csv(file_name)
        self.num_instances = self.data.shape[0]
        self.fraction = 1.
        self.random_indices = None
        
        # Add a new column to keep an integer representation of the label
        self.data['label_idx'] = self.data.Label.replace('b', 0).replace('s', 1).astype(np.uint8)
        self.data_columns = self.data.columns.drop(['Label', 'label_idx', 'EventId', 'Weight'])

    # delegate to the DataFrame
    def __getattr__(self, attrname):
        try:
            return getattr(self.wrapped, attrname)
        except: 
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

    def _compute_random_indices(self):
        if self.random_indices == None: 
            self.random_indices = np.random.choice(self.data.index, int(self.fraction * self.num_instances), replace=False)

    def get_data_fold(self, fold): 
        """
        Returns a named tuple (X, Y, Weights) 
        after computing the new weights
        """
        from collections import namedtuple
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


    def split_weights(indices):
        pass
            

def Calculate_AMS(s, b):
    assert s >= 0
    assert b >= 0
    b_regul = 10.
    return np.sqrt(2 * ((s + b + b_regul) * np.log(1 + s / (b + b_regul)) - s))