import numpy as np
import pandas as pd
import pylab as plt

from collections import namedtuple

class HiggsData:
    
    def __init__(self, file_name, is_test_data=None):
        self.data = pd.read_csv(file_name)
        self.num_instances = self.data.shape[0]
        self.fraction = 0.5
        self.random_indices = None
        self.remaining_index = self.data.index
        self.is_test_data = is_test_data

        if is_test_data == None and 'test' in file_name:
            self.is_test_data = is_test_data = True

        # Add a new column to keep an integer representation of the label
        dropped_columns = ['EventId']
        if not is_test_data: 
            self.data['label_idx'] = self.data.Label.replace('b', 0).replace('s', 1).astype(np.uint8)
            dropped_columns = ['Label', 'label_idx', 'EventId', 'Weight']
        
        self.data_columns = self.data.columns.drop(dropped_columns)

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
        if not self.is_test_data:
            self.data.hist(column='Weight', bins=50, alpha=1, normed=True, by=self.data.Label)
        else:
            print "[x] Error: You're kidding me? This is the test set."

    # Get the different dataset parts
    def get_attributes(self):
        return self.data[self.data_columns]
    
    def get_weights(self):
        if not self.is_test_data:
            return self.data.Weight
        else:
            print "[x] Error: You're kidding me? This is the test set."
    
    def get_labels(self):
        if self.is_test_data:
            return self.data.label_idx

    # Splitting the dataset into train and valid
    def _compute_random_indices(self):
        self.random_indices = np.random.choice(self.data.index, int(self.fraction * self.num_instances), replace=False)

    def set_valid_fraction(self, fraction):
        self.fraction = fraction
        self._compute_random_indices()

    def get_data_fold(self, fold, normed_weights=True): 
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
        if normed_weights:
            for l_idx in (0, 1):
                W[Y == l_idx] = 0.5 * W / W[Y == l_idx].sum()

        return namedtuple('Return', 'X Y Weights')(X, Y, W)

    def get_subset(self, number, clear=False, noise=None):

        if clear: 
            self.remaining_index = self.data.index
        indices = np.random.choice(self.remaining_index, number, replace=False)
        self.remaining_index.drop(indices)

        X = self.data[self.data_columns].ix[indices] 
        # X += np.random.normal(0., 0.01, X.shape)
        W = self.data.Weight.ix[indices]
        Y = self.data.label_idx.ix[indices]

        if noise:
            X += np.random.normal(0., X.replace(-999., np.nan).std() * noise, X.shape)

        return namedtuple('Return', 'X Y Weights')(X, Y, W)
        return self.data.ix[np.random.choice(self.data.index, number, replace=False)]


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def learning_curve_plot(predictors, train_X, train_Y, valid_X, valid_Y, with_train=True, log_scale=False):
    try:
        predictors = predictors.items()
    except:
        predictors = {'': predictors}.items()

    for name, predictor in predictors:
        iterations = np.arange(1, predictor.n_estimators + 1)
        p, = plt.plot(iterations, step_wise_performance(predictor, valid_X, valid_Y), '-', label=name + ' (test)')

        if with_train:
            plt.plot(iterations, step_wise_performance(predictor, train_X, train_Y), '--', color=p.get_color(), label=name + ' (train)')

        plt.legend(loc='best')

    if log_scale: plt.gca().set_xscale('log')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def _AMS(s, b, b_regul = 10.):

### FIXME: numerical errors lead to negative s and b
### tmp modifications. 
#     assert s >= 0 and b >= 0
    if s < 0: s=0
    if b < 0: b=0
    return np.sqrt(2 * ((s + b + b_regul) * np.log(1 + s / (b + b_regul)) - s))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def plot_AMS(scores, Y, W, weight_factor):

    sorted_indices = scores[:,1].argsort()
    signal_weight_sum = W[Y == 1].sum()
    bkgd_weight_sum = W[Y == 0].sum()

    ams = np.zeros(sorted_indices.shape[0])
    max_ams = 0
    threshold = -1

    for i, current_instance in enumerate(sorted_indices):
        try:
            ams[i] = _AMS(signal_weight_sum * weight_factor, bkgd_weight_sum * weight_factor)
        except:
            # tmp code for debugging the numerical error
            print i, '/', len(sorted_indices), '|', current_instance
            print signal_weight_sum
            print bkgd_weight_sum
            raise

        if ams[i] > max_ams:
            max_ams = ams[i]
            threshold = i

        if Y[current_instance] == 1:
            signal_weight_sum -= W[current_instance]
        else:
            bkgd_weight_sum -= W[current_instance]

    plt.plot(ams)
    plt.xlim(0, len(sorted_indices)-1)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def save_model(predictor, file_name):
    import cPickle as cP
    with open(file_name, 'w') as f:
        cP.dump(predictor, f, protocol=-1)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def load_model(file_name):
    import cPickle as cP
    with open(file_name) as f:
        predictor = cP.load(f)
    return predictor

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def plot_scores(scores, labels, weights=None, **kwargs):
    if weights != None:
        s_w = weights[labels==1]
        b_w = weights[labels==0]
    else:
        s_w = b_w = None
        
    plt.hist(scores[:,1][labels==1], alpha=.5, label='signal', weights=s_w, **kwargs)
    plt.hist(scores[:,1][labels==0], alpha=.5, label='bkgd', weights=b_w, **kwargs)
    plt.legend(loc='best')
#     gca().set_yscale('log')
#     print gca().get_yscale()
#     plt.semilogy()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def plot_log(scores, labels, weights=None, **kwargs):
    if weights != None:
        s_w = weights[labels==1]
        b_w = weights[labels==0]
    else:
        s_w = b_w = None
        
    plt.hist_s, bins_s = np.histogram(scores[:,1][labels==1], weights=s_w, **kwargs)
    plt.bar(bins_s[:-1], np.log(hist_s), width=.02, color='red', alpha=.5, label='signal')
    
    hist_b, bins_b = np.histogram(scores[:,1][labels==0], weights=b_w, **kwargs)
    plt.bar(bins_b[:-1], np.log(hist_b), width=.02, alpha=.5, label='bkgd')
    legend(loc='best')
    
#     gca().set_yscale('log')
#     print gca().get_yscale()
#     plt.semilogy()



# %load_ext hierarchymagic
# with open("output.dot", "w") as output_file:
#     tree.export_graphviz(random_forest.estimators_[0]) #feature_names=vec.get_feature_names(){}
# %%dot -f svg