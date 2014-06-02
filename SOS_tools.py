import numpy as np
import pandas as pd
import pylab as plt

from collections import namedtuple

from sklearn.utils import check_arrays, column_or_1d
from sklearn.utils.multiclass import type_of_target

pd.set_option('display.mpl_style', 'default') # Make the graphs a bit prettier
pd.set_option('display.max_columns', None) # Otherwise, the columns will be truncated
pd.set_option('display.max_rows', 35)

plt.rcParams['figure.figsize'] = (10.0, 6.0)
plt.rcParams['axes.linewidth'] = 2.5

class HiggsData:
    
    def __init__(self, file_name, is_test_data=None):
        self.data = pd.read_csv(file_name)
        self.num_instances = self.data.shape[0]
        self.split_fractions = (0.3, 0.3)
        self.random_indices = self._compute_random_indices()
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
    def _attr_hist(self, attr, **kwargs):

        plt.figure()
        alpha = kwargs.get('alpha', 0.5)
        bins = kwargs.get('bins', 100)
        normed = kwargs.get('normed', True)


        signal_data = self.data[self.data.label_idx == 1][attr].values
        bkgd_data = self.data[self.data.label_idx == 0][attr].values

        signal_weights = self.data[self.data.label_idx == 1].Weight.values
        bkgd_weights = self.data[self.data.label_idx == 0].Weight.values

        # self.data.hist(column=attr, bins=bins, alpha=alpha, normed=normed, by=self.data.Label, **kwargs)

        # self.data[self.data.label_idx == 1].hist(column=attr, bins=bins, alpha=alpha, normed=normed, label='signal', color='blue', **kwargs)
        # self.data[self.data.label_idx == 0].hist(column=attr, bins=bins, alpha=alpha, normed=normed, label='bkgd', color='red', **kwargs)

        plt.hist(signal_data, bins=bins, alpha=alpha, normed=normed, label='signal', color='blue', weights=signal_weights)
        plt.hist(bkgd_data, bins=bins, alpha=alpha, normed=normed, label='bkgd', color='red', weights=bkgd_weights)
        plt.legend(loc='best')

    def attributes_hist(self, columns=None, **kwargs):
        """
        Plot the histogram of the specified columns. If no column is specified,
        it plots all the columns. The method also accepts all matplotlib hist() parameters.

        Parameters:
        -----------
        columns: list of column names.
        """

        from IPython.display import HTML

        if columns == None: columns = self.data_columns
        else:
            if isinstance(columns, basestring):
                columns = [columns]

        for c in columns:
            self._attr_hist(c, **kwargs)
            

    def attributes_hist_grid(self):
        """
        Plot the histograms of all the columns in a grid.
        """
        self.data.hist(column=self.data_columns, bins=100, normed=True, figsize=(14,30), layout=(10, 3), weights=self.data.Weight) 
        
    def weights_hist(self):
        """
        Plot the histogram of the weights.
        """

        if not self.is_test_data:
            self.data.hist(column='Weight', bins=50, alpha=1, normed=True, by=self.data.Label)
            # self._attr_hist('Weight', alpha=1., bins=50)
        else:
            print "[x] Error: You're kidding me? This is the test set."

    # Get the different dataset parts
    def get_attributes(self):
        return self.data[self.data_columns].values
    
    def get_weights(self):
        if not self.is_test_data:
            return self.data.Weight.values
        else:
            print "[x] Error: You're kidding me? This is the test set."
    
    def get_labels(self):
        if self.is_test_data:
            return self.data.label_idx.values

    # Splitting the dataset 
    def _compute_random_indices(self):
        self.random_indices = np.random.permutation(self.data.index)
        # self.random_indices = np.random.choice(self.data.index, int(self.fraction * self.num_instances), replace=False)


    def set_split_fractions(self, fractions, ):
        """
        Set the fraction rates of the training and validation sets.

        Parameters:
        -----------
        fractions: A single real value or couple of real values. If it is a single value, the data is 
        split in two folds, train and test, and the value corresponds to the train set proportion.
        If it is a couple, the data is split in three folds, train, valid, and test. The couple of values 
        correspond then to the proportions of the train and valid data respectively.
        """

        try:
            iter(fractions)
        except TypeError:
            fractions = (fractions, 0)

        self.split_fractions = fractions
        self._compute_random_indices()


    def get_data_fold(self, fold, number=None, normed_weights=True):
        """
        Returns a named tuple (X, Y, Weights) 
        after computing the new weights
        """
        
        assert fold in ['train', 'valid', 'test'], '%s is not a correct fold name. Use either train, valid, or test.'

        if self.random_indices == None: self._compute_random_indices()

        # X = data[data_columns] # this creates a copy of the data
        # W = data[['Weight']] # force the creation of a compy 
        # Y = data.label_idx # return a view, not copies

        if fold == 'train':
            start_split = 0
            end_split = int(self.data.shape[0] * self.split_fractions[0])
        elif fold == 'valid':
            start_split = int(self.data.shape[0] * self.split_fractions[0])
            end_split = int(self.data.shape[0] * sum(self.split_fractions))
        elif fold == 'test':
            start_split = int(self.data.shape[0] * sum(self.split_fractions))
            end_split = self.data.shape[0] + 1

        indices = self.random_indices[start_split:end_split]

        if number != None and number < len(indices): 
            indices = indices[:number]

        X = self.data[self.data_columns].ix[indices]
        W = self.data.Weight.ix[indices]
        Y = self.data.label_idx.ix[indices]

        # Recalculating the weight
        if normed_weights:
            for l_idx in (0, 1):
                W[Y == l_idx] = 0.5 * W / W[Y == l_idx].sum()

        return namedtuple('Return', 'X Y Weights')(X.values, Y.values, W.values)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def _step_wise_performance(predictor, X, Y, weights=None, metric=None):
    assert len(X) == len(Y)

    if hasattr(weights, 'values'): weights = weights.values

    num_iterations = predictor.n_estimators
    if metric == None:
        metric = zero_one_loss

    stage_wise_perf = np.zeros(num_iterations)
    if hasattr(predictor, 'staged_predict'):
        # stage_wise_perf = list(predictor.staged_score(X, Y))
        for i, pred in enumerate(predictor.staged_predict(X)):
            stage_wise_perf[i] = metric(Y, pred, sample_weight=weights)

    else:
        prediction = np.zeros(X.shape[0])
        for i, t in enumerate(predictor.estimators_):
            prediction += t.predict(X)
            stage_wise_perf[i] = metric(Y, np.round( prediction / (i+1))) #, sample_weight=weights

    return stage_wise_perf 

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def learning_curve_plot(predictors, valid_X, valid_Y, valid_W=None, train_X=None, train_Y=None, train_W=None, log_scale=False):

    if hasattr(valid_W, 'values'): valid_W = valid_W.values
    if train_W != None and hasattr(train_W, 'values'): train_W = weights.values

    with_train = True if (train_X != None and valid_X != None) else False

    try:
        predictors = predictors.items()
    except:
        predictors = {'': predictors}.items()

    for name, predictor in predictors:
        iterations = np.arange(1, predictor.n_estimators + 1)
        p, = plt.plot(iterations, _step_wise_performance(predictor, valid_X, valid_Y, valid_W), '-', label=name + ' (test)')

        if with_train:
            plt.plot(iterations, _step_wise_performance(predictor, train_X, train_Y, train_W), '--', color=p.get_color(), label=name + ' (train)')

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

def plot_AMS(scores, labels, weights, weight_factor, **kwargs):
    """
    Compute the AMS for all possible decision thresholds and plot the 
    corresponding curve. 

    Returns the couple (best_AMS, threshold)
    """

    if hasattr(scores, 'values'): scores = scores.values
    if hasattr(labels, 'values'): labels = labels.values
    if hasattr(weights, 'values'): weights = weights.values

    sorted_indices = scores[:,1].argsort()
    signal_weight_sum = weights[labels == 1].sum()
    bkgd_weight_sum = weights[labels == 0].sum()

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

        if labels[current_instance] == 1:
            signal_weight_sum -= weights[current_instance]
        else:
            bkgd_weight_sum -= weights[current_instance]

    plt.plot(ams, **kwargs)
    if 'label' in kwargs: plt.legend(loc='best')

    plt.xlim(0, len(sorted_indices)-1)

    # print "[+] Best AMS:", max_ams
    return namedtuple('Return', 'best_AMS threshold')(max_ams, threshold)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def plot_AMS_2(scores, labels, weights, weight_factor, plot=True, **kwargs):
    """
    Compute the AMS for all possible decision thresholds and plot the 
    corresponding curve. 

    Returns the couple (best_AMS, threshold)
    """

    if scores.ndim == 2: 
        scores = scores[:,1]

    if hasattr(scores, 'values'): scores = scores.values
    if hasattr(labels, 'values'): labels = labels.values
    if hasattr(weights, 'values'): weights = weights.values

    sorted_indices = scores.argsort()
    sorted_scores = scores[sorted_indices]
    signal_weight_sum = weights[labels == 1].sum()
    bkgd_weight_sum = weights[labels == 0].sum()

    ams = [] #np.zeros(sorted_indices.shape[0])
    max_ams = 0
    last_threshold = threshold = scores[sorted_indices][0] - 0.0001

    for current_instance, s in zip(sorted_indices, sorted_scores):
        current_ams = _AMS(signal_weight_sum * weight_factor, bkgd_weight_sum * weight_factor)
        ams.append(current_ams)
        if current_ams > max_ams : #and last_threshold != threshold
            max_ams = current_ams
            last_threshold = threshold
            threshold = s

        if labels[current_instance] == 1:
            signal_weight_sum -= weights[current_instance]
        else:
            bkgd_weight_sum -= weights[current_instance]

    mid_threshold = (threshold + last_threshold) / 2.

    if plot:
        plt.plot(sorted_scores, ams, **kwargs)
        plt.axvline(mid_threshold, color='black', linewidth=1)
        if 'label' in kwargs: plt.legend(loc='best')
    
    return namedtuple('Return', 'best_AMS threshold')(max_ams, mid_threshold)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def hist_scores(scores, labels, weights=None, **kwargs):

    if hasattr(scores, 'values'): scores = scores.values
    if hasattr(labels, 'values'): labels = labels.values
    if hasattr(weights, 'values'): weights = weights.values

    if weights != None:
        s_w = weights[labels==1]
        b_w = weights[labels==0]
    else:
        s_w = b_w = None
    
    kwargs['bins'] = kwargs.get('bins', 100)
    kwargs['normed'] = kwargs.get('normed', True)
    kwargs['histtype'] = kwargs.get('histtype', 'stepfilled')

    if scores.ndim == 2: 
        scores = scores[:,1]

    plt.hist(scores[labels==1], alpha=.5, label='signal', weights=s_w, color='blue', **kwargs)
    plt.hist(scores[labels==0], alpha=.5, label='bkgd', weights=b_w, color='red', **kwargs)
    plt.legend(loc='best')

    # plt.xlim((0., 1.))
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def hist_scores_log(scores, labels, weights=None, ln=True, **kwargs): 

    if hasattr(scores, 'values'): scores = scores.values
    if hasattr(labels, 'values'): labels = labels.values
    if hasattr(weights, 'values'): weights = weights.values

    if weights != None:
        s_w = weights[labels==1]
        b_w = weights[labels==0]
    else:
        s_w = b_w = None
        
    kwargs['bins'] = kwargs.get('bins', 100)
    kwargs['normed'] = kwargs.get('normed', True)

    # weight_factor = None
    # if 'weight_factor' in kwargs:
    #     weight_factor = kwargs['weight_factor']
    #     del kwargs['weight_factor']

    if scores.ndim == 2: 
        scores = scores[:,1]
    hist_s, bins_s = np.histogram(scores[labels==1], weights=s_w, **kwargs)
    hist_b, bins_b = np.histogram(scores[labels==0], weights=b_w, **kwargs)

    if ln:
        nz_s = hist_s.nonzero()[0]
        hist_s[nz_s] = np.log(hist_s[nz_s])
        nz_b = hist_b.nonzero()[0]
        hist_b[nz_b] = np.log(hist_b[nz_b])
        hist_min = min(hist_s[nz_s].min(), hist_b[nz_b].min())
        hist_s[nz_s] -= hist_min 
        hist_b[nz_b] -= hist_min 


    width =  0.99 * (max(bins_s.max(), bins_b.max()) - min(bins_s.min(), bins_b.min()))  / kwargs['bins'] 
    
    plt.bar(bins_s[:-1], hist_s, width=width, color='blue', alpha=.5, label='signal')
    plt.bar(bins_b[:-1], hist_b, width=width, color='red', alpha=.5, label='bkgd')

    plt.legend(loc='best')
    # plt.xlim((0., 1.))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def generate_submission_file(file_name, predictor, higgs_data, threshold):
    
    data = higgs_data.get_attributes()
    scores = predictor.predict_proba(data)[:, 1]
    ranks = np.argsort(np.argsort(scores))
    indices = higgs_data.EventId.values

    predictions = map(lambda x: 'b' if x else 's', (scores > threshold).astype(int)) 
    
    with open(file_name, 'w') as f:
        f.write('EventId,RankOrder,Class\n')
        np.savetxt(f, zip(indices, ranks, predictions), delimiter=',', fmt='%s')

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

def zero_one_loss(y_true, y_pred, normalize=True, sample_weight=None):
    """Zero-one classification loss.

If normalize is ``True``, return the fraction of misclassifications
(float), else it returns the number of misclassifications (int). The best
performance is 0.

Parameters
----------
y_true : array-like or list of labels or label indicator matrix
Ground truth (correct) labels.

y_pred : array-like or list of labels or label indicator matrix
Predicted labels, as returned by a classifier.

normalize : bool, optional (default=True)
If ``False``, return the number of misclassifications.
Otherwise, return the fraction of misclassifications.

sample_weight : array-like of shape = [n_samples], optional
Sample weights.

Returns
-------
loss : float or int,
If ``normalize == True``, return the fraction of misclassifications
(float), else it returns the number of misclassifications (int).

Notes
-----
In multilabel classification, the zero_one_loss function corresponds to
the subset zero-one loss: for each sample, the entire set of labels must be
correctly predicted, otherwise the loss for that sample is equal to one.

See also
--------
accuracy_score, hamming_loss, jaccard_similarity_score

Examples
--------
>>> from sklearn.metrics import zero_one_loss
>>> y_pred = [1, 2, 3, 4]
>>> y_true = [2, 2, 3, 4]
>>> zero_one_loss(y_true, y_pred)
0.25
>>> zero_one_loss(y_true, y_pred, normalize=False)
1

In the multilabel case with binary indicator format:

>>> zero_one_loss(np.array([[0.0, 1.0], [1.0, 1.0]]), np.ones((2, 2)))
0.5

and with a list of labels format:

>>> zero_one_loss([(1, ), (3, )], [(1, 2), tuple()])
1.0


"""
    score = accuracy_score(y_true, y_pred,
                           normalize=normalize,
                           sample_weight=sample_weight)

    if normalize:
        return 1 - score
    else:
        if sample_weight is not None:
            n_samples = np.sum(sample_weight)
        else:
            n_samples = len(y_true)
        return n_samples - score

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def accuracy_score(y_true, y_pred, normalize=True, sample_weight=None):
    """Accuracy classification score.

    In multilabel classification, this function computes subset accuracy:
    the set of labels predicted for a sample must *exactly* match the
    corresponding set of labels in y_true.

    Parameters
    ----------
    y_true : array-like or list of labels or label indicator matrix
    Ground truth (correct) labels.

    y_pred : array-like or list of labels or label indicator matrix
    Predicted labels, as returned by a classifier.

    normalize : bool, optional (default=True)
    If ``False``, return the number of correctly classified samples.
    Otherwise, return the fraction of correctly classified samples.

    sample_weight : array-like of shape = [n_samples], optional
    Sample weights.

    Returns
    -------
    score : float
    If ``normalize == True``, return the correctly classified samples
    (float), else it returns the number of correctly classified samples
    (int).

    The best performance is 1 with ``normalize == True`` and the number
    of samples with ``normalize == False``.

    See also
    --------
    jaccard_similarity_score, hamming_loss, zero_one_loss

    Notes
    -----
    In binary and multiclass classification, this function is equal
    to the ``jaccard_similarity_score`` function.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.metrics import accuracy_score
    >>> y_pred = [0, 2, 1, 3]
    >>> y_true = [0, 1, 2, 3]
    >>> accuracy_score(y_true, y_pred)
    0.5
    >>> accuracy_score(y_true, y_pred, normalize=False)
    2

    In the multilabel case with binary indicator format:

    >>> accuracy_score(np.array([[0.0, 1.0], [1.0, 1.0]]), np.ones((2, 2)))
    0.5

    and with a list of labels format:

    >>> accuracy_score([(1, ), (3, )], [(1, 2), tuple()])
    0.0

    """

    # Compute accuracy for each possible representation
    y_type, y_true, y_pred = _check_clf_targets(y_true, y_pred)
    if y_type == 'multilabel-indicator':
        score = (y_pred != y_true).sum(axis=1) == 0
    elif y_type == 'multilabel-sequences':
        score = np.array([len(set(true) ^ set(pred)) == 0
                          for pred, true in zip(y_pred, y_true)])
    else:
        score = y_true == y_pred

    if normalize:
        if sample_weight is not None:
            return np.average(score, weights=sample_weight)
        return np.mean(score)
    else:
        if sample_weight is not None:
            return np.dot(score, sample_weight)
        return np.sum(score)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def _check_clf_targets(y_true, y_pred):
    """Check that y_true and y_pred belong to the same classification task

    This converts multiclass or binary types to a common shape, and raises a
    ValueError for a mix of multilabel and multiclass targets, a mix of
    multilabel formats, for the presence of continuous-valued or multioutput
    targets, or for targets of different lengths.

    Column vectors are squeezed to 1d.

    Parameters
    ----------
    y_true : array-like,

    y_pred : array-like

    Returns
    -------
    type_true : one of {'multilabel-indicator', 'multilabel-sequences', \
    'multiclass', 'binary'}
    The type of the true target data, as output by
    ``utils.multiclass.type_of_target``

    y_true : array or indicator matrix or sequence of sequences

    y_pred : array or indicator matrix or sequence of sequences
    """

    y_true, y_pred = check_arrays(y_true, y_pred, allow_lists=True)
    type_true = type_of_target(y_true)
    type_pred = type_of_target(y_pred)

    y_type = set([type_true, type_pred])
    if y_type == set(["binary", "multiclass"]):
        y_type = set(["multiclass"])

    if len(y_type) > 1:
        raise ValueError("Can't handle mix of {0} and {1}"
                         "".format(type_true, type_pred))

    # We can't have more than one value on y_type => The set is no more needed
    y_type = y_type.pop()

    # No metrics support "multiclass-multioutput" format
    if (y_type not in ["binary", "multiclass", "multilabel-indicator",
                       "multilabel-sequences"]):
        raise ValueError("{0} is not supported".format(y_type))

    if y_type in ["binary", "multiclass"]:
        y_true = column_or_1d(y_true)
        y_pred = column_or_1d(y_pred)

    return y_type, y_true, y_pred

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# output with weights 

# %load_ext hierarchymagic
# with open("output.dot", "w") as output_file:
#     tree.export_graphviz(random_forest.estimators_[0]) #feature_names=vec.get_feature_names(){}
# %%dot -f svg
