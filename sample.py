from functools import partial
from sklearn import model_selection
from sklearn.metrics import make_scorer, accuracy_score, precision_score, \
recall_score, f1_score
from multiprocessing import Manager
from tqdm import tqdm

def dct_selector(attr_name):
    """ Returns a function that select from a dictionary the attribute
    having attr_name as key
    """
    return lambda dct: dct[attr_name]

def generate_pow_range(base, min, max):
    """ Returns a range of values obtain from exponentiation of base
    and between min and max (differently by range max not excluded)
    i. e. generate_pow_range(2, 3, 5)
        -> [8, 16, 32]
    """
    return map(lambda x: base**x, range(min, max + 1))

def is_list(lst):
    """ Returns true is the provided attribute is a list
    """
    return type([]) == type(lst)
    
def list_of_list(lst):
    """ Returns true is the provided attribute is a list
    of list
    """
    for e in lst:
        if not is_list(e):
            return False
    return True

def get_simple_combinations(param_lst, comb_lst=[]):
    """ Takes as argument a list of attribute lists and returns all simple
    combinations of provided attributes
    i. e. get_simple_combinations([['a', 'b'], [1, 2, 3]])
        -> [['a', 1], ['a', 2], ['a', 3], ..., ['b', 3]]
    """
    if len(param_lst) == 0:
        return comb_lst 
    res = []
    for i in param_lst[0]:
        _tmp = comb_lst[:]
        _tmp.append(i)
        comb = get_simple_combinations(param_lst[1:], _tmp)
        if list_of_list(comb):
            res.extend(comb)
        else:
            res.append(comb)
    return res

METRICS = ("accuracy", "precision", "recall", "f1-score")

SELECTOR = 'accuracy'

def get_best_params(model_func, mat, res, params_names, params_values):
    # generate all simple combinations of parameters
    combs = get_simple_combinations(params_values)

    # choose to use 10-fold cross validation
    kfold = model_selection.KFold(n_splits=10) 

    metrics_lst = []
    total_combs = len(combs)

    pbar = tqdm(total=len(combs), desc='10-fold Cross Validation', leave=False)
    for tuple in combs:
        # for all combination generated match argument name and argument value
        # and create with them a dictionary
        args = dict(zip(params_names, tuple))

        # use args dictionary as kwargs for model_func
        # i.e. for model_func: partial(svm.SVC(kernel='rbf'))
        # use the ** operator to put parameters in args as argument for
        # model_func function
        #   model = model_func(**args)
        model = model_func(**args)

        metrics = dict()
        # take metrics with cross validation
        metrics['accuracy'] = model_selection.cross_val_score(model, mat, res, cv=kfold, \
            scoring='accuracy').mean() * 100.0

        # add metrics just obtained to metrics_lst list
        metrics_lst.append(metrics)
        
        pbar.update(1)

        if metrics['accuracy'] == 100.0:
            break

    pbar.close()

    # get index of model that produced best metrics
    best = max(enumerate(metrics_lst), \
        key=lambda x: x[1]['accuracy'])
        
    # return best comb and best mestrics
    return combs[best[0]], best[1] 
