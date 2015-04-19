from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import fbeta_score, make_scorer

# favour precision over recall, but use both
beta = 0.25
scoring = make_scorer(fbeta_score, beta=beta) 
#scoring = 'precision' # same as f1
#beta = 1              # same as f1

def grid_search(clf_factory, grid_hyperparameters, samples, labels, folds = 5, scoring=scoring):
    cross_val = StratifiedKFold(labels, n_folds=folds) # Use Stratified types here due to the imbalance in the labels
    clf = clf_factory()
    grid_search_with_cross_val = GridSearchCV(clf, grid_hyperparameters, scoring=scoring, cv=cross_val, n_jobs=-1)
    grid_search_with_cross_val.fit(samples, labels)
    best_estimator = grid_search_with_cross_val.best_estimator_
    print 'Best combination of hyperparameters for this classifier:'
    return best_estimator
