from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import auc, precision_recall_fscore_support
import numpy

def train_model(clf_factory, samples, labels, folds = 5, debug = False):
    samples_df = pandas.DataFrame(samples) # more familiar working with Pandas dataframes
    cross_validation = StratifiedKFold(labels, n_folds=folds) # Use Stratified types here due to the imbalance in the labels
    precisions, recalls, f_scores, AUCs = [], [], [], [] # capture metrics so we can calculate mean and std dev
    for train, test in cross_validation:
        samples_train, labels_train = samples_df.iloc[train], labels[train]
        samples_test, labels_test = samples_df.iloc[test], labels[test]
        clf = clf_factory()
        %time clf.fit(samples_train, labels_train)
        precision, recall, f_score, area = test_model(clf, samples_test, labels_test, debug)
        precisions.append(precision) 
        recalls.append(recall)
        AUCs.append(area)
        f_scores.append(f_score)     
    print("f_score:\t\tMean=%.5f\t\tStddev=%.5f"%(numpy.mean(f_scores), numpy.std(f_scores)))
    print("recall:\t\t\tMean=%.5f\t\tStddev=%.5f"%(numpy.mean(recalls), numpy.std(recalls)))
    print("precision:\t\tMean=%.5f\t\tStddev=%.5f"%(numpy.mean(precisions), numpy.std(precisions)))
    # print("AUC:\t\tMean=%.5f\t\tStddev=%.5f"%(numpy.mean(AUCs), numpy.std(AUCs)))

def test_model(clf, samples_test, labels_test, debug = False):
        predictions = clf.predict(samples_test)
        precision, recall, f_score, support = precision_recall_fscore_support(labels_test, predictions, beta=beta)
        area = auc(recall, precision)        
        if debug:
            print("f_score:\t\t%.5f"%(f_score[1]))  # the [1] just captures the score for our minority target label, not the majority label
            print("recall:\t\t\t%.5f"%(recall[1]))
            print("precision:\t\t%.5f"%(precision[1]))
            #print("AUC:\t\t%.5f"%(area))
        return precision[1], recall[1], f_score[1], area # just capture the score for our minority target label, not the majority label