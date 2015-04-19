import pandas
import numpy

dataset = pandas.read_csv("./dataset_v6.csv")
print 'Total samples & features in original dataset: ' + str(dataset.shape)

target_label = 'Lbl: Poor Hosp Rating 1SD'
#target_label = 'Lbl: Poor ER Rating 1SD'
#target_label = 'Lbl: Poor Pain Rating 1SD'
#target_label = 'Lbl: Poor Drugs Rating 1SD'

# Strip out any rows where there is no label to train or test with
dataset = dataset[dataset[target_label].notnull()]

# keep the agency and behavior fields we're interested in (exclude things like patient outcomes, phone numbers, addresses, etc)
samples = dataset[[
    'Offers Nursing Care Services',
    'Offers Physical Therapy Services',
    'Offers Occupational Therapy Services',
    'Offers Speech Pathology Services',
    'Offers Medical Social Services',
    'Offers Home Health Aide Services',

    'How often the home health team began their patients care in a timely manner',
    'How often the home health team taught patients (or their family caregivers) about their drugs',
    'How often the home health team checked patients risk of falling',
    'How often the home health team checked patients for depression',
    'How often the home health team made sure that their patients have received a flu shot for the current flu season.',
    'How often the home health team made sure that their patients have received a pneumococcal vaccine (pneumonia shot).',
    'With diabetes - how often the home health team got doctors orders and gave foot care and taught patients about foot care',
    'How often the home health team checked patients for pain',
    'How often the home health team treated their patients pain',
    'How often the home health team took doctor-ordered action to prevent pressure sores (bed sores)',
    'How often the home health team checked patients for the risk of developing pressure sores (bed sores)',

    'Count of non-reported behaviours',
    'Count of non-reported outcomes',
    'Non-trad Chronic'
]]

# (1) reshape the label matrix to be flat
# (2) change the boolean {false, true} values into {0,1} as sklearn modules expect
labels = numpy.reshape(dataset[target_label].astype(int), -1)

print 'There are %1i samples in the dataset (having eliminated samples where the target label was null).' % (dataset.shape[0])
print 'There are %1s poorly-performing agencies labelled in this dataset (label \'%2s\').' % ((sum(p == 1 for p in labels)), target_label)

from sklearn import preprocessing
# n.b. this switches samples from a pandas DataFrame to a NumPy array
samples = preprocessing.normalize(samples.astype(float))
samples = preprocessing.scale(samples)

# put test data aside
from sklearn.cross_validation import train_test_split
samples_train, samples_test, labels_train, labels_test = train_test_split(samples,labels, test_size=0.3, random_state=2)
percent_train_positives = (100 * (sum(p == 1 for p in labels_train))) // samples_train.shape[0]
percent_test_positives = (100 * (sum(p == 1 for p in labels_test))) // samples_test.shape[0]
print 'There are %1i samples in the training dataset and %2i in the test dataset.' % (samples_train.shape[0], samples_test.shape[0])
print 'Poor outcomes make up %1i %% of the training data and %2i %% of the test dataset.' % (percent_train_positives, percent_test_positives)

print 'New Features:' + \
    '\n\tOffers Nursing Care Services' + \
    '\n\tOffers Physical Therapy Services' + \
    '\n\tOffers Occupational Therapy Services' + \
    '\n\tOffers Speech Pathology Services' + \
    '\n\tOffers Medical Social Services' + \
    '\n\tOffers Home Health Aide Services' + \
    '\n\tCount of non-reported behaviours' + \
    '\n\tCount of non-reported outcomes' + \
    '\n\tNon-trad Chronic'
