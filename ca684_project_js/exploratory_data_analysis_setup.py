import pandas
dataset = pandas.read_csv("./dataset_v5.csv")

# Select the target lable and strip out any rows where there is no label to train or test with
target_label = 'Lbl: Poor Hosp Rating 1SD'
dataset = dataset[dataset[target_label].notnull()]

samples = dataset[[ # keep the agency behavior fields we're interested in
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
    'How often the home health team checked patients for the risk of developing pressure sores (bed sores)']]

# (1) reshape the label matrix to be flat and (2) change the boolean {false, true} values into {0,1} as sklearn modules expect
import numpy as np
from sklearn import preprocessing
labels = np.reshape(dataset[target_label].astype(int), -1)
samples = preprocessing.normalize(samples.astype(float))
samples = preprocessing.scale(samples)
