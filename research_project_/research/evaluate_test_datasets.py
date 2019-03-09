from nltk.metrics.confusionmatrix import ConfusionMatrix

from research.corpus_reader import read_test_corpus, get_feature_set_for_a_data_item
from research.helper import load_classifier

__author__ = 'RAJ-PC'

import collections
import nltk.classify.util
from nltk.metrics.scores import precision, recall, f_measure


# SHUFFLE TRAIN SET
# As in cross validation, the test chunk might have only negative or only positive data


def print_test_data_set_experiment_result(classifier, test_set_feats):
    refsets = collections.defaultdict(set)
    testsets = collections.defaultdict(set)

    gold_set = []
    observed_set = []

    for j, (feats, label) in enumerate(test_set_feats):
        refsets[label].add(j)
        observed = classifier.classify(feats)
        testsets[observed].add(j)

        gold_set.append(label)
        observed_set.append(observed)

    cv_accuracy = nltk.classify.util.accuracy(classifier, test_set_feats)

    cm = ConfusionMatrix(gold_set, observed_set)
    print('----------Confusion Matrix-------')
    print(cm.pretty_format(sort_by_count=True, show_percents=False, truncate=9))
    print('---------------------------------')

    cv_x_minus_y_precision = precision(refsets['x-y'], testsets['x-y'])
    cv_x_minus_y_recall = recall(refsets['x-y'], testsets['x-y'])
    cv_x_minus_y_fmeasure = f_measure(refsets['x-y'], testsets['x-y'])

    cv_x_plus_y_precision = precision(refsets['x+y'], testsets['x+y'])
    cv_x_plus_y_recall = recall(refsets['x+y'], testsets['x+y'])
    cv_x_plus_y_fmeasure = f_measure(refsets['x+y'], testsets['x+y'])

    cv_y_minus_x_precision = precision(refsets['y-x'], testsets['y-x'])
    cv_y_minus_x_recall = recall(refsets['y-x'], testsets['y-x'])
    cv_y_minus_x_fmeasure = f_measure(refsets['y-x'], testsets['y-x'])

    cv_z_plus_y_precision = precision(refsets['y+z'], testsets['y+z'])
    cv_z_plus_y_recall = recall(refsets['y+z'], testsets['y+z'])
    cv_z_plus_y_fmeasure = f_measure(refsets['y+z'], testsets['y+z'])

    cv_y_minus_z_precision = precision(refsets['y-z'], testsets['y-z'])
    cv_y_minus_z_recall = recall(refsets['y-z'], testsets['y-z'])
    cv_y_minus_z_fmeasure = f_measure(refsets['y-z'], testsets['y-z'])

    cv_z_minus_y_precision = precision(refsets['z-y'], testsets['z-y'])
    cv_z_minus_y_recall = recall(refsets['z-y'], testsets['z-y'])
    cv_z_minus_y_fmeasure = f_measure(refsets['z-y'], testsets['z-y'])

    cv_z_plus_w_precision = precision(refsets['z+w'], testsets['z+w'])
    cv_z_plus_w_recall = recall(refsets['z+w'], testsets['z+w'])
    cv_z_plus_w_fmeasure = f_measure(refsets['z+w'], testsets['z+w'])

    cv_z_minus_w_precision = precision(refsets['z-w'], testsets['z-w'])
    cv_z_minus_w_recall = recall(refsets['z-w'], testsets['z-w'])
    cv_z_minus_w_fmeasure = f_measure(refsets['z-w'], testsets['z-w'])

    print('---------------------------------------')

    print('accuracy(10 fold):', cv_accuracy)

    print('x-y precision:', cv_x_minus_y_precision)
    print('x-y recall', cv_x_minus_y_recall)
    print('x-y f-measure', cv_x_minus_y_fmeasure)

    print('x+y precision:', cv_x_plus_y_precision)
    print('x+y recall', cv_x_plus_y_recall)
    print('x+y f-measure', cv_x_plus_y_fmeasure)

    print('y-x precision:', cv_y_minus_x_precision)
    print('y-x recall', cv_y_minus_x_recall)
    print('y-x f-measure', cv_y_minus_x_fmeasure)

    print('y+z precision:', cv_z_plus_y_precision)
    print('y+z recall', cv_z_plus_y_recall)
    print('y+z f-measure', cv_z_plus_y_fmeasure)

    print('y-z precision:', cv_y_minus_z_precision)
    print('y-z recall', cv_y_minus_z_recall)
    print('y-z f-measure', cv_y_minus_z_fmeasure)

    print('z-y precision:', cv_z_minus_y_precision)
    print('z-y recall', cv_z_minus_y_recall)
    print('z-y f-measure', cv_z_minus_y_fmeasure)

    print('z+w precision:', cv_z_plus_w_precision)
    print('z+w recall', cv_z_plus_w_recall)
    print('z+w f-measure', cv_z_plus_w_fmeasure)

    print('z-w precision:', cv_z_minus_w_precision)
    print('z-w recall', cv_z_minus_w_recall)
    print('z-w f-measure', cv_z_minus_w_fmeasure)

    print('---------------------------------------')


def get_result_for_test_data():
    corpus = read_test_corpus('Roys_dataset_test')
    # corpus = read_test_corpus('aris_dataset_test')
    test_features = ([(get_feature_set_for_a_data_item(question.split(',')[0]), question.split(',')[1].rstrip()) for question in corpus])
    classifier = load_classifier('my_classifier.pickle')
    print_test_data_set_experiment_result(classifier, test_features)

get_result_for_test_data()