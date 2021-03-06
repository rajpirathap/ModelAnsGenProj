from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm.classes import SVC
from research.helper import get_percentage_train_and_test_data_set, get_train_features, save_classifier
from research.voted_classifier import VoteClassifier
import random
import collections
from nltk.metrics.confusionmatrix import ConfusionMatrix
from nltk.metrics.scores import precision, recall, f_measure

import nltk


def print_and_get_split_dataset_accuracy(test_classifiers, train_features):
    training_set, testing_set = get_percentage_train_and_test_data_set(train_features)

    classifiers = []

    for classifier in test_classifiers:
        classifier = classifier.train(training_set)
        classifiers.append(classifier)

    voted_classifier = VoteClassifier(classifiers)
    #save_classifier(voted_classifier, 'voted_classifier.pickle')
    print("voted_classifier accuracy percent:", (nltk.classify.accuracy(voted_classifier, testing_set))*100)


def print_voted_classifier_cross_validation_experiment_result(test_classifiers, trainfeats):
    random.shuffle(trainfeats)
    random.shuffle(trainfeats)
    n = 10
    subset_size = len(trainfeats) / n
    accuracy = []

    x_minus_y_precision = []
    x_minus_y_recall = []
    x_minus_y_fmeasure = []

    x_plus_y_precision = []
    x_plus_y_recall = []
    x_plus_y_fmeasure = []

    y_minus_x_precision = []
    y_minus_x_recall = []
    y_minus_x_fmeasure = []

    z_minus_y_precision = []
    z_minus_y_recall = []
    z_minus_y_fmeasure = []

    y_plus_z_precision = []
    y_plus_z_recall = []
    y_plus_z_fmeasure = []

    y_minus_z_precision = []
    y_minus_z_recall = []
    y_minus_z_fmeasure = []

    z_plus_w_precision = []
    z_plus_w_recall = []
    z_plus_w_fmeasure = []

    z_minus_w_precision = []
    z_minus_w_recall = []
    z_minus_w_fmeasure = []


    for i in range(n):
        testing_this_round = trainfeats[int(i * subset_size):][:int(subset_size)]
        training_this_round = trainfeats[:int(i * subset_size)] + trainfeats[int((i + 1) * subset_size):]
        trained_classifiers = []
        for test_classifier in test_classifiers:
            classifier = test_classifier.train(training_this_round)
            trained_classifiers.append(classifier)

        voted_classifier = VoteClassifier(trained_classifiers)
        refsets = collections.defaultdict(set)
        testsets = collections.defaultdict(set)

        gold_set = []
        observed_set = []

        for j, (feats, label) in enumerate(testing_this_round):
            refsets[label].add(j)
            observed = voted_classifier.classify(feats)
            testsets[observed].add(j)

            gold_set.append(label)
            observed_set.append(observed)

        cv_accuracy = nltk.classify.util.accuracy(voted_classifier, testing_this_round)

        cm = ConfusionMatrix(gold_set, observed_set)
        print('----------Confusion Matrix-------round {}------'.format(i))
        print(cm.pretty_format(sort_by_count=True, show_percents=False, truncate=9))
        print('---------------------------------------')

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

        accuracy.append(cv_accuracy)
        x_minus_y_precision.append(cv_x_minus_y_precision)
        x_minus_y_recall.append(cv_x_minus_y_recall)
        x_minus_y_fmeasure.append(cv_x_minus_y_fmeasure)

        x_plus_y_precision.append(cv_x_plus_y_precision)
        x_plus_y_recall.append(cv_x_plus_y_recall)
        x_plus_y_fmeasure.append(cv_x_plus_y_fmeasure)

        y_minus_x_precision.append(cv_y_minus_x_precision)
        y_minus_x_recall.append(cv_y_minus_x_recall)
        y_minus_x_fmeasure.append(cv_y_minus_x_fmeasure)

        y_plus_z_precision.append(cv_z_plus_y_precision)
        y_plus_z_recall.append(cv_z_plus_y_recall)
        y_plus_z_fmeasure.append(cv_z_plus_y_fmeasure)

        y_minus_z_precision.append(cv_y_minus_z_precision)
        y_minus_z_recall.append(cv_y_minus_z_recall)
        y_minus_z_fmeasure.append(cv_y_minus_z_fmeasure)

        z_minus_y_precision.append(cv_z_minus_y_precision)
        z_minus_y_recall.append(cv_z_minus_y_recall)
        z_minus_y_fmeasure.append(cv_z_minus_y_fmeasure)

        z_plus_w_precision.append(cv_z_plus_w_precision)
        z_plus_w_recall.append(cv_z_plus_w_recall)
        z_plus_w_fmeasure.append(cv_z_plus_w_fmeasure)

        z_minus_w_precision.append(cv_z_minus_w_precision)
        z_minus_w_recall.append(cv_z_minus_w_recall)
        z_minus_w_fmeasure.append(cv_z_minus_w_fmeasure)

    print('---------------------------------------')

    print('accuracy(10 fold):', sum(accuracy) / n)

    print('x-y precision:', the_sum(x_minus_y_precision) / n)
    print('x-y recall', the_sum(x_minus_y_recall) / n)
    print('x-y f-measure', the_sum(x_minus_y_fmeasure) / n)

    print('x+y precision:', the_sum(x_plus_y_precision) / n)
    print('x+y recall', the_sum(x_plus_y_recall) / n)
    print('x+y f-measure', the_sum(x_plus_y_fmeasure) / n)

    print('y-x precision:', the_sum(y_minus_x_precision) / n)
    print('y-x recall', the_sum(y_minus_x_recall) / n)
    print('y-x f-measure', the_sum(y_minus_x_fmeasure) / n)

    print('y+z precision:', the_sum(y_plus_z_precision) / n)
    print('y+z recall', the_sum(y_plus_z_recall) / n)
    print('y+z f-measure', the_sum(y_plus_z_fmeasure) / n)

    print('y-z precision:', the_sum(y_minus_z_precision) / n)
    print('y-z recall', the_sum(y_minus_z_recall) / n)
    print('y-z f-measure', the_sum(y_minus_z_fmeasure) / n)

    print('z-y precision:', the_sum(z_minus_y_precision) / n)
    print('z-y recall', the_sum(z_minus_y_recall) / n)
    print('z-y f-measure', the_sum(z_minus_y_fmeasure) / n)

    print('z+w precision:', the_sum(z_plus_w_precision) / n)
    print('z+w recall', the_sum(z_plus_w_recall) / n)
    print('z+w f-measure', the_sum(z_plus_w_fmeasure) / n)

    print('z-w precision:', the_sum(z_minus_w_precision) / n)
    print('z-w recall', the_sum(z_minus_w_recall) / n)
    print('z-w f-measure', the_sum(z_minus_w_fmeasure) / n)


    print('---------------------------------------')


def the_sum(aList):
    s = 0
    for x in aList:
        if x:
            s = s + x
    return s
