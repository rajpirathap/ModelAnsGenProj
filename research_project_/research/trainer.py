from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm.classes import SVC
from sklearn.tree.tree import DecisionTreeClassifier
from research.helper import save_classifier, get_train_features
from research.model_experiment import print_cross_validation_experiment_result
from research.voted_classifier import VoteClassifier
from research.voted_classifier_accuracy import print_voted_classifier_cross_validation_experiment_result, \
    print_and_get_split_dataset_accuracy

train_features = get_train_features()


def train_individual_classifier():
    #classifier = SklearnClassifier(SVC(), sparse=False)
    classifier = SklearnClassifier(DecisionTreeClassifier(random_state=0), sparse=False)
    # classifier = SklearnClassifier(GaussianNB(), sparse=False)
    # classifier = SklearnClassifier(RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1), sparse=False)
    print_cross_validation_experiment_result(classifier, train_features)
    classifier.train(train_features)
    save_classifier(classifier, 'my_classifier.pickle')


def train_ensemble_classifier():
    # classifier2 = SklearnClassifier(GaussianNB(), sparse=False)
    # classifier1 = SklearnClassifier(SVC(), sparse=False)
    # classifier3 = SklearnClassifier(RandomForestClassifier(), sparse=False)
    # classifier4 = SklearnClassifier(DecisionTreeClassifier(), sparse=False)
    classifier2 = SklearnClassifier(GaussianNB(), sparse=False)
    classifier1 = SklearnClassifier(SVC(degree=18, C=12), sparse=False)
    classifier3 = SklearnClassifier(RandomForestClassifier(max_depth=100, n_estimators=10), sparse=False)
    classifier4 = SklearnClassifier(DecisionTreeClassifier(min_samples_split=2, min_samples_leaf=2, max_leaf_nodes=30, splitter='best', random_state=0), sparse=False)
    test_classifiers = []
    test_classifiers.append(classifier1)
    test_classifiers.append(classifier2)
    test_classifiers.append(classifier3)
    test_classifiers.append(classifier4)

    trained_classifiers = []

    for classifier in test_classifiers:
        classifier = classifier.train(train_features)
        trained_classifiers.append(classifier)

    voted_classifier = VoteClassifier(trained_classifiers)
    save_classifier(voted_classifier, 'voted_classifier.pickle')

    print_and_get_split_dataset_accuracy(test_classifiers, train_features)
    print_voted_classifier_cross_validation_experiment_result(test_classifiers, train_features)


def train_ensemble_decision_tree_classifier():
    #min_samples_split, min_samples_leaf, max_leaf_nodes, splitter
    classifier1 = SklearnClassifier(DecisionTreeClassifier(random_state=0), sparse=False)
    classifier2 = SklearnClassifier(DecisionTreeClassifier(max_depth=20, min_samples_split=3, min_samples_leaf=4, max_leaf_nodes=35, splitter='best', random_state=0), sparse=False)
    classifier3 = SklearnClassifier(DecisionTreeClassifier(max_depth=30, min_samples_split=2, min_samples_leaf=2, max_leaf_nodes=40, splitter='best', random_state=0), sparse=False)
    test_classifiers = []
    test_classifiers.append(classifier1)
    test_classifiers.append(classifier2)
    test_classifiers.append(classifier3)

    trained_classifiers = []

    for classifier in test_classifiers:
        classifier = classifier.train(train_features)
        trained_classifiers.append(classifier)

    voted_classifier = VoteClassifier(trained_classifiers)
    save_classifier(voted_classifier, 'voted_classifier_decision_tree.pickle')

    print_and_get_split_dataset_accuracy(test_classifiers, train_features)
    print_voted_classifier_cross_validation_experiment_result(test_classifiers,
                                                              train_features)



train_individual_classifier()
#train_ensemble_classifier()
#train_ensemble_decision_tree_classifier()
