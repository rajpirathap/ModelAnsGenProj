import pickle
from nltk.tag import pos_tag
from research.corpus_reader import read_corpus, get_feature_set_for_a_data_item
import random

__author__ = 'RAJ-PC'


def get_numeric_values(sentence):
    tagged_sent = pos_tag(sentence.split())
    numbers = [word for word, pos in tagged_sent if pos == 'CD']
    return numbers


def save_classifier(classifier, name):
    f = open(name, 'wb')
    pickle.dump(classifier, f, -1)
    f.close()


def load_classifier(name):
    f = open(name, 'rb')
    classifier = pickle.load(f)
    f.close()
    return classifier


# def isfloat(x):
#     try:
#         a = float(x)
#     except ValueError:
#         return False
#     else:
#         return True

def is_float(string):
    try:
        return float(string) and '.' in string  # True if string is a number with a dot
    except ValueError:  # if string is not a number
        return False


# def isint(x):
#     try:
#         a = float(x)
#         b = int(a)
#     except ValueError:
#         return False
#     else:
#         return a == b


def any_numeric_is_float(num_1, num_2):
    if is_float(num_1) or is_float(num_2):
        return True
    return False


def calculate(is_float_calculation, observed, numerics):
    if observed == 'x-y':
        if is_float_calculation:
            return float(numerics[0]) - float(numerics[1])
        else:
            return int(numerics[0]) - int(numerics[1])
    elif observed == 'x+y':
        if is_float_calculation:
            return float(numerics[0]) + float(numerics[1])
        else:
            return int(numerics[0]) + int(numerics[1])
    elif observed == 'y-x':
        if is_float_calculation:
            return float(numerics[1]) - float(numerics[0])
        else:
            return int(numerics[1]) - int(numerics[0])
    elif observed == 'y+z':
        if is_float_calculation:
            return float(numerics[1]) + float(numerics[0])
        else:
            return int(numerics[1]) + int(numerics[0])
    elif observed == 'y-z':
        if is_float_calculation:
            return float(numerics[0]) - float(numerics[1])
        else:
            return int(numerics[0]) - int(numerics[1])
    elif observed == 'z-y':
        if is_float_calculation:
            return float(numerics[1]) - float(numerics[0])
        else:
            return int(numerics[1]) - int(numerics[0])
    elif observed == 'z+w':
        if is_float_calculation:
            return float(numerics[0]) + float(numerics[1])
        else:
            return int(numerics[0]) + int(numerics[1])

    elif observed == 'z-w':
        if is_float_calculation:
            return float(numerics[0]) - float(numerics[1])
        else:
            return int(numerics[0]) - int(numerics[1])

#
# def get_result(observed, test_data):
#     numeric = get_numeric(test_data)
#     if observed == 'equ1':
#         return int(numeric[0]) - int(numeric[1])
#     elif observed == 'equ2':
#         return int(numeric[0]) - int(numeric[1])
#
#


def get_train_features():
    corpus = read_corpus()
    train_features = ([(get_feature_set_for_a_data_item(question.split(',')[0]), question.split(',')[1].rstrip()) for question in corpus])
    return train_features


def get_percentage_train_and_test_data_set(featuresets):
    random.shuffle(featuresets)
    random.shuffle(featuresets)

    training_set = featuresets[:900]
    testing_set = featuresets[100:]
    return training_set, testing_set

