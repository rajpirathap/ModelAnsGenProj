from research.feature_extractor import return_feature_vector
from research.sent_dict_builder import build_sentence_dictionary_for_a_question
from research.tokenizer import tokenize

__author__ = 'RAJ-PC'


def get_feature_set_for_a_data_item(sentence):
    print(sentence)
    sent_dict_arr = build_sentence_dictionary_for_a_question(tokenize(sentence))
    feature_set_vector = return_feature_vector(sent_dict_arr)
    return feature_set_vector


def read_corpus():
    with open("/Users/rajpirathapsakthithasan/Files/RAJPIRATHAP/Mora-Msc/semi4/research_project/research/dataset/dataset.txt", 'r') as question:
        dataset = [question for question in question.readlines()]
        return dataset



def read_test_corpus(corpus):
    with open("F:\\RAJPIRATHAP\\Mora-Msc\\semi4\\research_project\\research\\dataset\\{}".format(corpus), 'r') as question:
        dataset = [question for question in question.readlines()]
        return dataset










# http://stackoverflow.com/questions/23329051/how-to-read-and-label-line-by-line-a-text-file-using-nltk-corpus-in-python


# def read_corpus_and_return_feature_sets():
#     reviews = []
#     with open('F:\RAJPIRATHAP\Mora-Msc\semi4\dataset\change_research\compare_type_1.txt', 'r') as question_1, open(
#             'F:\RAJPIRATHAP\Mora-Msc\semi4\dataset\change_research\compare_type_2.txt', 'r') as question_2:
#
#          reviews = ([(get_feature_set_for_a_data_item(question), 'equ1') for question in question_1.readlines()] + [(get_feature_set_for_a_data_item(question), 'equ2') for question in question_2.readlines()])
#     return reviews


# from nltk.corpus import PlaintextCorpusReader
#
# # RegEx or list of file names
# files = ".*\.txt"
#
# corpus0 = PlaintextCorpusReader("F:\RAJPIRATHAP\Mora-Msc\semi4\dataset\chane_research", files)
#
# print('read')

