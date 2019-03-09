__author__ = 'RAJ-PC'

from nltk.tokenize import sent_tokenize
from nltk.tag import pos_tag


def extract_sentence_dict(sentence, index, length):
    sent_dict = {}

    sent_dict['index'] = index
    sent_dict['sentence'] = sentence
    tagged_sent = pos_tag(sentence.split())
    # print(tagged_sent)
    proper_nouns = [word for word, pos in tagged_sent if pos == 'NNP']
    nouns = [word for word, pos in tagged_sent if pos == 'NNS']
    numbers = [word for word, pos in tagged_sent if pos == 'CD']
    adjectives = [word for word, pos in tagged_sent if pos == 'JJR']
    verbs = [word for word, pos in tagged_sent if pos == 'VBD']
    verb_base = [word for word, pos in tagged_sent if pos == 'VB']
    singular_noun = [word for word, pos in tagged_sent if pos == 'NN']
    prepositions = [word for word, pos in tagged_sent if pos == 'IN']
    ques_forms = [word for word, pos in tagged_sent if pos == 'WRB']

    sent_dict['ques_length'] = length

    if len(numbers) > 0:
        sent_dict['val'] = numbers[0]
    else:
        sent_dict['val'] = None

    if len(numbers) > 1:
        sent_dict['val2'] = numbers[1]
    else:
        sent_dict['val2'] = None

    if len(proper_nouns) > 0:
        sent_dict['prop_noun'] = proper_nouns[0]
    else:
        sent_dict['prop_noun'] = None

    if len(proper_nouns) > 1:
        sent_dict['prop_noun2'] = proper_nouns[1]
    else:
        sent_dict['prop_noun2'] = None

    if len(nouns) > 0:
        sent_dict['noun1'] = nouns[0]
    else:
        sent_dict['noun1'] = None

    if len(nouns) > 1:
        sent_dict['noun2'] = nouns[1]
    else:
        sent_dict['noun2'] = None

    if len(adjectives) > 0:
        sent_dict['adjective'] = adjectives[0]
    else:
        sent_dict['adjective'] = None

    if len(verbs) > 0:
        sent_dict['verb1'] = verbs[0]
    else:
        sent_dict['verb1'] = None

    if len(verbs) > 1:
        sent_dict['verb2'] = verbs[1]
    else:
        sent_dict['verb2'] = None

    if len(singular_noun) > 0:
        sent_dict['singular_noun1'] = singular_noun[0]
    else:
        sent_dict['singular_noun1'] = None

    if len(singular_noun) > 1:
        sent_dict['singular_noun2'] = singular_noun[1]
    else:
        sent_dict['singular_noun2'] = None

    if len(prepositions) > 0:
        sent_dict['preposition'] = prepositions[0]
    else:
        sent_dict['preposition'] = None

    if len(verb_base) > 0:
        sent_dict['verb_base'] = verb_base[0]
    else:
        sent_dict['verb_base'] = None

    if len(ques_forms) > 0:
        sent_dict['ques_form'] = ques_forms[0]
    else:
        sent_dict['ques_form'] = None
    return sent_dict


def build_sentence_dictionary_for_a_question(question_text):
    sent_arr = []

    sentences = sent_tokenize(question_text)
    length = len(sentences)
    for idx, sentence in enumerate(sentences):
        dict = extract_sentence_dict(sentence, idx, length)
        sent_arr.append(dict)

    return sent_arr


def get_position_dict_of_words_in_a_sentence(sentence):
    position = {}
    words = sentence.split(" ")
    for idx, word in enumerate(words):
        position[word] = idx
    return position


def find_distance_between_words(first_word, second_word, position_dict):
    first = None
    second = None
    for key, value in position_dict.items():
        if key == first_word:
            first = value
        if key == second_word:
            second = value

    if first == None or second == None:
        return 100

    return abs(first - second) - 1


def is_positive_comparative():
    pass
