from nltk.util import bigrams
from research.sent_dict_builder import get_position_dict_of_words_in_a_sentence, find_distance_between_words
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from research.word_collection import NEGATIVE_WORDS, WORD_ADDITION

__author__ = 'RAJ-PC'


def check_comparative_words_in_dict(dict_arr):
    if dict_arr[0]['ques_length'] == 3:
        if dict_arr[1]['adjective'] is not None:
            if dict_arr[1]['adjective'] in ['more', 'less']:
                return True

        elif dict_arr[2]['adjective'] is not None:
            if dict_arr[1]['adjective'] in ['more', 'less']:
                return True
        return False

    # if dict_arr[0]['ques_length'] == 2:

    else:
        return False


def having_conjunctive_preposition_as_word_than(dict_arr):
    if dict_arr[0]['ques_length'] == 2:
        if dict_arr[1]['preposition'] in ['than']:
            return True
        return False

    if dict_arr[0]['ques_length'] == 3:
        if dict_arr[1]['preposition'] in ['than']:
            return True

        elif dict_arr[2]['preposition'] in ['than']:  # for last question type of compare
            return True
        return False


def is_distance_to_comparative_word_less_than_threshold(dict_arr):
    if dict_arr[0]['ques_length'] == 3:
        if check_comparative_words_in_dict(dict_arr):
            position_dict = get_position_dict_of_words_in_a_sentence(dict_arr[1]['sentence'])
            distance = find_distance_between_words(dict_arr[1]['adjective'], dict_arr[1]['noun1'], position_dict)
            if distance < 3:
                return True
            else:
                return False
        else:
            return False
    else:
        return False


def is_comparative_positive(dict_arr):
    if dict_arr[0]['ques_length'] == 3:
        if 'more' in dict_arr[1]['sentence']:
            return True
        elif 'more' in dict_arr[2]['sentence']:
            return True

        return False
    return False


def is_comparative_negative(dict_arr):
    if dict_arr[0]['ques_length'] == 3:
        if 'less' in dict_arr[1]['sentence']:
            return True
        elif 'less' in dict_arr[2]['sentence']:
            return True

        return False
    return False


def pronoun_matching_with_first_sent(dict_arr):
    if dict_arr[0]['ques_length'] == 3:
        if dict_arr[0]['prop_noun'] == dict_arr[1]['prop_noun2']:
            return True
        else:
            return False
    return False


def pronoun_matching_with_third_sent(dict_arr):
    if dict_arr[0]['ques_length'] == 3:
        if dict_arr[0]['prop_noun'] == dict_arr[2]['prop_noun']:
            return True
        else:
            return False
    return False


def is_sent_having_two_matching_verbs_with_singular_nouns(dict_arr):  # TODO: need to consider the order of the words
    if dict_arr[0]['ques_length'] == 2:
        if dict_arr[0]['verb1'] and dict_arr[0]['verb2']:
            if lemmatize(dict_arr[0]['verb1']) in [dict_arr[1]['verb_base'],
                                                   dict_arr[1]['singular_noun1']] and lemmatize(
                dict_arr[0]['verb2']) in [dict_arr[1]['singular_noun1'], dict_arr[1]['singular_noun2']]:
                return True
            return False
        return False
    return False


def is_val1_greater_than_val2(dict_arr):
    from research.helper import any_numeric_is_float
    if dict_arr[0]['ques_length'] == 2:
        is_float_calculation = any_numeric_is_float(dict_arr[0]['val'], dict_arr[0]['val2'])
        if is_float_calculation:
            if dict_arr[0]['val'] and dict_arr[0]['val2']:
                if float(dict_arr[0]['val']) > float(dict_arr[0]['val2']):
                    return True
                return False
            return False
        else:
            if dict_arr[0]['val'] and dict_arr[0]['val2']:
                if int(dict_arr[0]['val']) > int(dict_arr[0]['val2']):
                    return True
                return False
            return False

    if dict_arr[0]['ques_length'] == 3:
        if dict_arr[1]['val'] and dict_arr[1]['val2']:
            is_float_calculation = any_numeric_is_float(dict_arr[1]['val'], dict_arr[1]['val2'])
            if is_float_calculation:
                if float(dict_arr[1]['val']) > float(dict_arr[1]['val2']):
                    return True
                return False
            else:
                if int(dict_arr[1]['val']) > int(dict_arr[1]['val2']):
                    return True
                return False

        elif dict_arr[0]['val'] and dict_arr[1]['val']:
            is_float_calculation = any_numeric_is_float(dict_arr[0]['val'], dict_arr[1]['val'])
            if is_float_calculation:
                if float(dict_arr[0]['val']) > float(dict_arr[1]['val']):
                    return True
                return False
            else:
                if int(dict_arr[0]['val']) > int(dict_arr[1]['val']):
                    return True
                return False
        return False
    return False


def is_last_sentence_is_question(dict_arr):
    if dict_arr[0]['ques_length'] == 2:
        if dict_arr[1]['ques_form']:
            return True
        return False
    if dict_arr[0]['ques_length'] == 3:
        if dict_arr[2]['ques_form']:
            return True
        return False


def is_sentence_has_two_matching_noun_parts(dict_arr):
    if dict_arr[0]['ques_length'] == 3:
        sentence_2 = dict_arr[1]['sentence']
        sentence_3 = dict_arr[2]['sentence']

        tagged_sent_2 = pos_tag(sentence_2.split())
        tagged_sent_3 = pos_tag(sentence_3.split())
        coordinating_conjunction = [word for word, pos in tagged_sent_2 if pos == 'CC']
        singular_nouns_2 = [word for word, pos in tagged_sent_2 if pos == 'NN']
        singular_nouns_3 = [word for word, pos in tagged_sent_3 if pos == 'NN']
        if set(singular_nouns_2).intersection(singular_nouns_3):
            if len(coordinating_conjunction) > 0 and coordinating_conjunction[0] == 'and':
                if len(singular_nouns_2) > 2:
                    position_dict = get_position_dict_of_words_in_a_sentence(sentence_2)
                    distance_1 = find_distance_between_words(coordinating_conjunction[0], singular_nouns_2[0],
                                                             position_dict)
                    distance_2 = find_distance_between_words(coordinating_conjunction[0], singular_nouns_2[1],
                                                             position_dict)
                    distance_3 = find_distance_between_words(coordinating_conjunction[0], singular_nouns_2[2],
                                                             position_dict)
                    if distance_1 < 2 and distance_2 < 2 and distance_3 < 3:
                        return True
                    return False
                return False
            return False
    return False


def is_question_having_two_nearest_numeric_values(dict_arr):
    if dict_arr[0]['ques_length'] == 2:
        if dict_arr[0]['val'] and dict_arr[0]['val2']:
            return True
        return False

    if dict_arr[0]['ques_length'] == 3:
        if dict_arr[1]['val'] and dict_arr[1]['val2']:
            return True
        return False


def is_matching_proper_nouns_existing_in_all_parts(dict_arr):
    if dict_arr[0]['ques_length'] == 3:
        sent_1 = pos_tag(dict_arr[0]['sentence'].split())
        sent_2 = pos_tag(dict_arr[1]['sentence'].split())
        sent_3 = pos_tag(dict_arr[2]['sentence'].split())
        prop_noun_1 = [word for word, pos in sent_1 if pos == 'NNP']
        prop_noun_2 = [word for word, pos in sent_2 if pos == 'NNP']
        prop_noun_3 = [word for word, pos in sent_3 if pos == 'NNP']
        if set(prop_noun_1).intersection(prop_noun_2) and set(prop_noun_1).intersection(prop_noun_3):
            return True
        else:
            return False

    if dict_arr[0]['ques_length'] == 2:
        if dict_arr[0]['prop_noun'] is not None:
            if dict_arr[0]['prop_noun'] == dict_arr[1]['prop_noun']:
                return True
        return False
    return False


def is_unordered_matching_proper_nouns_existing_in_all_parts(dict_arr):
    if dict_arr[0]['ques_length'] == 3:
        sent_1 = pos_tag(dict_arr[0]['sentence'].split())
        sent_2 = pos_tag(dict_arr[1]['sentence'].split())
        sent_3 = pos_tag(dict_arr[2]['sentence'].split())
        prop_noun_1 = [word for word, pos in sent_1 if pos == 'NNP']
        prop_noun_2 = [word for word, pos in sent_2 if pos == 'NNP']
        prop_noun_3 = [word for word, pos in sent_3 if pos == 'NNP']
        if set(prop_noun_1).intersection(prop_noun_2) and set(prop_noun_1).intersection(prop_noun_3):
            if prop_noun_2[0] != prop_noun_3[0]:
                return True
            else:
                return False
        else:
            return False
    return False


def is_having_same_verb_in_sentence(dict_arr):
    if dict_arr[0]['ques_length'] == 3:
        if dict_arr[1]['verb1'] == dict_arr[1]['verb2']:
            return True
        return False
    return False


def is_comparative_word_within_prepositions(dict_arr):
    if dict_arr[0]['ques_length'] == 3:
        position_dict = get_position_dict_of_words_in_a_sentence(dict_arr[2]['sentence'])
        distance_1 = find_distance_between_words(dict_arr[2]['prop_noun'], dict_arr[2]['preposition'], position_dict)
        distance_2 = find_distance_between_words(dict_arr[2]['preposition'], dict_arr[2]['prop_noun2'], position_dict)
        if distance_1 < 2 and distance_2 < 2:
            return True
        else:
            return False
    return False


def is_sent_two_having_two_proper_nouns(dict_arr):
    if dict_arr[0]['ques_length'] == 3:
        if dict_arr[1]['prop_noun'] and dict_arr[1]['prop_noun2']:
            return True
        return False
    return False


def is_possessive_pronoun_near_to_noun_in_second_sentence(dict_arr):
    if dict_arr[0]['ques_length'] == 3:
        sent = pos_tag(dict_arr[1]['sentence'].split())
        possessive_pronoun_arr = [word for word, pos in sent if pos == 'PRP$']
        if possessive_pronoun_arr:
            possessive_pronoun = possessive_pronoun_arr[0]

            nouns = [word for word, pos in sent if pos == 'NNS']
            if not len(nouns):
                nouns = [word for word, pos in sent if pos == 'NN']

            noun = nouns[0]
            if possessive_pronoun and noun:
                position_dict = get_position_dict_of_words_in_a_sentence(dict_arr[1]['sentence'])
                distance = find_distance_between_words(noun, possessive_pronoun, position_dict)
                if distance < 2:
                    return True
                return False
            return False
        return False
    return False


def is_having_positive_impact_on_last_sent(dict_arr):
    positive_words = ['gave', 'threw']
    if dict_arr[0]['ques_length'] == 3:
        sent_pairs = get_bigrams(dict_arr[1]['sentence'])
        index_of_pair = check_bigram_exist(sent_pairs)  # gave him
        if index_of_pair:
            pair = sent_pairs[index_of_pair]
            tokenized_pair = word_tokenize(pair)
            if set(tokenized_pair).intersection(positive_words):
                return True
            return False
        return False
    return False


def is_having_negative_impact_on_last_sentence(dict_arr):
    negative_words = NEGATIVE_WORDS
    if dict_arr[0]['ques_length'] == 2:
        sent = pos_tag(dict_arr[0]['sentence'].split())
        coordinating_conjunctions = [word for word, pos in sent if pos == 'CC']  # check 'but', 'and', 'or'  is exist
        if 'but' in coordinating_conjunctions:
            verbs = [word for word, pos in sent if pos == 'VBN']  # identify 'broken' as negative change
            if set(verbs).intersection(negative_words):
                return True
            return False
        elif 'and' in coordinating_conjunctions:
            verbs = [word for word, pos in sent if pos == 'VBD']  # identify 'broken' as negative change
            if set(verbs).intersection(negative_words):
                return True
            return False

        return False

    if dict_arr[0]['ques_length'] == 3:
        sent = pos_tag(dict_arr[1]['sentence'].split())
        position_dict = get_position_dict_of_words_in_a_sentence(dict_arr[1]['sentence'])
        nouns = [word for word, pos in sent if
                 pos == 'NNP']
        verbs = [word for word, pos in sent if
                 pos == 'VBD']  # identify verb or past tense exist in second sent # for change type
        if len(nouns) and len(verbs):
            noun = nouns[0]
            verb = verbs[0]
            if set(verbs).intersection(negative_words):
                distance = find_distance_between_words(noun, verb, position_dict)
                if 0 <= distance <= 2:
                    return True
                return False
            return False
        return False
    return False


def is_having_action_to_make_quantity_reduction(dict_arr):
    negative_words = NEGATIVE_WORDS
    if dict_arr[0]['ques_length'] == 3:
        position_dict = get_position_dict_of_words_in_a_sentence(dict_arr[1]['sentence'])
        prop_noun = dict_arr[1]['prop_noun']
        if prop_noun is None:
            prop_noun = dict_arr[1]['noun1']
        verb = dict_arr[1]['verb1']
        val2 = dict_arr[1]['val']
        distance_between_propnoun_and_verb = find_distance_between_words(prop_noun, verb, position_dict)
        distance_between_verb_and_val = find_distance_between_words(verb, val2, position_dict)
        if (0 <= distance_between_propnoun_and_verb < 2) and (0 <= distance_between_verb_and_val < 2) and set([verb]).intersection(negative_words):
            return True
        return False
    return False


def is_change_action_made_to_same_noun_entity(dict_arr):
    if dict_arr[0]['ques_length'] == 2:
        str_arr = dict_arr[0]['sentence']
        str_arr = str_arr.split()
        noun = dict_arr[0]['noun1']
        val1 = dict_arr[0]['val']
        val2 = dict_arr[0]['val2']
        if noun and val1 and val2:
            indices = [i for i, x in enumerate(str_arr) if x == noun]
            val1_index = str_arr.index(val1)
            val2_index = str_arr.index(val2)
            if len(indices) > 1 and val1_index and val2_index:
                diff_1 = (int(indices[0]) - int(val1_index))
                diff_2 = (int(indices[1]) - int(val2_index))
                if 0 < diff_1 <= 3 and 0 < diff_2 <= 3:
                    return True
                return False
            return False
        return False
    if dict_arr[0]['ques_length'] == 3:
        return False


def is_contains_co_reference_resolution(dict_arr):
    if dict_arr[0]['ques_length'] == 2:
        proper_noun = dict_arr[0]['prop_noun']
        last_sentence = pos_tag(dict_arr[1]['sentence'].split())
        personal_pronouns = [word for word, pos in last_sentence if pos == 'PRP']
        if len(personal_pronouns) > 0 and proper_noun is not None:
            return True
        return False
    if dict_arr[0]['ques_length'] == 3:
        last_sentence = pos_tag(dict_arr[2]['sentence'].split())
        proper_noun = dict_arr[0]['prop_noun']
        personal_pronouns = [word for word, pos in last_sentence if pos == 'PRP']
        if len(personal_pronouns) > 0 and proper_noun is not None:
            return True
        return False
    return False


def is_contain_two_verb_lemmas(dict_arr):
    if dict_arr[0]['ques_length'] == 3:
        first_sentence = pos_tag(dict_arr[0]['sentence'].split())
        second_sentence = pos_tag(dict_arr[1]['sentence'].split())
        verb_lemmas_1 = [word for word, pos in first_sentence if pos == 'VBZ']
        verb_lemmas_2 = [word for word, pos in second_sentence if pos == 'VBN']
        if len(verb_lemmas_1) > 0 and  len(verb_lemmas_2) > 0:
            return True
        return False
    return False


def is_sentence_contains_nouns_for_collective_noun(dict_arr):
    if dict_arr[0]['ques_length'] == 2:
        noun1 = dict_arr[0]['noun1']
        noun2 = dict_arr[0]['noun2']
        noun3 = dict_arr[1]['noun1']
        if noun1 and noun2:
            if noun3:
                return True
            return False
        return False
    if dict_arr[0]['ques_length'] == 3:
        noun1 = dict_arr[0]['noun1']
        noun2 = dict_arr[0]['noun2']
        singular_noun1 = dict_arr[0]['singular_noun1']
        singular_noun2 = dict_arr[0]['singular_noun1']
        if noun1 and noun2 and singular_noun1 and singular_noun2:
            last_sentence_noun = dict_arr[2]['noun1']
            if last_sentence_noun == noun1:
                return True
            return False
        return False
    return False


def is_contains_two_action_makers(dict_arr):
    if dict_arr[0]['ques_length'] == 2:
        prop_noun = dict_arr[0]['prop_noun']
        prop_noun2 = dict_arr[0]['prop_noun2']
        if prop_noun is not None and prop_noun2 is not None:
            return True
        return False
    if dict_arr[0]['ques_length'] == 3:
        prop_noun = dict_arr[0]['prop_noun']
        prop_noun2 = dict_arr[0]['prop_noun2']
        if prop_noun is not None and prop_noun2 is not None:
            return True
        return False
    return False


def is_last_sentence_contains_any_addition_from_action_makers(dict_arr):
    words_addition = WORD_ADDITION
    if dict_arr[0]['ques_length'] == 2:
        words = dict_arr[1]['sentence'].split()
        if set(words).intersection(words_addition):
            return True
        return False
    if dict_arr[0]['ques_length'] == 3:
        words = dict_arr[2]['sentence'].split()
        if set(words).intersection(words_addition):
            return True
        return False
    return False


def is_proper_nouns_are_splitted(dict_arr):
    if dict_arr[0]['ques_length'] == 3:
        sent_1 = pos_tag(dict_arr[0]['sentence'].split())
        sent_2 = pos_tag(dict_arr[1]['sentence'].split())
        sent_3 = pos_tag(dict_arr[2]['sentence'].split())
        prop_noun_1 = [word for word, pos in sent_1 if pos == 'NNP']
        prop_noun_2 = [word for word, pos in sent_2 if pos == 'NNP']
        prop_noun_3 = [word for word, pos in sent_3 if pos == 'NNP']
        if set(prop_noun_1).intersection(prop_noun_2) and set(prop_noun_1).intersection(prop_noun_3):
            return True
        else:
            return False

    if dict_arr[0]['ques_length'] == 2:
        if dict_arr[0]['prop_noun'] is not None:
            if dict_arr[0]['prop_noun'] == dict_arr[1]['prop_noun']:
                return True
        return False
    return False


def is_collective_noun_splitted(dict_arr):
    if dict_arr[0]['ques_length'] == 3:
        sent_1 = pos_tag(dict_arr[0]['sentence'].split())
        sent_2 = pos_tag(dict_arr[1]['sentence'].split())
        sent_3 = pos_tag(dict_arr[2]['sentence'].split())
        nouns1 = [word for word, pos in sent_1 if pos == 'NNS']
        nouns2 = [word for word, pos in sent_2 if pos == 'NNS']
        nouns3 = [word for word, pos in sent_3 if pos == 'NNS']
        filtered_nouns = set(nouns2).difference(nouns1)
        missed_noun = list(set(filtered_nouns).difference(nouns3))
        if nouns1 and nouns2 and nouns3 and len(missed_noun) > 0:
            if missed_noun[0] != nouns3[0]:
                return True
            return False

    return False


def is_question_length_is_two(question_text):
    sentences = sent_tokenize(question_text)
    if len(sentences) == 2:
        return True
    return False


def is_question_length_is_three(question_text):
    sentences = sent_tokenize(question_text)
    if len(sentences) == 3:
        return True
    return False


def lemmatize(word):
    lemmatizer = WordNetLemmatizer()

    return lemmatizer.lemmatize(word, 'v')


def get_bigrams(sentence):
    tokens = word_tokenize(sentence)
    pairs = [" ".join(pair) for pair in bigrams(tokens)]

    return pairs


def check_bigram_exist(sent_pairs):
    for idx, pair in enumerate(sent_pairs):
        tagged_sent = pos_tag(pair.split())
        if tagged_sent[0][1] == 'VBD' and tagged_sent[1][1] == 'PRP':
            return idx
    return False


def return_feature_vector(dict_arr):
    feature_vec = {}
    feature_vec['is_having_comparative'] = check_comparative_words_in_dict(dict_arr)
    feature_vec['is_comparative_in_threshold'] = is_distance_to_comparative_word_less_than_threshold(dict_arr)
    feature_vec['is_positive_comparative'] = is_comparative_positive(dict_arr)  # from research, program level, structural level
    feature_vec['is_comparative_negative'] = is_comparative_negative(dict_arr) # from research, program level, structural level
    feature_vec['is_first_last_sent_pronoun_matching'] = pronoun_matching_with_first_sent(dict_arr)
    feature_vec['is_verbs_and_singular_matching'] = is_sent_having_two_matching_verbs_with_singular_nouns(dict_arr)
    feature_vec['is_having_conjunctive_preposition'] = having_conjunctive_preposition_as_word_than(dict_arr) # from research, program level
    feature_vec['is_val1_greater_than_val2'] = is_val1_greater_than_val2(dict_arr)
    feature_vec['is_last_sentence_ques'] = is_last_sentence_is_question(dict_arr) #from research, sentence level
    feature_vec['is_sent_having_two_matching_noun'] = is_sentence_has_two_matching_noun_parts(dict_arr) # from reserach, entity related
    feature_vec['is_having_two_nearest_numeric_values'] = is_question_having_two_nearest_numeric_values(dict_arr)
    feature_vec['is_having_matching_proper_nouns'] = is_matching_proper_nouns_existing_in_all_parts(dict_arr)
    feature_vec['is_having_same_verb_in_sentence'] = is_having_same_verb_in_sentence(dict_arr)
    feature_vec['is_comparative_word_within_prepositions'] = is_comparative_word_within_prepositions(dict_arr) # from research, program level feature
    feature_vec['is_sent_two_having_two_proper_nouns'] = is_sent_two_having_two_proper_nouns(dict_arr)
    feature_vec[
        'is_unordered_matching_proper_nouns_existing_in_all_parts'] = is_unordered_matching_proper_nouns_existing_in_all_parts(
        dict_arr)



    # change type features
    feature_vec['is_pronoun_matching_with_third_sent'] = pronoun_matching_with_third_sent(dict_arr)
    feature_vec[
        'is_possessive_pronoun_near_to_noun_in_second_sentence'] = is_possessive_pronoun_near_to_noun_in_second_sentence(
        dict_arr)
    feature_vec['is_having_positive_impact_on_last_sent'] = is_having_positive_impact_on_last_sent(dict_arr)
    feature_vec['is_having_negative_impact_on_last_sent'] = is_having_negative_impact_on_last_sentence(dict_arr)
    feature_vec['is_having_action_to_make_quantity_reduction'] = is_having_action_to_make_quantity_reduction(dict_arr)
    feature_vec['is_change_action_made_to_same_noun_entity'] = is_change_action_made_to_same_noun_entity(dict_arr)
    feature_vec['is_contains_coreference_resolution'] = is_contains_co_reference_resolution(dict_arr)
    feature_vec['is_contain_two_verb_lemmas'] = is_contain_two_verb_lemmas(dict_arr) # from research, action related features

    #whole-part type features
    feature_vec['is_sentence_contains_nouns_for_collective_noun'] = is_sentence_contains_nouns_for_collective_noun(dict_arr)
    feature_vec['is_contains_two_action_makers'] = is_contains_two_action_makers(dict_arr)  # from research, action related features
    feature_vec['is_last_sentence_contains_any_addition_from_action_makers'] = is_last_sentence_contains_any_addition_from_action_makers(dict_arr)
    feature_vec['is_proper_nouns_are_splitted'] = is_proper_nouns_are_splitted(dict_arr)
    feature_vec['is_collective_noun_splitted'] = is_collective_noun_splitted(dict_arr)  # from research, action related features

    return feature_vec


    #roy's features
    # feature_vec['is_having_comparative'] = check_comparative_words_in_dict(dict_arr)
    # feature_vec['is_comparative_in_threshold'] = is_distance_to_comparative_word_less_than_threshold(dict_arr)
    # feature_vec['is_positive_comparative'] = is_comparative_positive(dict_arr)  # from research, program level, structural level
    # feature_vec['is_comparative_negative'] = is_comparative_negative(dict_arr)
    # feature_vec['is_first_last_sent_pronoun_matching'] = pronoun_matching_with_first_sent(dict_arr)
    # feature_vec[
    #     'is_unordered_matching_proper_nouns_existing_in_all_parts'] = is_unordered_matching_proper_nouns_existing_in_all_parts(
    #     dict_arr)
    # feature_vec['is_having_positive_impact_on_last_sent'] = is_having_positive_impact_on_last_sent(dict_arr)
    # feature_vec['is_having_negative_impact_on_last_sent'] = is_having_negative_impact_on_last_sentence(dict_arr)
    # feature_vec['is_having_action_to_make_quantity_reduction'] = is_having_action_to_make_quantity_reduction(dict_arr)
    # feature_vec['is_change_action_made_to_same_noun_entity'] = is_change_action_made_to_same_noun_entity(dict_arr)
    # feature_vec['is_comparative_word_within_prepositions'] = is_comparative_word_within_prepositions(dict_arr) # from research, program level feature

