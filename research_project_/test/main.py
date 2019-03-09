from nltk.tokenize import sent_tokenize
from research.corpus_reader import read_corpus, get_feature_set_for_a_data_item
from research.feature_extractor import get_bigrams, check_bigram_exist
from research.sent_dict_builder import find_distance_between_words, get_position_dict_of_words_in_a_sentence, \
    build_sentence_dictionary_for_a_question
from research.tokenizer import tokenize

__author__ = 'RAJ-PC'

print('test')

# import nltk.data
#
# nltk.download()


# from nltk.corpus import brown
#nltk.data.path.append("F:\Program Files\JetBrains\nltkdata");
#
# text=nltk.word_tokenize("mary has 20 apples")
# print(nltk.pos_tag(text))

# brown.words()

#nltk.data.path


# from nltk.tag import pos_tag
#
# sentence = "How many seashells does Mike now have ?"
# # sentence = "she gave Mike 63 of the seashells"
# # sentence = "Joan found 79 seashells on the beach"
# tagged_sent = pos_tag(sentence.split())
# print(tagged_sent)
# # [('Michael', 'NNP'), ('Jackson', 'NNP'), ('likes', 'VBZ'), ('to', 'TO'), ('eat', 'VB'), ('at', 'IN'), ('McDonalds', 'NNP')]
#
# propernouns = [word for word,pos in tagged_sent if pos == 'NNP']
# nouns = [word for word,pos in tagged_sent if pos == 'NNS']
# number = [word for word,pos in tagged_sent if pos == 'CD']
# print(nouns)
# print(propernouns)
# print(number)
# ['Michael','Jackson', 'McDonalds']
# sentences = sent_tokenize('Sam found 35 seashells on the beach . he gave Joan 18 of the seashells . How many seashells does he now have ? ')
#
# print(sentences[0])
# print(sentences[1])
# print(sentences[2])
from nltk.tag import pos_tag

# sentence = sent_tokenize('Bill  has 9 marbles. Jim has 7 more marbles than Bill. How many marbles does Jim have?')

# print(sentence[0])
# print(sentence[1])
# print(sentence[2])
# sentence = "Bill  has 9 marbles"
# sentence = "How many marbles does Jim have?"#"Jim has 7 more marbles than Bill"
# tagged_sent = pos_tag(sentence.split())
# print(tagged_sent)
#
#
# propernouns = [word for word,pos in tagged_sent if pos == 'NNP']
# nouns = [word for word,pos in tagged_sent if pos == 'NNS']
# number = [word for word,pos in tagged_sent if pos == 'CD']
# print(nouns)
# print(propernouns)
# print(number)


# sentence = sent_tokenize('Bill  has 9 marbles. He has 7 more marbles than Jim. How many marbles does Jim have?')
#
# print(sentence[0])
# print(sentence[1])
# print(sentence[2])
# sentence = "Bill  has 9 marbles"
# sentence = "Bill has 9 marbles . Jim has 7 less marbles than Bill . How many marbles does Jim have ?"#How many marbles does Jim have?"#"Jim has 7 more marbles than Bill"
# tagged_sent = pos_tag(sentence.split())
# print(tagged_sent)


# propernouns = [word for word,pos in tagged_sent if pos == 'NNP']
# nouns = [word for word,pos in tagged_sent if pos == 'NNS']
# number = [word for word,pos in tagged_sent if pos == 'CD']
# print(nouns)
# print(propernouns)
# print(number)

#sentence = 'Dina made cookies . She used 0.625 kg flour and 0.25 kg sugar . How much more flour than sugar did Dina use '
#sentence = 'Greg and Sharon own neighboring cornfields . Greg harvested 0.4 acre of corn on Monday and Sharon harvested 0.1 acre . How many more acres did Greg harvest than Sharon ?'

#dic = build_sentence_dictionary_for_a_question(sentence)
# result = get_position_dict_of_words_in_a_sentence(sentence)
# for key, value in result.items():
#     if key == 'Mary':
#         print('hehe')

# ans = find_distance_between_words('flowere', 'from', result)
# print(ans)

#abcd
#
# from nltk.stem import WordNetLemmatizer
#
# lemmatizer = WordNetLemmatizer()
#
# print(lemmatizer.lemmatize("" , 'v'))
# sentence = 'Sam had 9 dollars in his bank. His dad gave him 7 dollars. How many dollars does Sam have now?'
# sent = tokenize(sentence)
# print(sent)

#read_corpus()


# Joan found 70 seashells on the beach . she gave Sam some of her seashells . She has 27 seashell . How many seashells did she give to Sam ?
#
# Sam had 9 dimes in his bank . His dad gave him 7 dimes . How many dimes does Sam have now ?
# Tom found 7 seashells but 4 were broken . How many unbroken seashells did Tom find ?
# There are 7 crayons in the drawer . Mary took 3 crayons out of the drawer . How many crayons are there now ?
# Dan picked 9 limes and gave Sara 4 of the limes . How many limes does Dan have now ?
# Tom has 30 violet balloons , he gave Fred 16 of the balloons . How many violet balloons does he now have ?
# Mike had 33 quarters and 87 nickels in his bank . His dad borrowed 75 nickels from Mike . How many nickels does he have now ?
#sentence = "Sam had 9 dimes in his bank. His dad gave him 7 dimes. How many dimes does Sam have now?"
#sentence = "Tom found 7 seashells but 4 were broken. How many unbroken seashells did Tom find?"
#sentence = "There are 7 crayons in the drawer. Mary took 3 crayons out of the drawer. How many crayons are there now?"
#sentence = "Dan picked 9 limes and gave Sara 4 of the limes. How many limes does Dan have now?"
#sentence = "Tom has 30 green balloons and he gave Fred 16 of the balloons. How many green balloons does he now have?"
#sentence = "Reeta had 45 fresh apples in her bag. Her sister gave her 8 more apples. How many apples does Reeta have now?"
#sentence = "Bill has 9 marbles. Jim has 7 more marbles than Bill. How many marbles does Jim have?"
#pairs = get_bigrams("His dad gave him 7 dimes")
#check_bigram_exist(pairs)
sentence = "The Richmond Tigers sold a total of 9570 tickets last season. If they sold 3867 tickets in the first half of the season.  how many tickets did they sell in the second half?"
#sentence = "Dina made cookies. She used 0.625 kg flour and 0.25 kg sugar. How much more flour than sugar did she use?"

test_feature_vector = get_feature_set_for_a_data_item(sentence)
print(test_feature_vector)