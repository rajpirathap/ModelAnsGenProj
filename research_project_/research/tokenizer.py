# from nltk.tokenize import WordPunctTokenizer
#
# __author__ = 'RAJ-PC'
#
#
# def tokenize(sentence):
#     word_list = WordPunctTokenizer().tokenize(sentence)
#     tokenized_sent = " ".join(word_list)
#     return tokenized_sent


from nltk.tokenize import word_tokenize


def tokenize(sentence):
    word_list = word_tokenize(sentence)
    tokenized_sent = " ".join(word_list)
    return tokenized_sent