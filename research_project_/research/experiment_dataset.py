# from research.corpus_reader import get_feature_set_for_a_data_item
# from research.helper import get_numeric_values, load_classifier, any_numeric_is_float, calculate
#
#
# classifier = load_classifier('voted_classifier.pickle')
# #classifier = load_classifier('voted_classifier.pickle')
#
# with open("F:\\RAJPIRATHAP\\Mora-Msc\\semi4\\research_project\\research\\dataset\\Roys_dataset", "r") as ins:
#     for index, line in enumerate(ins):
#         test_data = question = line.rstrip()
#         test_feature_vector = get_feature_set_for_a_data_item(test_data)
#         #print(test_feature_vector)
#         # numeric_values = get_numeric_values(test_data)
#         #
#         # observed = classifier.classify(test_feature_vector)
#         # #print("Predicted formula is : {}".format(observed))
#         #
#         # is_float_calculation = any_numeric_is_float(numeric_values[0], numeric_values[1])
#         # value = calculate(is_float_calculation, observed, numeric_values)
#         # print("{}".format(value))
