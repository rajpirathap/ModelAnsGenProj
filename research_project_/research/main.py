from research.corpus_reader import get_feature_set_for_a_data_item
from research.helper import get_numeric_values, load_classifier, any_numeric_is_float, calculate


classifier = load_classifier('my_classifier.pickle')

test_data = "Joe and Tom have 8 marbles when they put all their marbles together. Joe has 3 marbles. How many marbles does Tom have?"
#test_data = "In March it rained 0.81 inches. It rained 0.35 inches greater in April than in March. How much did it rain in April?"
test_feature_vector = get_feature_set_for_a_data_item(test_data)
#print(test_feature_vector)
numeric_values = get_numeric_values(test_data)

observed = classifier.classify(test_feature_vector)
print("Predicted formula is : {}".format(observed))

is_float_calculation = any_numeric_is_float(numeric_values[0], numeric_values[1])
value = calculate(is_float_calculation, observed, numeric_values)
print("Final Value is : {}".format(value))



# print(get_result(observed, test_data))