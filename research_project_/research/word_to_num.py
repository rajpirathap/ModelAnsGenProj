from word2number import w2n


sentence = "Nancy grew 6 potatoes. Sandy grew 9 potatoes. How many potatoes did they grow in total?"
sentence2 = "Sam found eighty thousand five hundred seashells and Mary found thirty thousand seashells on the beach. How many seashells did they find together?"
sentence3 = "Kamala and Vimala put fourteen pennies into a piggy bank. Vimala put in nine hundred fifty five thousand pennies. How many pennies did Kamala put in?"
sentence4 = "Gloria collects one point two one peanuts. Gloria's father gives Gloria eight hundred ninety-one more. How many peanuts does Gloria have?"
sentence5 = "Susan starts with seventy eight pencils. She gives thirty four to Patrick. How many pencils does Susan end with?"




num_start_pos = 0
num_end_pos = 0
formatted_string = ''
sentence = sentence.translate({ord(x): ' ' for x in [',', '-']})
words = sentence.split()
for idx, word in enumerate(words):
    try:
        current = w2n.word_to_num(word)
        if current:
            current_index = idx
            if num_start_pos == 0:
                num_start_pos = current_index
            next_word = words[current_index + 1]
            if next_word:
                # if next_word is 'and' or 'point':
                #     continue
                next_current = w2n.word_to_num(next_word)
                if next_current:
                    num_end_pos = current_index + 1
    except:
        is_num = False
        if num_start_pos > 0 and num_end_pos == 0:
            num_end_pos = num_start_pos

        if num_start_pos > 0 and num_end_pos > 0:
            num_string = ' '.join(words[num_start_pos:num_end_pos+1])
            next_current = w2n.word_to_num(num_string)
            if next_current:
                is_num = True
                formatted_string += ' ' + str(next_current)
                num_start_pos = 0
                num_end_pos = 0
        if not is_num:
            formatted_string += ' ' + word

print(formatted_string.lstrip())
