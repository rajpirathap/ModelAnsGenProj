

with open("q.txt", "r") as ins:
    for index, line in enumerate(ins):
        question = line.rstrip()
        numbers = [int(s) for s in question.split() if s.isdigit()]
        if len(numbers) == 2:
            question_parts = question.split(".")
            if len(question_parts) <= 3:
                print("{}".format(index+1))
                # array.append(line)

