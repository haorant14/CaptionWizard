import os


dataset_file = open("new_dataset.txt", "rt", encoding="utf-8")
lines = []
while True:
    line = dataset_file.readline()
    if line == "":
        break
    lines.append(line)
dataset_file.close()

output_file = open("finetuning_captions.txt", "wt", encoding="utf-8")
output_file.write("image,caption\n")


i = 0
while i < len(lines):
    temp_list = lines[i].split(": ")
    if os.path.exists("Images/" + temp_list[0]):
        output_file.write(temp_list[0] + "," + temp_list[1])
    print(i)
    i += 1


output_file.close()








