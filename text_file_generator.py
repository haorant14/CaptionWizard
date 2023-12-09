
import pandas as pd
import os
from langdetect import detect





# This function removes @something and #something in the input string
def hashtag_and_at_remover(input_string):
    # removing @sth
    while True:
        if input_string.find("@") == -1:
            break
        temp_list = input_string.split(" ")
        redundant_words = []
        i = 0
        while i < len(temp_list):
            current_word = temp_list[i]
            if current_word.find("@") != -1:
                redundant_words.append(current_word)
            i += 1

        i = 0
        while i < len(redundant_words):
            redundant_word = redundant_words[i]
            temp_list.remove(redundant_word)
            i += 1

        temp_string = ""
        i = 0
        while i < len(temp_list):
            temp_string = temp_string + temp_list[i] + " "
            i += 1
        input_string = temp_string.rstrip(" ")

    # removing #sth
    while True:
        if input_string.find("#") == -1:
            break
        temp_list = input_string.split(" ")
        redundant_words = []
        i = 0
        while i < len(temp_list):
            current_word = temp_list[i]
            if current_word.find("#") != -1:
                redundant_words.append(current_word)
            i += 1

        i = 0
        while i < len(redundant_words):
            redundant_word = redundant_words[i]
            temp_list.remove(redundant_word)
            i += 1

        temp_string = ""
        i = 0
        while i < len(temp_list):
            temp_string = temp_string + temp_list[i] + " "
            i += 1
        input_string = temp_string.rstrip(" ")

    return input_string


# main part of the code starts here
df = pd.read_csv("clean_data.csv", lineterminator='\n')
df = df[["img_p", "preprocessed_captions"]]
img_caption_list = df.values.tolist()

# generating the dataset.txt file
dataset_file = open("new_dataset.txt", "wt", encoding="utf-8")
i = 0
while i < len(img_caption_list):
    if os.path.exists("All/" + img_caption_list[i][0]):
        caption = str(img_caption_list[i][1])
        caption = caption.replace("\n", " ")
        caption = hashtag_and_at_remover(caption)
        caption = caption.lower()
        dataset_file.write(str(img_caption_list[i][0]) + ": " + caption + "\n")

    print(i)
    i += 1
dataset_file.close()




















