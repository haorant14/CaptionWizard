import os
import shutil
from PIL import Image
from transformers import AutoTokenizer, ViTImageProcessor, VisionEncoderDecoderModel


def text_smoother(input_text, input_number):
    temp_list = input_text.split("  ")
    while True:
        exist = False
        i = 0
        while i < len(temp_list):
            if temp_list[i] == "":
                exist = True
                break
            i += 1
        if not exist:
            break
        else:
            temp_list.remove("")
    output = ""
    if len(temp_list) < input_number:
        i = 0
        while i < len(temp_list):
            output = output + " " + temp_list[i]
            i += 1
        return output.lstrip(" ")
    else:
        i = 0
        while i < input_number:
            output = output + " " + temp_list[i]
            i += 1
        return output.lstrip(" ")


def base_model_caption_generator(input_image_address):
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    image_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    img = Image.open(input_image_address).convert("RGB")
    pixel_values = image_processor(img, return_tensors="pt").pixel_values.to("cpu")
    generated_ids = model.generate(pixel_values)
    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text


def finetuned_model_caption_generator(input_image_address):
    model = VisionEncoderDecoderModel.from_pretrained("Image_Caption_Generator")
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    image_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    img = Image.open(input_image_address).convert("RGB")
    pixel_values = image_processor(img, return_tensors="pt").pixel_values.to("cpu")
    generated_ids = model.generate(pixel_values)
    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    generated_text = text_smoother(generated_text, 3)
    return generated_text


test_data_file = open("test_data.txt", "rt", encoding="utf-8")
lines = []
while True:
    line = test_data_file.readline()
    if line == "":
        break
    lines.append(line)
test_data_file.close()

test_data = []
i = 0
while i < len(lines):
    temp_list = lines[i].split(": ")
    test_data.append(temp_list)
    i += 1

os.mkdir("Test_Data")
i = 1
while i <= len(test_data):
    os.mkdir("Test_Data//" + str(i))
    i += 1
i = 1
while i <= len(test_data):
    current_element = test_data[i - 1]
    shutil.copyfile("All//" + current_element[0], "Test_Data//" + str(i) + "//" + current_element[0])
    base_model_caption_file = open("Test_Data//" + str(i) + "//base_model_caption.txt", "wt", encoding="utf-8")
    image_address = "Test_Data//" + str(i) + "//" + current_element[0]
    base_model_caption_file.write(base_model_caption_generator(image_address) + "\n")
    base_model_caption_file.close()
    finetuned_model_caption_file = open("Test_Data//" + str(i) + "//finetuned_model_caption.txt", "wt", encoding="utf-8")
    finetuned_model_caption_file.write(finetuned_model_caption_generator(image_address) + "\n")
    finetuned_model_caption_file.close()
    groundtruth_caption_file = open("Test_Data//" + str(i) + "//groundtruth_caption.txt", "wt", encoding="utf-8")
    groundtruth_caption_file.write(current_element[1] + "\n")
    groundtruth_caption_file.close()
    print(i)
    i += 1










