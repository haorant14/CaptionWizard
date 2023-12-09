import os
import datasets
import pandas as pd
from PIL import Image
import multiprocessing as mp
from sklearn.model_selection import train_test_split
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import VisionEncoderDecoderModel, ViTImageProcessor
from transformers import AutoTokenizer, default_data_collator
class ImageCaptionDataGenerator(Dataset):
    def __init__(self, df, root_dir, tokenizer, feature_extractor, transform=None):
        self.root_dir = root_dir
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.df = df
        self.transform = transform
        self.max_length = 150

    def __len__(self, ):
        return len(self.df)

    def __getitem__(self, idx):
        caption = self.df.caption.iloc[idx]
        image = self.df.image.iloc[idx]
        img_path = os.path.join(self.root_dir, image)
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        pixel_values = self.feature_extractor(img, return_tensors="pt").pixel_values
        captions = self.tokenizer(caption, padding='max_length', max_length=self.max_length).input_ids
        captions = [caption if caption != self.tokenizer.pad_token_id else -100 for caption in captions]
        encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(captions)}
        return encoding


class ModelConfigurations:
    epochs = 5
    image_size = (224, 224)
    train_batch_size = 8
    test_batch_size = 8
    test_epochs = 1
    LR = 5e-5
    seed = 42
    max_length = 128
    summary_length = 20
    weight_decay = 0.01
    mean = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)
    train_pct = 0.95
    num_workers = mp.cpu_count()
    label_mask = -100
    top_k = 1000
    top_p = 0.95


def train_and_test_data_generator(input_data_frame, output_file_name):
    images = input_data_frame["image"].tolist()
    captions = input_data_frame["caption"].tolist()
    output_file = open(output_file_name, "wt", encoding="utf-8")
    i = 0
    while i < len(images):
        output_file.write(images[i] + ": " + captions[i] + "\n")
        i += 1
    output_file.close()


def rouge_metric_calculator(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
    rouge_output = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])["rouge2"].mid
    return {
        "rouge2_precision": round(rouge_output.precision, 4),
        "rouge2_recall": round(rouge_output.recall, 4),
        "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
    }


def inputs_with_special_tokens_handler(self, token_ids_0, token_ids_1=None):
    outputs = [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
    return outputs


if torch.cuda.is_available():
    device = torch.device("cuda")
    print("There are %d GPU(s) available" % torch.cuda.device_count())
    print("Using the GPU: ", torch.cuda.get_device_name(0))
else:
    print("No GPU is detected, using the CPU instead")
    device = torch.device("cpu")

os.environ["WANDB_DISABLED"] = "true"
AutoTokenizer.build_inputs_with_special_tokens = inputs_with_special_tokens_handler
rouge = datasets.load_metric("rouge")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer.pad_token = tokenizer.unk_token
transforms = transforms.Compose(
    [transforms.Resize(ModelConfigurations.image_size), transforms.ToTensor()])
df = pd.read_csv("images_captions.txt", on_bad_lines="skip")
train_df, test_df = train_test_split(df, test_size=0.1)
train_and_test_data_generator(train_df, "train_data.txt")
train_and_test_data_generator(test_df, "test_data.txt")
print(len(train_df))
print(len(test_df))
df.head()
train_dataset = ImageCaptionDataGenerator(train_df, root_dir="Images/", tokenizer=tokenizer, feature_extractor=feature_extractor, transform=transforms)
val_dataset = ImageCaptionDataGenerator(test_df, root_dir="Images/", tokenizer=tokenizer, feature_extractor=feature_extractor, transform=transforms)
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
model.config.decoder_start_token_id = tokenizer.cls_token_id
model.config.pad_token_id = tokenizer.pad_token_id
model.config.vocab_size = model.config.decoder.vocab_size
model.config.eos_token_id = tokenizer.sep_token_id
model.config.decoder_start_token_id = tokenizer.bos_token_id
model.config.max_length = 128
model.config.early_stopping = True
model.config.no_repeat_ngram_size = 3
model.config.length_penalty = 2.0
model.config.num_beams = 4
training_args = Seq2SeqTrainingArguments(
    output_dir="Image_Caption_Generator",
    per_device_train_batch_size=ModelConfigurations.train_batch_size,
    per_device_eval_batch_size=ModelConfigurations.test_batch_size,
    predict_with_generate=True,
    evaluation_strategy="epoch",
    do_train=True,
    do_eval=True,
    logging_steps=1024,
    save_steps=2048,
    warmup_steps=1024,
    learning_rate=5e-5,
    num_train_epochs=ModelConfigurations.epochs,
    overwrite_output_dir=True,
    save_total_limit=1,
)


trainer = Seq2SeqTrainer(
    tokenizer=feature_extractor,
    model=model,
    args=training_args,
    compute_metrics=rouge_metric_calculator,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=default_data_collator,
)
trainer.train()
trainer.save_model("Image_Caption_Generator")





