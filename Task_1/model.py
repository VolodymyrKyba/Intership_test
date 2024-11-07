import random
from transformers import BertTokenizer
import torch
import tensorflow as tf, tf_keras
from transformers import BertForTokenClassification, Trainer, TrainingArguments
from sklearn.metrics import classification_report
from torch.utils.data import Dataset
from transformers import DataCollatorForTokenClassification


def load_sentences_and_labels(file_path):
    token_sentences = []
    token_labels = []
    sentence = []
    sentence_labels = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():  # If the line is not empty
                word, label = line.strip().split()
                sentence.append(word)
                sentence_labels.append(label)
            else:  # An empty line means the end of a sentence
                if sentence:
                    token_sentences.append(sentence)
                    token_labels.append(sentence_labels)
                    sentence = []
                    sentence_labels = []
        if sentence:  # Add the last sentence if there is one
            token_sentences.append(sentence)
            token_labels.append(sentence_labels)
    return token_sentences, token_labels


sentences, labels = load_sentences_and_labels("mountains_ner_dataset.txt")


def split_data(tag_sentences, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    train_size = int(
        train_ratio * len(tag_sentences)
    )  # Calculate the size of each set
    val_size = int(val_ratio * len(tag_sentences))
    train_set = tag_sentences[:train_size]  # Split into respective sets
    val_set = tag_sentences[train_size : train_size + val_size]
    test_set = tag_sentences[train_size + val_size :]
    return train_set, val_set, test_set


train_set, val_set, test_set = split_data(sentences)
train_label, val_label, test_label = split_data(labels)


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


def tokenize_and_preserve_labels(sentence, labels):
    tokenized_sentence = []
    label_ids = []
    for word, label in zip(sentence, labels):
        tokenized_word = tokenizer.tokenize(word)  # Tokenize the word
        tokenized_sentence.extend(tokenized_word)
        label_ids.extend(
            [label] + ["O"] * (len(tokenized_word) - 1)
        )  # Assign the original tag only to the first subtoken, others get "O"
    input_ids = tokenizer.convert_tokens_to_ids(
        tokenized_sentence
    )  # Convert tokens to IDs
    return input_ids, label_ids


tokenized_data = []
for sentence, label in zip(train_set, train_label):
    input_ids, label_ids = tokenize_and_preserve_labels(sentence, label)
    tokenized_data.append((input_ids, label_ids))


def add_special_tokens(input_ids, label_ids):
    # IDs for the [CLS] and [SEP] tokens
    cls_token_id = tokenizer.cls_token_id
    sep_token_id = tokenizer.sep_token_id
    input_ids = [cls_token_id] + input_ids + [sep_token_id]
    label_ids = ["O"] + label_ids + ["O"]  # Add "O" tag for special tokens
    return input_ids, label_ids


final_data = []
for input_ids, label_ids in tokenized_data:
    input_ids, label_ids = add_special_tokens(input_ids, label_ids)
    final_data.append((input_ids, label_ids))


def create_attention_mask(input_ids):
    return [1] * len(input_ids)


final_data_with_masks = []  # Add attention masks to the data
for input_ids, label_ids in final_data:
    attention_mask = create_attention_mask(input_ids)
    final_data_with_masks.append((input_ids, attention_mask, label_ids))


def prepare_data(sentences, labels):
    tokenized_data = []
    for sentence, label in zip(sentences, labels):
        input_ids, label_ids = tokenize_and_preserve_labels(sentence, label)
        input_ids, label_ids = add_special_tokens(input_ids, label_ids)
        attention_mask = create_attention_mask(input_ids)
        tokenized_data.append((input_ids, attention_mask, label_ids))
    return tokenized_data


train_data = prepare_data(train_set, train_label)
val_data = prepare_data(val_set, val_label)
test_data = prepare_data(test_set, test_label)

# Number of unique tags in the dataset, including "O" for the general background
label_list = list(
    set([label for sentence_labels in train_label for label in sentence_labels])
)
num_labels = len(label_list)
label_map = {label: i for i, label in enumerate(label_list)}  # Map labels to IDs

model = BertForTokenClassification.from_pretrained(
    "bert-base-uncased", num_labels=num_labels
)


class NERDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


# Convert the data to encoded objects for BERT
def encode_labels(labels, label_map):
    encoded_labels = []
    for sentence_labels in labels:
        encoded_labels.append([label_map[label] for label in sentence_labels])
    return encoded_labels


train_labels_encoded = encode_labels([x[2] for x in train_data], label_map)
val_labels_encoded = encode_labels([x[2] for x in val_data], label_map)

train_encodings = {
    "input_ids": [x[0] for x in train_data],
    "attention_mask": [x[1] for x in train_data],
}
val_encodings = {
    "input_ids": [x[0] for x in val_data],
    "attention_mask": [x[1] for x in val_data],
}

train_dataset = NERDataset(train_encodings, train_labels_encoded)
val_dataset = NERDataset(val_encodings, val_labels_encoded)

data_collator = DataCollatorForTokenClassification(tokenizer)
training_args = TrainingArguments(
    output_dir="./results",  # Directory for saving results
    evaluation_strategy="epoch",  # Evaluate model at each epoch
    learning_rate=2e-5,  # Initial learning rate
    per_device_train_batch_size=8,  # Batch size for training
    per_device_eval_batch_size=8,  # Batch size for evaluation
    num_train_epochs=3,  # Number of epochs
    weight_decay=0.01,  # Regularization coefficient
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
)
trainer.train()

trainer.save_model("./best_model")
tokenizer.save_pretrained("./saved_mode2")
