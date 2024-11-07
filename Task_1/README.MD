![Alt text](./122.jpg)

# Mountain Named Entity Recognition (NER) using BERT
This project implements a Named Entity Recognition (NER) model using the BERT architecture for recognizing mountain names in text. It utilizes the Hugging Face transformers library and PyTorch.
<p align="left"> 
</a>   <a href="https://pandas.pydata.org/" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/2ae2a900d2f041da66e950e4d48052658d850630/icons/pandas/pandas-original.svg" alt="pandas" width="40" height="40"/> </a> 
</a> <a href="https://www.python.org" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" alt="python" width="40" height="40"/> </a> <a href="https://pytorch.org/" target="_blank" rel="noreferrer"> <img src="https://www.vectorlogo.zone/logos/pytorch/pytorch-icon.svg" alt="pytorch" width="40" height="40"/> </a> <a href="https://scikit-learn.org/" target="_blank" rel="noreferrer"> <img src="https://upload.wikimedia.org/wikipedia/commons/0/05/Scikit_learn_logo_small.svg" alt="scikit_learn" width="40" height="40"/> </a> <a href="https://seaborn.pydata.org/" target="_blank" rel="noreferrer"> <img src="https://seaborn.pydata.org/_images/logo-mark-lightbg.svg" alt="seaborn" width="40" height="40"/> </a> <a href="https://www.tensorflow.org" target="_blank" rel="noreferrer"> <img src="https://www.vectorlogo.zone/logos/tensorflow/tensorflow-icon.svg" alt="tensorflow" width="40" height="40"/> </a> </p>


## Project Overview
The goal of this project is to train a model that can identify and classify mountain names in text. The NER model is fine-tuned using a custom dataset containing sentences with mountain names annotated with the B-Mountain and I-Mountain labels. The model is based on BERT (Bidirectional Encoder Representations from Transformers), a transformer-based model that has achieved state-of-the-art results in various NLP tasks.

## Features
- Preprocesses and tokenizes data for token classification tasks.
- Fine-tunes a BERT model to recognize mountain names using a custom dataset.
- Saves the trained model and tokenizer for future inference.
- Evaluates the model during training on validation data to prevent overfitting.
## Requirements
To run this project, you need to have the following Python libraries installed:

- transformers (for Hugging Face models and tokenizers)
- torch (for deep learning and model training)
- sklearn (for evaluation metrics)
- pandas (for data manipulation)
- numpy (for numerical operations)
You can install them using pip:
```
pip install transformers torch sklearn pandas numpy
```
## Data creating
Download a mountain names
```python
data = pd.read_csv("allCountries_small.txt", sep="\t", header=None, low_memory=False)
...
print(mountain_names)
```
Output:
``` 
['Roc Meler', 'Pic de les Abelletes', 'Roc de Port Dret'...]
```
Create  a sentences:
```python
sentences = []
for name in mountain_names:  
    sentence = f"The mountain {name} is well-known."
    sentences.append((sentence, name))
    ...
    sentences.append((sentence, name))
```
Save to BIO-file
``` python
with open("mountains_ner_dataset.txt", "w",encoding="utf-8") as f:
    for sentence, mountain_name in sentences:
        bio_format = sentence_to_bio(sentence, mountain_name)
        for word, tag in bio_format:
            f.write(f"{word}\t{tag}\n")
        f.write("\n")  
```
## Dataset
The dataset consists of text sentences where mountain names are annotated with B-Mountain and I-Mountain tags. This is a custom dataset, and the training data is expected to be in a tab-separated format (e.g., mountains_ner_dataset.txt).

Example:
```

Himalayas B-Mountain
Mount I-Mountain
Khushkak I-Mountain
...
```
## Training the Model
- Clone or download this repository.
- Ensure that your dataset file is properly formatted (tab-separated, with mountain names labeled with B-Mountain and I-Mountain).
- Run the following script to start the training:

```
python train_model.py
```
This will train the model on your dataset and save the best model and tokenizer to the **./best_model and ./saved_mode2** directories.

## Model loool like :
```python
training_args = TrainingArguments(
    output_dir="./results",  # Directory for saving results
    evaluation_strategy="epoch",  # Evaluate model at each epoch
    learning_rate=2e-5,  # Initial learning rate
    per_device_train_batch_size=8,  # Batch size for training
    per_device_eval_batch_size=8,  # Batch size for evaluation
    num_train_epochs=3,  # Number of epochs
    weight_decay=0.01,  # Regularization coefficient
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
)
trainer.train()
```
## Here we have a link to model weights:
https://huggingface.co/VolodymyrKyba/NER_BERT_model
## Inference
Once the model is trained, you can use it to predict mountain names in new sentences:
```python
# Load the saved model and tokenizer
ner_pipeline = pipeline("ner", model="./best_model", tokenizer="./saved_mode2")
```
### Text we have:
"One of the most breathtaking places for travelers is the Himalayas, home to Mount Khushkak and Kanchenjunga, which attract climbers from all over the world. In Europe, Mont Blanc, the highest peak in the Alps, is also a popular destination for those seeking to conquer summits. In South America, the Andes mountain range stretches for thousands of kilometers, including famous peaks such as Aconcagua and Huascarán.In Africa, Mount Kilimanjaro stands as the highest peak on the continent, easily accessible without specialized climbing gear. Another impressive peak is Mount Meru, located in Tanzania, which is known for its rugged terrain. Moving to North America, the Rocky Mountains boast some of the most iconic mountain landscapes, with peaks like Mount Elbert and Mount Whitney standing tall.In the Pacific region, Mount Fuji in Japan is not only a cultural symbol but also a popular climbing destination. The mountains of New Zealand, like Mount Cook and Mount Taranaki, offer incredible hiking and climbing experiences. Finally, the Alps and the Carpathians continue to draw mountaineers and adventurers looking for challenging terrain and magnificent views."
### Output of searched mountain names:
```
['Himalayas', 'Khushkak', 'Kanchenjunga', 'Mont', 'Blanc', 'Alps', 'Aconcagua', 'Huascaran', 'Kilimanjaro', 'Meru', 'Tanzania', 'Elbert', 'Whitney', 'Fuji', 'Cook', 'Taranaki', 'Carpathians']
```
## File Structure

```
project/
│
├── allCoutries.txt           # Dataset of mountain name
├── mountains_ner_dataset.txt # Custom dataset for training the model
├── data_create.ipynb         # Creating BIO dataset
├── demo.ipynb                # Demo file
├── requirements.txt          # requirements
├── inference_model.py        # File of inference
├── model.py                  # Main script for training the NER 
├── best_model/               # Saved model after training
├── saved_mode2/              # Saved tokenizer after training
└── README.md                 # Project description and 
```

## License
This project is licensed under the MIT License.
