#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


get_ipython().system("unzip -qq '/content/drive/MyDrive/data.zip'")


# In[ ]:


get_ipython().system('pip install -q -U --no-cache-dir gdown --pre')
get_ipython().system('pip install -q transformers datasets evaluate accelerate')
get_ipython().system('pip install -q requests nlpaug sentencepiece')


# In[ ]:


get_ipython().system('pip install -q optuna')


# In[ ]:


#!nvidia-smi


# In[ ]:


# Standard modules

import os
import numpy as np
import pandas as pd

from tqdm import tqdm

# Pytorch

from torch import nn, tensor

# Hugging face

from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset, ClassLabel, load_metric
import evaluate


# ## Import data
# 

# In[ ]:


get_ipython().system('pip install -qq adapters')


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np


# In[ ]:





# In[ ]:



from sklearn.utils.class_weight import compute_class_weight
import torch

from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score, precision_score, recall_score
from datasets import DatasetDict
from tqdm import tqdm
import time
from transformers import RobertaConfig
from adapters import AutoAdapterModel
from adapters import AdapterTrainer
from transformers import RobertaConfig
from transformers import RobertaTokenizer




def make_dataframe(input_folder, labels_folder=None):
    #MAKE TXT DATAFRAME
    text = []

    for fil in tqdm(filter(lambda x: x.endswith('.txt'), os.listdir(input_folder))):

        iD, txt = fil[7:].split('.')[0], open(input_folder +fil, 'r', encoding='utf-8').read()
        text.append((iD, txt))

    df_text = pd.DataFrame(text, columns=['id','text']).set_index('id')

    df = df_text

    #MAKE LABEL DATAFRAME
    if labels_folder:
        labels = pd.read_csv(labels_folder, sep='\t', header=None)
        labels = labels.rename(columns={0:'id',1:'type'})
        labels.id = labels.id.apply(str)
        labels = labels.set_index('id')

        #JOIN
        df = labels.join(df_text)[['text','type']]

    return df
def read_lang_data(train_folder, train_labels, val_folder, val_labels):
  # read train data
  df_train_lang = make_dataframe(train_folder, train_labels)
  df_train_lang = df_train_lang.rename(columns={"type" : "label"})

  # read test data
  df_val_lang = make_dataframe(val_folder, val_labels)
  df_val_lang = df_val_lang.rename(columns={"type" : "label"})

  return df_train_lang, df_val_lang


recall_metric = evaluate.load("recall")
precision_metric = evaluate.load("precision")
accuracy_metric = evaluate.load("accuracy")

def eval_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    results = {}
    results.update(accuracy_metric.compute(predictions=preds, references = labels))
    results.update(recall_metric.compute(predictions=preds, references=labels, average="macro"))
    results.update(precision_metric.compute(predictions=preds, references=labels, average="macro"))

    # Calculate macro F1 score
    f1_macro = f1_score(labels, preds, average="macro")
    results.update({"eval_f1_macro": f1_macro})

    # Calculate micro F1 score
    f1_micro = f1_score(labels, preds, average="micro")
    results.update({"eval_f1_micro": f1_micro})

    return results



data_dict = {}
root_folder = '/home/ubunti/datastore/riepin/Test_Rie/New_Project/0_data/raw/'
languages = ['en', 'fr', 'ge', 'it', 'po', 'ru']

for lang in languages:

  train_folder = f"data/{lang}/train-articles-subtask-1/"
  train_labels = f"data/{lang}/train-labels-subtask-1.txt"
  dev_folder =  f"data/{lang}/dev-articles-subtask-1/"
  dev_labels =  f"data/{lang}/dev-labels-subtask-1.txt"

  df_train, df_dev = read_lang_data(train_folder, train_labels, dev_folder, dev_labels)

  data_dict[lang] = {'train': df_train, 'dev': df_dev, 'combined': pd.concat([df_train, df_dev])}


# In[ ]:


datasets = {}

for key,el in data_dict.items():
  labels = ['opinion', 'satire', 'reporting']
  ClassLabels = ClassLabel(num_classes=len(labels), names=labels)

  # Create hugging face dataset, adjusted to torch format and splitted for train/val in 80/20 ratio
  dataset = Dataset.from_pandas(el['combined'], preserve_index=True).cast_column("label", ClassLabels).train_test_split(test_size=0.2)
  val_dataset = Dataset.from_pandas(el['dev'], preserve_index=True).cast_column("label", ClassLabels)
  # combined_dataset = Dataset.from_pandas(el['combined'], preserve_index=True).cast_column("label", ClassLabels)

  datasets[key] = DatasetDict({'train': dataset['train'], 'test': dataset['test'], 'val': val_dataset})


# In[ ]:



# tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-uncased")
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

def encode_batch(batch):
  """Encodes a batch of input data using the model tokenizer."""
  return tokenizer(batch["text"], max_length=80, truncation=True, padding="max_length")


# datasets
tokenized_datasets = {}

for key,el in datasets.items():

    dataset = el.map(encode_batch, batched=True)
    # The transformers model expects the target class column to be named "labels"
    dataset = dataset.rename_column(original_column_name="label", new_column_name="labels")
    # Transform to pytorch tensors and only output the required columns
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    tokenized_datasets[key] = dataset

    # return AutoModelForSequenceClassification.from_pretrained("bert-base-multilingual-uncased", num_labels=len(labels))


# In[ ]:



def model_init_bert():

    config = AutoConfig.from_pretrained(
        "xlm-roberta-large",
        num_labels=3,
    )
    model = AutoAdapterModel.from_pretrained(
        "xlm-roberta-large",
        config=config,
    )

    model.add_adapter("semeval_3_3", config="seq_bn")
    model.add_classification_head(
        "semeval_3_3",
        num_labels=3,
        id2label={0: 'opinion', 1: 'satire', 2: 'reporting'}
        )
    # Activate the adapter
    model.train_adapter("semeval_3_3")

    return model


# In[ ]:


merged_train_set = pd.DataFrame()
merged_test_set = pd.DataFrame()
merged_val_set = pd.DataFrame()

df_train_list = []
df_test_list = []
df_val_list = []

for key,el in tokenized_datasets.items():

    df_train_list.append(el['train'].to_pandas())
    df_test_list.append(el['test'].to_pandas())
    df_val_list.append(el['val'].to_pandas())

merged_train_set = pd.concat(df_train_list)
merged_test_set = pd.concat(df_test_list)
merged_val_set = pd.concat(df_val_list)

train_dataset = Dataset.from_pandas(merged_train_set)
test_dataset = Dataset.from_pandas(merged_test_set)
val_dataset = Dataset.from_pandas(merged_val_set)

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])


# In[ ]:


from sklearn.model_selection import ParameterGrid
import matplotlib.pyplot as plt


adapter_bert = model_init_bert()

# Define the hyperparameter grid for tuning
hyperparameter_grid = {
    "num_train_epochs": [10, 15],
    "learning_rate": [3e-5, 4e-5, 5e-5],
    "per_device_batch_size": [8, 16],
}
best_params_global = None
best_f1_macro_global = 0.0
all_results = []

best_evaluate_results = {}

# Lists to store results for plotting
iterations = []
avg_f1_macros = []

# Perform grid search
for i, params in enumerate(ParameterGrid(hyperparameter_grid)):
    print('-----'*4)
    print(f"Iteration {i + 1}: Training with hyperparameters {params}")

    #### Training and evaluating bert
    training_args = TrainingArguments(
        output_dir="output_trainer",
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        skip_memory_metrics=True,
        num_train_epochs=params["num_train_epochs"],
        per_device_train_batch_size=params["per_device_batch_size"],
        per_device_eval_batch_size=params["per_device_batch_size"],
        learning_rate=params["learning_rate"],
        report_to="all"
    )

    trainer = AdapterTrainer(
        model=adapter_bert,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=eval_metrics,
    )


    trainer.train()
    # Evaluate on the combined validation set for parameter tuning
    evaluate_results = {}
    for lang in languages:
        ans_combined = trainer.evaluate(tokenized_datasets[lang]['val'])
        evaluate_results[lang] = ans_combined

    # Calculate average F1 macro across all languages
    avg_f1_macro = sum(result['eval_f1_macro'] for result in evaluate_results.values()) / len(languages)

    all_results.append({
        'hyperparameters': params,
        'avg_f1_macro': avg_f1_macro,
        'individual_results': evaluate_results,
        'trainers': trainer
    })

    # Update lists for plotting
    iterations.append(str(params))
    avg_f1_macros.append(avg_f1_macro)

    # Update best parameters if the current set performs better globally
    if avg_f1_macro > best_f1_macro_global:
        best_f1_macro_global = avg_f1_macro
        best_params_global = params
        best_evaluate_results = evaluate_results

print('\n'*3)
print("Best global hyperparameters:", best_params_global)
print("Best global F1 macro:", best_f1_macro_global)


# Visualize results
plt.plot(iterations, avg_f1_macros, marker='o')
plt.xlabel('Iteration')
plt.ylabel('Average F1 Macro')
plt.title('Grid Search for Hyperparameter Tuning')
plt.xticks(rotation=90)  # Rotate x-axis tick labels

plt.show()


# In[ ]:


all_results[10]


# In[ ]:


trainer = all_results[10]['trainers']


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

# Assuming you have evaluate_results dictionary containing macro and micro F1 scores for each language
languages = ['en', 'fr', 'ge', 'it', 'po', 'ru']

# Extract F1 scores for each language
f1_macro_scores = [best_evaluate_results[lang]['eval_f1_macro'] for lang in languages]
f1_micro_scores = [best_evaluate_results[lang]['eval_f1_micro'] for lang in languages]

# Set the width of the bars
bar_width = 0.35

# Set the positions for the bars on X-axis
r1 = np.arange(len(languages))
r2 = [x + bar_width for x in r1]

# Create grouped bar plot
bars1 = plt.bar(r1, f1_macro_scores, width=bar_width, alpha=0.8, label='F1 Macro')
bars2 = plt.bar(r2, f1_micro_scores, width=bar_width, alpha=0.8, label='F1 Micro')

# Add labels, title, and legend
plt.xlabel('Language', fontweight='bold')
plt.xticks([r + bar_width/2 for r in range(len(languages))], languages)
plt.ylabel('F1 Score')
plt.title('F1 Scores for Each Language (Macro and Micro)')
plt.legend()

for bar in bars1:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.01, round(yval, 3), ha='center', va='bottom')


for bar in bars2:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.01, round(yval, 3), ha='center', va='bottom')

# Show the plot
plt.show()


# In[ ]:





# # Predict

# In[ ]:


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


data_pred_dict = {}
root_folder = '/home/ubunti/datastore/riepin/Test_Rie/New_Project/0_data/raw/'
languages = [ 'en','es', 'fr', 'ge', 'gr', 'it', 'ka', 'po', 'ru']
label_mapping = {0: 'opinion', 1: 'satire', 2: 'reporting'}

# Replace the values in the 'label' column

for lang in languages:

  predict_folder = f"data/{lang}/test-articles-subtask-1/"

  df_pred_lang = make_dataframe(predict_folder, labels_folder = None)

  data_pred_dict[lang] = {'pred': df_pred_lang}

# All labels

pred_datasets = {}

for key,el in data_pred_dict.items():
  pred_dataset = Dataset.from_pandas(el['pred'], preserve_index=True)

  pred_datasets[key] = pred_dataset

pred_tokenized_datasets = {}

for key,el in pred_datasets.items():
    pred_tokenized_datasets[key] = el.map(tokenize_function, batched=True, remove_columns=["text"])

# pred_results = {}
languages = [ 'en','es', 'fr', 'ge', 'gr', 'it', 'ka', 'po', 'ru']

for language in languages:
    pred_ans = trainer.predict(pred_tokenized_datasets[language])

    max_indices = np.argmax(pred_ans[0], axis=1)
    indexes = data_pred_dict[language]['pred'].index.tolist()

    pred_ans_df = pd.DataFrame({'Index': indexes, 'Value': max_indices})

    pred_ans_df['Value'] = pred_ans_df['Value'].replace(label_mapping)


    # pred_results[language] = pred_ans_df

    pred_ans_df.to_csv(f'{language}_adapter_xml_large_seq_bn.txt', sep='\t', index=False, header=False)

