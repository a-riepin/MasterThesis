#!/usr/bin/env python
# coding: utf-8

# In[1]:


from google.colab import drive
drive.mount('/content/drive')


# In[2]:


get_ipython().system("unzip -qq '/content/drive/MyDrive/data.zip'")


# In[3]:


get_ipython().system('pip install -q -U --no-cache-dir gdown --pre')
get_ipython().system('pip install -q transformers datasets evaluate accelerate')
get_ipython().system('pip install -q requests nlpaug sentencepiece')
get_ipython().system('pip install torch -U')


# In[4]:


get_ipython().system('pip install -q googletrans==3.1.0a0')


# In[5]:


# Standard modules
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

# Pytorch
from torch import nn, tensor

# Hugging face
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset, ClassLabel, DatasetDict, load_metric
import evaluate
from sklearn.metrics import f1_score, precision_score, recall_score
import googletrans
from googletrans import Translator
from sklearn.utils.class_weight import compute_class_weight
import torch
from sklearn.model_selection import ParameterGrid
import matplotlib.pyplot as plt


# In[6]:


# Function to create dataframes
def make_dataframe(input_folder, labels_folder=None):
    text = []
    for fil in tqdm(filter(lambda x: x.endswith('.txt'), os.listdir(input_folder))):
        iD, txt = fil[7:].split('.')[0], open(input_folder + fil, 'r', encoding='utf-8').read()
        text.append((iD, txt))
    df_text = pd.DataFrame(text, columns=['id', 'text']).set_index('id')
    df = df_text

    if labels_folder:
        labels = pd.read_csv(labels_folder, sep='\t', header=None)
        labels = labels.rename(columns={0: 'id', 1: 'type'})
        labels.id = labels.id.apply(str)
        labels = labels.set_index('id')
        df = labels.join(df_text)[['text', 'type']]

    return df

# Function to read language data
def read_lang_data(train_folder, train_labels, val_folder, val_labels):
    df_train_lang = make_dataframe(train_folder, train_labels).rename(columns={"type": "label"})
    df_val_lang = make_dataframe(val_folder, val_labels).rename(columns={"type": "label"})
    return df_train_lang, df_val_lang


# In[7]:


# Evaluation metrics
recall_metric = evaluate.load("recall")
precision_metric = evaluate.load("precision")
accuracy_metric = evaluate.load("accuracy")

def eval_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    results = accuracy_metric.compute(predictions=preds, references=labels)
    results.update(recall_metric.compute(predictions=preds, references=labels, average="macro"))
    results.update(precision_metric.compute(predictions=preds, references=labels, average="macro"))
    results.update({
        "eval_f1_macro": f1_score(labels, preds, average="macro"),
        "eval_f1_micro": f1_score(labels, preds, average="micro")
    })
    return results


# In[8]:



# Data loading
def load_data():
    data_dict = {}
    languages = ['en', 'fr', 'ge', 'it', 'po', 'ru']
    for lang in languages:
        train_folder = f"data/{lang}/train-articles-subtask-1/"
        train_labels = f"data/{lang}/train-labels-subtask-1.txt"
        dev_folder = f"data/{lang}/dev-articles-subtask-1/"
        dev_labels = f"data/{lang}/dev-labels-subtask-1.txt"
        df_train, df_dev = read_lang_data(train_folder, train_labels, dev_folder, dev_labels)
        data_dict[lang] = {'train': df_train, 'dev': df_dev, 'combined': pd.concat([df_train, df_dev])}
    return data_dict


# In[9]:


# Create datasets
def create_datasets(data_dict):
    datasets = {}
    labels = ['opinion', 'satire', 'reporting']
    ClassLabels = ClassLabel(num_classes=len(labels), names=labels)
    for key, el in data_dict.items():
        dataset = Dataset.from_pandas(el['combined'], preserve_index=True).cast_column("label", ClassLabels).train_test_split(test_size=0.2)
        val_dataset = Dataset.from_pandas(el['dev'], preserve_index=True).cast_column("label", ClassLabels)
        datasets[key] = DatasetDict({'train': dataset['train'], 'test': dataset['test'], 'val': val_dataset})
    return datasets


# In[10]:


# Google Translator augmentation
def augment_data(datasets):
    translator = Translator()
    en_labeled_dataset = datasets['en']['train'].concatenate(datasets['en']['test'])
    to_translate_en_report_satire = en_labeled_dataset.filter(lambda example: example['label'] != 0)
    destinations = ['zh-CN', 'ja']
    new_items = []
    for destination in destinations:
        for article in to_translate_en_report_satire:
            translated_text = translator.translate(article['text'], src='en', dest=destination).text
            new_text = translator.translate(translated_text, src=destination, dest='en').text
            new_article = {'label': article['label'], 'text': new_text}
            new_items.append(new_article)
    for item in new_items:
        datasets['en']['train'] = datasets['en']['train'].add_item(item)
    return datasets


# In[11]:


# Tokenization
def tokenize_datasets(datasets, tokenizer):
    tokenized_datasets = {}
    for key, el in datasets.items():
        tokenized_datasets[key] = el.map(lambda examples: tokenizer(examples["text"], padding="max_length", truncation=True), batched=True, remove_columns=["text"])
    return tokenized_datasets


# In[14]:


def model_init_roberta():
    return AutoModelForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels=3)


# In[20]:


# Main script
def main():
    data_dict = load_data()
    datasets = create_datasets(data_dict)
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    tokenized_datasets = tokenize_datasets(datasets, tokenizer)

    merged_train_set = pd.concat([el['train'].to_pandas() for el in tokenized_datasets.values()])
    merged_test_set = pd.concat([el['test'].to_pandas() for el in tokenized_datasets.values()])
    merged_val_set = pd.concat([el['val'].to_pandas() for el in tokenized_datasets.values()])

    train_dataset = Dataset.from_pandas(merged_train_set)
    test_dataset = Dataset.from_pandas(merged_test_set)
    val_dataset = Dataset.from_pandas(merged_val_set)

    y = train_dataset['label']
    labels = ['opinion', 'satire', 'reporting']

    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=np.asarray(y))
    class_weights = tensor(class_weights, dtype=torch.float).cuda()

    # Define the model initialization function
    def model_init_roberta():
        return AutoModelForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels=len(labels))

    # Define the hyperparameter grid for tuning
    hyperparameter_grid = {
        "num_train_epochs": [2,3],
        "learning_rate": [3e-5],
        "per_device_batch_size": [8],
    }

    best_params_global = None
    best_f1_macro_global = 0.0
    best_model_for_english = 0.0
    all_results = []

    # Lists to store results for plotting
    iterations = []
    avg_f1_macros = []

    # Perform grid search
    for i, params in enumerate(ParameterGrid(hyperparameter_grid)):
        print('-----'*4)
        print(f"Iteration {i + 1}: Training with hyperparameters {params}")

        # Training and evaluating bert
        training_args = TrainingArguments(
            output_dir="output_trainer",
            evaluation_strategy="epoch",
            logging_strategy="epoch",
            skip_memory_metrics=True,
            num_train_epochs=params["num_train_epochs"],
            per_device_train_batch_size=params["per_device_batch_size"],
            per_device_eval_batch_size=params["per_device_batch_size"],
            learning_rate=params["learning_rate"],
            report_to="all",
            fp16=True
        )

        trainer = Trainer(
            model_init=model_init_roberta,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=eval_metrics,
        )

        trainer.train()
        # Evaluate on the combined validation set for parameter tuning
        evaluate_results = {}
        for lang in datasets.keys():
            ans_combined = trainer.evaluate(tokenized_datasets[lang]['val'])
            evaluate_results[lang] = ans_combined

        # Calculate average F1 macro across all languages
        avg_f1_macro = sum(result['eval_f1_macro'] for result in evaluate_results.values()) / len(datasets)

        all_results.append({
            'hyperparameters': params,
            'avg_f1_macro': avg_f1_macro,
            'individual_results': evaluate_results,
            'trainer': trainer
        })

        # Update best parameters if the current set performs better globally
        if avg_f1_macro > best_f1_macro_global:
            best_f1_macro_global = avg_f1_macro
            best_params_global = params
            best_index = i

        # Check if the F1 macro for English is the highest
        if evaluate_results['en']['eval_f1_macro'] > best_model_for_english:
            best_model_for_english = evaluate_results['en']['eval_f1_macro']
            best_model_params_for_english = params
            best_model_trainer_for_english = trainer

    print('\n'*3)
    print("Best global hyperparameters:", best_params_global)
    print("Best global F1 macro:", best_f1_macro_global)
    print("Best model index:", best_index)
    print("Best model hyperparameters for English:", best_model_params_for_english)
    print("Best F1 macro for English:", best_model_for_english)

    # Visualize results
    plt.plot(range(len(all_results)), [result['avg_f1_macro'] for result in all_results], marker='o')
    plt.xlabel('Iteration')
    plt.ylabel('Average F1 Macro')
    plt.title('Grid Search for Hyperparameter Tuning')
    plt.xticks(rotation=90)  # Rotate x-axis tick labels

    plt.show()


    # Save the index of the model with the best parameters to a variable
    best_model_index = best_index

    # Set trainer to the best model trainer
    trainer = all_results[best_model_index]['trainer']

    # Prediction code
    data_pred_dict = {}
    languages = [ 'en','es', 'fr', 'ge', 'gr', 'it', 'ka', 'po', 'ru']
    label_mapping = {0: 'opinion', 1: 'satire', 2: 'reporting'}

    for lang in languages:
        predict_folder = f"data/{lang}/test-articles-subtask-1/"
        df_pred_lang = make_dataframe(predict_folder, labels_folder=None)
        data_pred_dict[lang] = {'pred': df_pred_lang}

    pred_datasets = {key: Dataset.from_pandas(el['pred'], preserve_index=True) for key, el in data_pred_dict.items()}
    pred_tokenized_datasets = {key: el.map(lambda examples: tokenizer(examples["text"], padding="max_length", truncation=True), batched=True, remove_columns=["text"]) for key, el in pred_datasets.items()}

    for language in languages:
        pred_ans = trainer.predict(pred_tokenized_datasets[language])
        max_indices = np.argmax(pred_ans[0], axis=1)
        indexes = data_pred_dict[language]['pred'].index.tolist()
        pred_ans_df = pd.DataFrame({'Index': indexes, 'Value': max_indices})
        pred_ans_df['Value'] = pred_ans_df['Value'].replace(label_mapping)
        pred_ans_df.to_csv(f'{language}_augmented_all_best.txt', sep='\t', index=False, header=False)


# In[21]:


if __name__ == "__main__":
    main()


# In[ ]:




