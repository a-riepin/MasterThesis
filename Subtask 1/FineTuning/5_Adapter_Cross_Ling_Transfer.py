
from sklearn.utils.class_weight import compute_class_weight
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

# Pytorch
import torch
from torch import nn, tensor

# Hugging face
from transformers import AutoTokenizer, AutoConfig, TrainingArguments

from datasets import Dataset, ClassLabel, DatasetDict, load_metric
import evaluate

from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedShuffleSplit, ParameterGrid
import matplotlib.pyplot as plt
from adapters.composition import Stack
from adapters import AutoAdapterModel, AdapterTrainer, AdapterConfig

def make_dataframe(input_folder, labels_folder=None):
    text = []
    for fil in tqdm(filter(lambda x: x.endswith('.txt'), os.listdir(input_folder))):
        iD, txt = fil[7:].split('.')[0], open(input_folder + fil, 'r', encoding='utf-8').read()
        text.append((iD, txt))
    df_text = pd.DataFrame(text, columns=['id','text']).set_index('id')
    df = df_text
    if labels_folder:
        labels = pd.read_csv(labels_folder, sep='\t', header=None)
        labels = labels.rename(columns={0:'id',1:'type'})
        labels.id = labels.id.apply(str)
        labels = labels.set_index('id')
        df = labels.join(df_text)[['text','type']]
    return df

def read_lang_data(train_folder, train_labels, val_folder, val_labels):
    df_train_lang = make_dataframe(train_folder, train_labels)
    df_train_lang = df_train_lang.rename(columns={"type" : "label"})
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
    results.update(accuracy_metric.compute(predictions=preds, references=labels))
    results.update(recall_metric.compute(predictions=preds, references=labels, average="macro"))
    results.update(precision_metric.compute(predictions=preds, references=labels, average="macro"))
    f1_macro = f1_score(labels, preds, average="macro")
    results.update({"eval_f1_macro": f1_macro})
    f1_micro = f1_score(labels, preds, average="micro")
    results.update({"eval_f1_micro": f1_micro})
    return results

data_dict = {}
root_folder = '/home/ubunti/datastore/riepin/Test_Rie/New_Project/0_data/raw/'
languages = ['en', 'fr', 'ge']

for lang in languages:
    train_folder = f"{root_folder}/{lang}/train-articles-subtask-1/"
    train_labels = f"{root_folder}/{lang}/train-labels-subtask-1.txt"
    dev_folder =  f"{root_folder}/{lang}/dev-articles-subtask-1/"
    dev_labels =  f"{root_folder}/{lang}/dev-labels-subtask-1.txt"
    df_train, df_dev = read_lang_data(train_folder, train_labels, dev_folder, dev_labels)
    data_dict[lang] = {'train': df_train, 'dev': df_dev, 'combined': pd.concat([df_train, df_dev])}

tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

label2id = {'opinion': 0, 'satire': 1, 'reporting': 2}
id2label = {0: 'opinion', 1: 'satire', 2: 'reporting'}
labels = ['opinion', 'satire', 'reporting']
ClassLabels = ClassLabel(num_classes=len(labels), names=labels)

def encode_batch(batch):
    """Encodes a batch of input data using the model tokenizer."""
    return tokenizer(batch["text"], max_length=80, truncation=True, padding="max_length")

def create_datasets(data_dict):
    datasets = {}
    for key, el in data_dict.items():
        combined_df = el['combined'].reset_index(drop=True)  # Reset index without creating __index_level_0__
        X = combined_df.index.values
        y = combined_df['label'].values

        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for train_index, test_index in sss.split(X, y):
            train_df = combined_df.iloc[train_index]
            test_df = combined_df.iloc[test_index]

        train_dataset = Dataset.from_pandas(train_df[['text', 'label']], preserve_index=False).cast_column("label", ClassLabels)
        test_dataset = Dataset.from_pandas(test_df[['text', 'label']], preserve_index=False).cast_column("label", ClassLabels)
        val_dataset = Dataset.from_pandas(el['dev'].reset_index(drop=True)[['text', 'label']], preserve_index=False).cast_column("label", ClassLabels)

        datasets[key] = DatasetDict({'train': train_dataset, 'test': test_dataset, 'val': val_dataset})
    return datasets

datasets = create_datasets(data_dict)

# Tokenize and format datasets
tokenized_datasets = {}
for key, el in datasets.items():
    dataset = el.map(encode_batch, batched=True)
    dataset = dataset.rename_column(original_column_name="label", new_column_name="labels")
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    tokenized_datasets[key] = dataset

def model_init_bert():
    config = AutoConfig.from_pretrained(
        "bert-base-multilingual-cased",
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )
    model = AutoAdapterModel.from_pretrained(
        "bert-base-multilingual-cased",
        config=config,
    )

    lang_adapter_config = AdapterConfig.load("seq_bn", reduction_factor=2)
    model.load_adapter("en/wiki@ukp", config=lang_adapter_config)
    model.load_adapter("de/wiki@ukp", config=lang_adapter_config)
    model.load_adapter("fr/wiki@ukp", config=lang_adapter_config)
    model.load_adapter("es/wiki@ukp", config=lang_adapter_config)
    model.load_adapter("ka/wiki@ukp", config=lang_adapter_config)
    model.load_adapter("el/wiki@ukp", config=lang_adapter_config)

    model.add_adapter("semeval_3_3", config="seq_bn")
    model.add_classification_head(
        "semeval_3_3",
        num_labels=len(label2id),
        id2label=id2label,
    )

    model.train_adapter(["semeval_3_3"])
    model.active_adapters = Stack("en", "de", "fr", "semeval_3_3")

    return model

adapter_bert = model_init_bert()

# Merging datasets for training
merged_train_set = pd.concat([el['train'].to_pandas() for el in tokenized_datasets.values()])
merged_test_set = pd.concat([el['test'].to_pandas() for el in tokenized_datasets.values()])
merged_val_set = pd.concat([el['val'].to_pandas() for el in tokenized_datasets.values()])

train_dataset = Dataset.from_pandas(merged_train_set)
test_dataset = Dataset.from_pandas(merged_test_set)
val_dataset = Dataset.from_pandas(merged_val_set)

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Grid Search and Training
hyperparameter_grid = {
    "num_train_epochs": [10, 15, 30],
    "learning_rate": [2e-5, 3e-5, 4e-5, 5e-5],
    "per_device_batch_size": [8,16],
}

best_params_global = None
best_f1_macro_global = 0.0
all_results = []
best_evaluate_results = {}

iterations = []
avg_f1_macros = []

for i, params in enumerate(ParameterGrid(hyperparameter_grid)):
    print('-----' * 4)
    print(f"Iteration {i + 1}: Training with hyperparameters {params}")

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
    evaluate_results = {}
    for lang in languages:
        ans_combined = trainer.evaluate(tokenized_datasets[lang]['val'])
        evaluate_results[lang] = ans_combined

    avg_f1_macro = sum(result['eval_f1_macro'] for result in evaluate_results.values()) / len(languages)

    all_results.append({
        'hyperparameters': params,
        'avg_f1_macro': avg_f1_macro,
        'individual_results': evaluate_results,
        'trainers': trainer
    })

    iterations.append(str(params))
    avg_f1_macros.append(avg_f1_macro)

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
plt.xticks(rotation=90)
plt.show()



all_results[1]

trainer = all_results[1]['trainers']

languages = ['en', 'fr', 'ge']

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

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

def predict_labels(language, pred_tokenized_dataset, trainer, label_mapping):
    # Activate the stack for zero-shot prediction
    if language == 'gr':
        trainer.model.active_adapters = Stack('el', "semeval_3_3")
    else:
        trainer.model.active_adapters = Stack(language, "semeval_3_3")

    pred_ans = trainer.predict(pred_tokenized_dataset)
    max_indices = np.argmax(pred_ans.predictions, axis=1)
    indexes = data_pred_dict[language]['pred'].index.tolist()
    pred_ans_df = pd.DataFrame({'Index': indexes, 'Value': max_indices})
    pred_ans_df['Value'] = pred_ans_df['Value'].replace(label_mapping)
    pred_ans_df.to_csv(f'{language}_adapter_cross_ling.txt', sep='\t', index=False, header=False)

# Load prediction data
data_pred_dict = {}
languages = ['es', 'ka', 'gr']
for lang in languages:
    predict_folder = f"{root_folder}/{lang}/test-articles-subtask-1/"
    df_pred_lang = make_dataframe(predict_folder)
    data_pred_dict[lang] = {'pred': df_pred_lang}

# Prepare datasets
pred_datasets = {key: Dataset.from_pandas(el['pred'], preserve_index=True) for key, el in data_pred_dict.items()}
pred_tokenized_datasets = {key: el.map(tokenize_function, batched=True, remove_columns=["text"]) for key, el in pred_datasets.items()}

# Use the existing trainer
trainer = all_results[1]['trainers']

# Predict and save results for each language
label_mapping = {0: 'opinion', 1: 'satire', 2: 'reporting'}
for language in languages:
    predict_labels(language, pred_tokenized_datasets[language], trainer, label_mapping)