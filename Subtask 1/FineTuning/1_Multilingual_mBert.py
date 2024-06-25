!pip install -q -U --no-cache-dir gdown --pre
!pip install -q transformers datasets evaluate accelerate
!pip install -q requests nlpaug sentencepiece

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
from sklearn.model_selection import StratifiedShuffleSplit
import torch

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
languages = ['en', 'fr', 'ge', 'it', 'po', 'ru']

for lang in languages:
    train_folder = f"{root_folder}/{lang}/train-articles-subtask-1/"
    train_labels = f"{root_folder}/{lang}/train-labels-subtask-1.txt"
    dev_folder =  f"{root_folder}/{lang}/dev-articles-subtask-1/"
    dev_labels =  f"{root_folder}/{lang}/dev-labels-subtask-1.txt"
    df_train, df_dev = read_lang_data(train_folder, train_labels, dev_folder, dev_labels)
    data_dict[lang] = {'train': df_train, 'dev': df_dev, 'combined': pd.concat([df_train, df_dev])}

display(data_dict['en']['train'])

labels = ['opinion', 'satire', 'reporting']
ClassLabels = ClassLabel(num_classes=len(labels), names=labels)

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

datasets['en']['train']

from sklearn.utils.class_weight import compute_class_weight

tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-uncased")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = {}
for key,el in datasets.items():
    tokenized_datasets[key] = el.map(tokenize_function, batched=True, remove_columns=["text"])

model = AutoModelForSequenceClassification.from_pretrained("bert-base-multilingual-uncased", num_labels=len(labels))

def model_init_bert():
    return AutoModelForSequenceClassification.from_pretrained("bert-base-multilingual-uncased", num_labels=len(labels))

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

y = train_dataset['label']

class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=np.asarray(y))
class_weights = tensor(class_weights, dtype=torch.float).cuda()
print(class_weights)

from torch.utils.data import DataLoader

def create_dataloader(dataset, batch_size, shuffle=False, num_workers=0, pin_memory=False):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)

from sklearn.model_selection import ParameterGrid
import matplotlib.pyplot as plt
from transformers import TrainingArguments, Trainer
import multiprocessing

languages = ['en', 'fr', 'ge', 'it', 'po', 'ru']

num_workers  = multiprocessing.cpu_count()

hyperparameter_grid = {
    "num_train_epochs": [10, 15],
    "learning_rate": [1e-5, 2e-5, 3e-5, 4e-5],
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
        save_strategy = "epoch",
        skip_memory_metrics=True,
        save_total_limit=1,
        load_best_model_at_end=True,
        num_train_epochs=params["num_train_epochs"],
        per_device_train_batch_size=params["per_device_batch_size"],
        per_device_eval_batch_size=params["per_device_batch_size"],
        learning_rate=params["learning_rate"],
        report_to="all",
    )

    train_loader = create_dataloader(train_dataset, batch_size=params["per_device_batch_size"], shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = create_dataloader(test_dataset, batch_size=params["per_device_batch_size"], num_workers=num_workers, pin_memory=True)

    trainer = Trainer(
        model_init=model_init_bert,
        args=training_args,
        train_dataset=train_loader.dataset,
        eval_dataset=val_loader.dataset,
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

print('\n' * 3)
print("Best global hyperparameters:", best_params_global)
print("Best global F1 macro:", best_f1_macro_global)

plt.plot(iterations, avg_f1_macros, marker='o')
plt.xlabel('Iteration')
plt.ylabel('Average F1 Macro')
plt.title('Grid Search for Hyperparameter Tuning')
plt.xticks(rotation=90)
plt.show()

for run in all_results:
    hyperparams = run['hyperparameters']
    avg_f1_macro = run['avg_f1_macro']

    print(f"Hyperparameters: {hyperparams}")
    print(f"Average F1 Macro: {avg_f1_macro}")

    for lang, metrics in run['individual_results'].items():
        eval_f1_macro = metrics['eval_f1_macro']
        print(f"Language: {lang}, Eval F1 Macro: {eval_f1_macro}")

    print('---' * 10)

trainer = all_results[0]['trainers']

import matplotlib.pyplot as plt
import numpy as np

languages = ['en', 'fr', 'ge', 'it', 'po', 'ru']

f1_macro_scores = [best_evaluate_results[lang]['eval_f1_macro'] for lang in languages]
f1_micro_scores = [best_evaluate_results[lang]['eval_f1_micro'] for lang in languages]

bar_width = 0.35

r1 = np.arange(len(languages))
r2 = [x + bar_width for x in r1]

bars1 = plt.bar(r1, f1_macro_scores, width=bar_width, alpha=0.8, label='F1 Macro')
bars2 = plt.bar(r2, f1_micro_scores, width=bar_width, alpha=0.8, label='F1 Micro')

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

plt.show()

data_pred_dict = {}
languages = [ 'en','es', 'fr', 'ge', 'gr', 'it', 'ka', 'po', 'ru']
label_mapping = {0: 'opinion', 1: 'satire', 2: 'reporting'}

for lang in languages:
    predict_folder = f"{root_folder}/{lang}/test-articles-subtask-1/"
    df_pred_lang = make_dataframe(predict_folder, labels_folder = None)
    data_pred_dict[lang] = {'pred': df_pred_lang}

pred_datasets = {key: Dataset.from_pandas(el['pred'], preserve_index=True) for key, el in data_pred_dict.items()}
pred_tokenized_datasets = {key: el.map(tokenize_function, batched=True, remove_columns=["text"]) for key, el in pred_datasets.items()}

for language in languages:
    pred_ans = trainer.predict(pred_tokenized_datasets[language])
    max_indices = np.argmax(pred_ans[0], axis=1)
    indexes = data_pred_dict[language]['pred'].index.tolist()
    pred_ans_df = pd.DataFrame({'Index': indexes, 'Value': max_indices})
    pred_ans_df['Value'] = pred_ans_df['Value'].replace(label_mapping)
    pred_ans_df.to_csv(f'ST1_hyperparam_{language}.txt', sep='\t', index=False, header=False)