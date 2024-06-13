#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install -q -U --no-cache-dir gdown --pre')
get_ipython().system('pip install -q transformers datasets evaluate accelerate')
get_ipython().system('pip install -q requests nlpaug sentencepiece')

get_ipython().system('pip install -q sentence-transformers')
get_ipython().system('pip install -q bitsandbytes einops wandb')
get_ipython().system('pip install -q -U trl git+https://github.com/huggingface/peft.git')

get_ipython().system(' pip install -q einops')


# In[2]:


from google.colab import drive
drive.mount('/content/drive')


# In[3]:


get_ipython().system("unzip -qq '/content/drive/MyDrive/data.zip'")


# In[6]:


# Standard modules
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, pipeline, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
from datasets import Dataset, DatasetDict, ClassLabel
import random


# In[7]:


# function to make a DataFrame from text files
def make_dataframe(input_folder, labels_folder=None):
    text = []
    for fil in tqdm(filter(lambda x: x.endswith('.txt'), os.listdir(input_folder))):
        iD, txt = fil[7:].split('.')[0], open(input_folder + fil, 'r', encoding='utf-8').read()
        txt = txt.replace('\n\n', '.\n\n', 1)
        text.append((iD, txt))

    df_text = pd.DataFrame(text, columns=['id','text']).set_index('id')

    if labels_folder:
        labels = pd.read_csv(labels_folder, sep='\t', header=None)
        labels = labels.rename(columns={0:'id',1:'type'})
        labels.id = labels.id.apply(str)
        labels = labels.set_index('id')
        df = labels.join(df_text)[['text','type']]
    else:
        df = df_text

    return df

# function to read language data
def read_lang_data(train_folder, train_labels, val_folder, val_labels):
    df_train_lang = make_dataframe(train_folder, train_labels).rename(columns={"type" : "label"})
    df_val_lang = make_dataframe(val_folder, val_labels).rename(columns={"type" : "label"})
    return df_train_lang, df_val_lang


# In[8]:


# evaluation metrics
def eval_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    accuracy_metric = evaluate.load("accuracy")
    recall_metric = evaluate.load("recall")
    precision_metric = evaluate.load("precision")

    results = {}
    results.update(accuracy_metric.compute(predictions=preds, references=labels))
    results.update(recall_metric.compute(predictions=preds, references=labels, average="macro"))
    results.update(precision_metric.compute(predictions=preds, references=labels, average="macro"))

    f1_macro = f1_score(labels, preds, average="macro")
    f1_micro = f1_score(labels, preds, average="micro")
    results.update({"eval_f1_macro": f1_macro, "eval_f1_micro": f1_micro})

    return results


# In[9]:


# Load and prepare datasets
languages = ['en', 'fr', 'ge', 'it', 'po', 'ru']
data_dict = {}

for lang in languages:
    train_folder = f"data/{lang}/train-articles-subtask-1/"
    train_labels = f"data/{lang}/train-labels-subtask-1.txt"
    dev_folder = f"data/{lang}/dev-articles-subtask-1/"
    dev_labels = f"data/{lang}/dev-labels-subtask-1.txt"

    df_train, df_dev = read_lang_data(train_folder, train_labels, dev_folder, dev_labels)
    data_dict[lang] = {'combined': pd.concat([df_train, df_dev])}

datasets = {}

for key, el in data_dict.items():
    labels = ['opinion', 'satire', 'reporting']
    ClassLabels = ClassLabel(num_classes=len(labels), names=labels)
    dataset = Dataset.from_pandas(el['combined'], preserve_index=True)
    datasets[key] = DatasetDict({'train': dataset})

df_train_list = []

for key, el in datasets.items():
    df_train_list.append(el['train'].to_pandas())

merged_train_set = pd.concat(df_train_list)
merged_train_set['text'] = merged_train_set['text'].apply(lambda x: x.replace('\n', ' '))


# In[10]:


# Login to Hugging Face
from huggingface_hub import notebook_login
notebook_login()


# In[50]:


# Load and configure the model
#model_name = "meta-llama/Meta-Llama-3-8B"
#model_name = "OpenBuddy/openbuddy-falcon-7b-v5-fp16"
model_name = "mistralai/Mixtral-8x7B-v0.1"

config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, trust_remote_code=True)
model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Define the text generation pipeline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)


# In[56]:


config.max_position_embeddings


# In[51]:


# Prepare prediction datasets
data_pred_dict = {}
languages = ['en', 'es', 'fr', 'ge', 'gr', 'it', 'ka', 'po', 'ru']

for lang in languages:
    predict_folder = f"data/{lang}/test-articles-subtask-1/"
    df_pred_lang = make_dataframe(predict_folder)
    data_pred_dict[lang] = {'pred': df_pred_lang}


# In[89]:


# Split text into segments
def split_text(text, max_length=350):
    segments = []
    current_segment = ""
    for sentence in text.split("."):
        if len(current_segment) + len(sentence) + 1 <= max_length:
            current_segment += sentence + "."
        else:
            segments.append(current_segment.strip())
            current_segment = sentence + "."
    if current_segment:
        segments.append(current_segment.strip())
    return segments

# Generate training prompts for a specific language
def get_train_lang(lang, max_length=500, n_samples=2):
    all_prompts = []
    grouped_data = data_dict[lang]['combined'].groupby('label')
    check_prompts = []

    for label, group in grouped_data:
        group = group.sample(frac=1)
        selected_elements = group.head(n_samples)
        selected_values = selected_elements['text'].tolist()

        for text in selected_values:
            segments = split_text(text, max_length)
            segmented_text = segments[0]
            prompt = segmented_text + ' | class: ' + label
            check_prompts.append(prompt)

    processed_check_prompts = [x.replace('\n', ' ') for x in check_prompts]
    random.shuffle(processed_check_prompts)

    newline_text_check = "Text: " + "\nText: ".join(processed_check_prompts[:])
    newline_text_check = "\nCategory: ".join(newline_text_check.split(" | class: "))

    return newline_text_check


# In[90]:


languages = [ 'en', 'fr', 'ge', 'it', 'ru', 'po']
#languages = [ 'ru']
for lang in languages:
    # Get the training data for the language
    text = get_train_lang(lang)
    print(lang)
    #print(get_train_lang(lang))

    # Tokenize the text
    tokens = tokenizer(text)

    # Count the number of tokens
    num_tokens = len(tokens['input_ids'])

    # Print the number of tokens for the current language
    print(f"Number of tokens for {lang}: {num_tokens}")


# In[29]:


get_ipython().system('pip install -q googletrans==3.1.0a0')


# In[30]:


from googletrans import Translator
import json

translator = Translator()


# In[86]:


# Define max_length and n_samples as parameters
max_length = 500
n_samples = 3

# Perform predictions
all_dfs = []

for lang in tqdm(languages[:]):
    val_set = Dataset.from_pandas(data_pred_dict[lang]['pred'])
    df_ans = []

    for batch in val_set:
        text = batch['text']
        id_df = batch['id']

        segments = split_text(text, max_length)
        segmented_text = segments[0]
        processed_prompt_text = [x.replace('\n', ' ') for x in [segmented_text]]
        prompt_text = "\n Text: " + "\nText: ".join(processed_prompt_text)
        prompt_text += '\nCategory:'

        try:
            newline_text = get_train_lang(lang, max_length, n_samples)
        except:
            newline_text = get_train_lang('en', max_length, n_samples)

            texts = newline_text.split('Text: ')[1:]
            clean_texts = [text.split('\nCategory:')[0] for text in texts]
            translate_orig_texts = clean_texts

            dest_language = 'es' if lang == 'es' else 'el' if lang == 'gr' else 'ka' if lang == 'ka' else 'en'

            translated_texts = []
            for item in translate_orig_texts:
                translated_text = translator.translate(item, src="en", dest=dest_language).text
                translated_texts.append('Text: ' + translated_text)

            new_text = []
            index = 0
            segments = newline_text.split('\n')
            for i, s in enumerate(segments):
                if i <= (n_samples * 2) and i % 2 == 0:
                    new_text.append(translated_texts[index])
                    index += 1
                else:
                    new_text.append(s)
            final_new_text = '\n'.join(new_text)
            newline_text = final_new_text

        final_prompt = newline_text + prompt_text

        sequences = pipe(final_prompt, max_new_tokens=1, top_k=10)
        clean_label = sequences[0]['generated_text'].split('Category: ')[-1].strip()

        if 'opinion' in clean_label:
            clean_label = 'opinion'
        elif 'reporting' in clean_label:
            clean_label = 'reporting'
        elif 'satire' in clean_label or 'sat' in clean_label:
            clean_label = 'satire'
        else:
            clean_label = ''

        ans = {'id': id_df, 'label': clean_label}
        df_ans.append(ans)

    df = pd.DataFrame(df_ans)
    all_dfs.append(df)


# In[87]:


for i, l in enumerate(languages[:len(all_dfs)]):
    df_write = all_dfs[i]
    df_write.index = df_write['id']
    df_write = df_write[['label']]
    df_write.to_csv(f'mixtral_transl_{l}_{max_length}_{n_samples}.txt', sep='\t', header=False)

