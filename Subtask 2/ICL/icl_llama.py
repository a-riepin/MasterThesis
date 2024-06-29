# -*- coding: utf-8 -*-
"""ICL_Mixtral_ST2

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1PpMt8G6G1B83ZYopWYTHKqm86eBRiqfp
"""

!pip install -q transformers datasets evaluate accelerate sentence-transformers
!pip install -q -U --no-cache-dir gdown --pre
!pip install -q transformers datasets evaluate accelerate
!pip install -q requests nlpaug sentencepiece
!pip install -q datasets bitsandbytes einops wandb
!pip install -q -U trl transformers accelerate git+https://github.com/huggingface/peft.git

from google.colab import drive
drive.mount('/content/drive')

!unzip -qq '/content/drive/MyDrive/data.zip'

import os
import re
import string
import json
import numpy as np
import pandas as pd
from sklearn import metrics
from bs4 import BeautifulSoup
import transformers
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, AutoTokenizer, BertModel, BertConfig, AutoModel, AdamW
import warnings
from pathlib import Path
from tqdm import tqdm
from typing import Optional, List, Union, Tuple, Dict
import random

warnings.filterwarnings('ignore')

pd.set_option("display.max_columns", None)

base_data_path = Path("data")
data_folder = {
    "train": "train-articles-subtask-2",
    "dev": "dev-articles-subtask-2",
    "test": "test-articles-subtask-2"
}

labels_file_paths = {
    "train": "train-labels-subtask-2.txt",
    "dev": "dev-labels-subtask-2.txt",
    "test": "test-labels-subtask-2.txt"
}

collected_data = {
    "train": None,
    "dev": None,
    "test": None
}

def merge_database(old: Optional[pd.DataFrame], new: pd.DataFrame) -> pd.DataFrame:
    if old is None:
        return new
    return pd.concat(objs=[old, new], axis="index", ignore_index=False, verify_integrity=True, sort=False)

def make_dataframe(input_folder, labels_folder=None):
    #MAKE TXT DATAFRAME
    text = []
    for fil in tqdm(filter(lambda x: x.endswith('.txt'), os.listdir(input_folder))):
        iD, txt = fil[7:].split('.')[0], open(str(input_folder) + '/' + fil, 'r', encoding='utf-8').read()
        text.append((iD, txt))

    df_text = pd.DataFrame(text, columns=['id', 'text']).set_index('id')
    df = df_text

    #MAKE LABEL DATAFRAME
    if labels_folder:
        labels = pd.read_csv(labels_folder, sep='\t', header=None)
        labels = labels.rename(columns={0: 'id', 1: 'frames'})
        labels.id = labels.id.apply(str)
        labels = labels.set_index('id')

        #JOIN
        df = labels.join(df_text)[['text', 'frames']]
    return df

LANGUAGE_ABBREVIATION_TO_FULL = {
    "en": "english",
    "fr": "french",
    "it": "italian",
    "ru": "russian",
    "po": "polish",
    "ge": "german",
    "es": "spanish",
    "gr": "greek",
    "ka": "georgian",
    "es2en": "english",
    "fr2en": "english",
    "ge2en": "english",
    "gr2en": "english",
    "it2en": "english",
    "ka2en": "english",
    "po2en": "english",
    "ru2en": "english"
}

LANGUAGE_FULL_TO_ABBREVIATION = {
    "english": "en",
    "french": "fr",
    "italian": "it",
    "russian": "ru",
    "polish": "po",
    "german": "ge",
    "spanish": "es",
    "greek": "gr",
    "georgian": "ka"
}

def get_pure_language_abbreviation(language_tag: str) -> str:
    if "2" in language_tag:
        return language_tag.split(sep="2")[-1]
    return language_tag

languages = ['en', 'fr', 'ge', 'it', 'po', 'ru']

for language in languages:
    if len(language) > 2 and not language.endswith("2en"):
        language = LANGUAGE_FULL_TO_ABBREVIATION.get(language, language)
    for split in ("train", "dev"):
        input_folder = base_data_path.joinpath(language, data_folder[split])
        labels_file = base_data_path.joinpath(language, labels_file_paths[split])
        if not input_folder.exists():
            continue
        df = make_dataframe(
            input_folder=input_folder,
            labels_folder=str(labels_file.absolute()) if labels_file.exists() else None
        )
        df["language"] = language
        collected_data["train"] = merge_database(old=collected_data["train"], new=df)

df_train = collected_data["train"]

df_train['frames_list'] = df_train['frames'].apply(lambda x: [frame.strip() for frame in x.split(',')])

frames_list = [
    "Economic",
    "Capacity_and_resources",
    "Morality",
    "Fairness_and_equality",
    "Legality_Constitutionality_and_jurisprudence",
    "Policy_prescription_and_evaluation",
    "Crime_and_punishment",
    "Security_and_defense",
    "Health_and_safety",
    "Quality_of_life",
    "Cultural_identity",
    "Public_opinion",
    "Political",
    "External_regulation_and_reputation"
]

def idx2class(idx_list):
    return [frames_list.index(i) for i in idx_list]

df_train['Emotions'] = df_train['frames_list'].apply(idx2class)

for frame in frames_list:
    df_train[frame] = df_train['frames_list'].apply(lambda x: 1 if frame in x else 0)

df_train_final = df_train[['text', 'frames_list', 'language', 'frames']]

def split_text(text, max_length=800):
    segments = []
    current_segment = ""
    for sentence in text.split("."):
        if len(current_segment) + len(sentence) + 1 <= max_length:  # Add 1 for the dot
            current_segment += sentence + "."
        else:
            segments.append(current_segment.strip())
            current_segment = sentence + "."
    if current_segment:  # Add the last segment
        segments.append(current_segment.strip())
    return segments

def get_train_lang(language):
    all_prompts = []
    check_prompts = []
    group = df_train_final[df_train_final['language'] == language].sample(frac=1)
    selected_values = group.head(8)[['text', 'frames']]
    for index, row in selected_values.iterrows():
        label = ', '.join(row['frames'].split(','))
        text = row['text']
        segments = split_text(text)
        segmented_text = segments[0] if len(segments[0]) >= 10 else segments[1]
        prompt = f"{segmented_text} | class: {label}"
        check_prompts.append(prompt)
    processed_check_prompts = [x.replace('\n', ' ') for x in check_prompts]
    random.shuffle(processed_check_prompts)
    newline_text_check = "Text: " + "\nText: ".join(processed_check_prompts[:])
    newline_text_check = "\nCategory: ".join(newline_text_check.split(" | class: "))
    return newline_text_check



df_train_final.head()

def split_text(text, max_length):
    segments = []
    current_segment = ""
    for sentence in text.split("."):
        if len(current_segment) + len(sentence) + 1 <= max_length:  # Add 1 for the dot
            current_segment += sentence + "."
        else:
            segments.append(current_segment.strip())
            current_segment = sentence + "."
    if current_segment:  # Add the last segment
        segments.append(current_segment.strip())
    return segments

def get_train_lang(language, max_length, n_samples):
    all_prompts = []
    check_prompts = []

    group = df_train_final[df_train_final['language'] == language].sample(frac=1)
    selected_texts = []
    collected_frames = set()

    for index, row in group.iterrows():
        if len(selected_texts) >= n_samples:
            break

        frames = set(row['frames'].split(','))
        if not frames.issubset(collected_frames):
            selected_texts.append(row)
            collected_frames.update(frames)

    for row in selected_texts:
        label = ', '.join(row['frames'].split(','))
        text = row['text']
        segments = split_text(text, max_length=max_length)
        segmented_text = segments[0] if len(segments[0]) >= 10 else segments[1]
        prompt = f"{segmented_text} | class: {label}"
        check_prompts.append(prompt)

    processed_check_prompts = [x.replace('\n', ' ') for x in check_prompts]
    random.shuffle(processed_check_prompts)
    newline_text_check = "Text: " + "\nText: ".join(processed_check_prompts[:])
    newline_text_check = "\nCategory: ".join(newline_text_check.split(" | class: "))
    return newline_text_check

# Login to Hugging Face
from huggingface_hub import notebook_login
notebook_login()

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoConfig

model_name = "meta-llama/Meta-Llama-3-8B"

config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
#config.max_position_embeddings = 8192

model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, trust_remote_code=True)
model.config.use_cache = False

# Modify the max_position_embeddings parameter
#config.max_position_embeddings = 8184

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

languages = [ 'en', 'fr', 'ge', 'it', 'ru', 'po']
#languages = [ 'ru']
for lang in languages:
    # Get the training data for the language
    text = get_train_lang(lang, 1000, 8)

    print(len(text))


    # Tokenize the text
    tokens = tokenizer(text)

    # Count the number of tokens
    num_tokens = len(tokens['input_ids'])

    # Print the number of tokens for the current language
    print(f"Number of tokens for {lang}: {num_tokens}")

from transformers import pipeline

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)

test_data = pd.DataFrame()
languages = ['en', 'es', 'fr', 'ge', 'gr', 'it', 'ka', 'po', 'ru']

for language in languages:
    if len(language) > 2 and not language.endswith("2en"):
        language = LANGUAGE_FULL_TO_ABBREVIATION.get(language, language)
    split = "test"
    input_folder = base_data_path.joinpath(language, data_folder[split])
    labels_file = base_data_path.joinpath(language, labels_file_paths[split])
    if not input_folder.exists():
        continue
    df = make_dataframe(
        input_folder=input_folder,
        labels_folder=str(labels_file.absolute()) if labels_file.exists() else None
    )
    df["language"] = language
    test_data = merge_database(old=test_data, new=df)

import torch
from datasets import Dataset
from tqdm import tqdm
import pandas as pd
import os

torch.manual_seed(0)

all_dfs = []

max_length = 1500
n_samples = 14

def run_inference(languages, max_length, n_samples):
    save_dir = '/content/drive/My Drive'
    for lang in tqdm(languages):
        val_set = test_data[test_data['language'] == lang]

        # Create a dataset from the val_set DataFrame
        dataset = Dataset.from_pandas(val_set)

        def preprocess_function(examples):
            results = []
            for text in examples['text']:
                segments = split_text(text, max_length=max_length)
                segmented_text = segments[0]
                processed_prompt_text = [x.replace('\n', ' ') for x in [segmented_text]]
                prompt_text = "\n Text: " + "\nText: ".join(processed_prompt_text)
                prompt_text = prompt_text + '\nCategory:'

                try:
                    newline_text = get_train_lang(lang, max_length=max_length, n_samples=n_samples)
                except:
                    newline_text = get_train_lang('en', max_length=max_length, n_samples=n_samples)

                if len(newline_text) < 10:
                    newline_text = get_train_lang('en', max_length=max_length, n_samples=n_samples)

                final_prompt = newline_text + prompt_text
                results.append(final_prompt)
            return {'prompt_text': results}

        dataset = dataset.map(preprocess_function, batched=True)

        def infer_function(batch):
            sequences = pipe(
                batch['prompt_text'],
                max_new_tokens=100,
                top_k=10,
            )
            clean_labels = []
            for seq_batch in sequences:
                for seq in seq_batch:
                    clean_label = seq['generated_text'].split('Category: ')[-1].strip()
                    if '\n' in clean_label:
                        clean_label = clean_label.split('\n')[0].strip()
                    if 'Text' in clean_label:
                        clean_label = clean_label.split('Text')[0].strip()

                    new_list = [frame for frame in frames_list if frame in clean_label]
                    my_string = ','.join(new_list)
                    clean_labels.append(my_string)
            return {'label': clean_labels}

        results = dataset.map(infer_function, batched=True, batch_size=8)  # Adjust batch_size as needed

        df = pd.DataFrame({
            'id': val_set.index,
            'label': results['label']
        })

        all_dfs.append(df)

        # Save the DataFrame to Google Drive
        save_path = os.path.join(save_dir, f'{lang}_llama_{max_length}_{n_samples}.txt')
        df.to_csv(save_path, sep='\t', header=False, index=False)
        print(f"Saved results to {save_path}")

languages = ['en']

run_inference(languages=languages, max_length=max_length, n_samples=n_samples)