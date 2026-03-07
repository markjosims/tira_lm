import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
import argparse
import os

csv_file = 'data/sentences.csv'
data_path = 'data/tokenized_datasets'
model_checkpoint = 'facebook/mbart-large-50-many-to-many-mmt'
tira_lang_code = "sw_KE"
english_lang_code = "en_XX"

def tokenize_data(df: pd.DataFrame, model_checkpoint: str) -> DatasetDict:
    print("\n🔄 Converting to HF Dataset format...")
    train_mask = df['split'] == 'train'
    val_mask = df['split'] == 'validation'

    train_dataset = Dataset.from_pandas(df[train_mask].reset_index(drop=True))
    val_dataset = Dataset.from_pandas(df[val_mask].reset_index(drop=True))
    datasets = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset
    })

    print(f"\n⬇️  Loading tokenizer ({model_checkpoint})...")
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=False)
    tokenizer.src_lang = tira_lang_code
    tokenizer.tgt_lang = english_lang_code

    def preprocess_function(examples):
        input_strs = examples['sentence']
        model_inputs = tokenizer(input_strs, max_length=128, padding="max_length", truncation=True)
        with tokenizer.as_target_tokenizer():
            label_strs = examples['translation']
            labels = tokenizer(label_strs, max_length=128, padding="max_length", truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    print("\n🔢 Tokenizing data...")
    tokenized_datasets = datasets.map(preprocess_function, batched=True, remove_columns=datasets['train'].column_names)
    return tokenized_datasets

def main():

    print("="*40)
    print("🛠️ Tira translation dataset tokenization")
    print("="*40)

    args = get_args()

    print(f"\n📂 Loading raw data from {args.csv_file}...")
    df = pd.read_csv(args.csv_file)
    tokenized_datasets = tokenize_data(df, args.model_checkpoint)

    train_path = os.path.join(args.data_path, 'train')
    val_path = os.path.join(args.data_path, 'validation')

    print(f"\n💾 Saving tokenized data to disk...")
    tokenized_datasets["train"].save_to_disk(train_path)
    tokenized_datasets["validation"].save_to_disk(val_path)
    print("\n✅ Data preparation complete!")

def get_args():
    parser = argparse.ArgumentParser(description="Prepare Tira Translation Data for mBART Training")
    parser.add_argument(
        '--sentences_list',
        type=str,
        default='data/sentences.csv',
        help='Path to save the processed sentences list CSV'
    )
    parser.add_argument(
        '--data_path',
        type=str,
        default=data_path,
        help='Path to save the tokenized datasets'
    )
    parser.add_argument(
        '--model-checkpoint',
        type=str,
        default=model_checkpoint,
        help='Pre-trained model checkpoint to use for tokenization'
    )
    return parser.parse_args()