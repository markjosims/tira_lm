"""
Inference script for generating contextual word embeddings for ABX
sentences and computing distance metrics between them.
"""

import hydra
import torch
import wandb
import os
from omegaconf import DictConfig, OmegaConf
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
)
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F

class ABXDataset(Dataset):
    def __init__(self, df, text_columns):
        self.df = df
        self.text_columns = text_columns

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = {}
        for col in self.text_columns:
            item[col] = self.df.iloc[idx][col+'_tokenized']
        return item
    
def get_word_token_indices(sentence_tokens, word_tokens):
    """
    Get the indices of the word tokens within the sentence tokens.
    Checks that the word tokens are a contiguous subsequence of the
    sentence tokens and that they only occur once.
    """
    sentence_tokens = sentence_tokens.tolist()
    word_tokens = word_tokens.tolist()
    
    found_word = False
    indices = []
    for i in range(len(sentence_tokens) - len(word_tokens) + 1):
        if sentence_tokens[i:i+len(word_tokens)] == word_tokens:
            if found_word:
                raise ValueError("Word tokens occur multiple times in the sentence.")
            indices.extend(range(i, i+len(word_tokens)))
            found_word = True

    if not found_word:
        raise ValueError("Word tokens not found in the sentence.")
    return indices

def get_batch_embeddings(model, batch):
    """
    Compute embeddings for each sentence in the batch, then
    get contextual word embeddings by averaging the token embeddings
    for the word tokens.
    """
    embeddings = {}
    for item in ['a', 'b', 'x']:
        sentence = 'sentence_' + item
        word = 'word_' + item
        outputs = model(input_ids=batch[sentence])
        sentence_embeddings = outputs.encoder_last_hidden_state
        batch_embeddings = []
        for sentence_embeddings, sentence_tokens, word_tokens in zip(
            sentence_embeddings, batch[sentence], batch[word]
        ):
            word_token_indices = get_word_token_indices(sentence_tokens, word_tokens)
            word_embedding = sentence_embeddings[word_token_indices].mean(dim=0)
            batch_embeddings.append(word_embedding)
        embeddings[item] = torch.stack(batch_embeddings)
    return embeddings

def score_batch(model, batch):
    """
    Compute embeddings for the batch and then compute cosine similarity
    between the contextual word embeddings of sentence_x and sentence_a,
    and sentence_x and sentence_b. Return a boolean tensor indicating whether
    sentence_x is closer to sentence_a than sentence_b, as well as the similarity scores.
    """
    embeddings = get_batch_embeddings(model, batch)
    a_x_similarity = F.cosine_similarity(embeddings['a'], embeddings['x'])
    b_x_similarity = F.cosine_similarity(embeddings['b'], embeddings['x'])
    scores = a_x_similarity > b_x_similarity
    return scores, a_x_similarity, b_x_similarity


@hydra.main(version_base="1.3", config_path="../../conf/mbart", config_name="embedding_comparison")
def main(cfg: DictConfig):
    # 1. Setup WandB
    os.environ["WANDB_PROJECT"] = cfg.wandb.project
    
    # 2. Load Model & Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model.name)

    # 3. Load Dataset
    df = pd.read_csv(cfg.data.path)

    # 4. Tokenize Dataset and define DataLoader
    text_columns = [
        'sentence_a', 'sentence_b', 'sentence_x',
        'word_a', 'word_b', 'word_x',
    ]
    for col in text_columns:
        df[col+'_tokenized'] = tokenizer(
            df[col].tolist(),
            return_tensors='pt',
            truncation=True,
            padding='max_length',
            max_length=cfg.model.max_length
        )['input_ids']

    dataset = ABXDataset(df, text_columns)
    dataloader = DataLoader(dataset, batch_size=cfg.inference.batch_size)
    
    # 5. Compute Embeddings and Distances

    all_scores = []
    all_a_x_similarities = []
    all_b_x_similarities = []

    for batch in dataloader:
        scores, a_x_similarity, b_x_similarity = score_batch(model, batch)
        all_scores.append(scores.cpu())
        all_a_x_similarities.append(a_x_similarity.cpu())
        all_b_x_similarities.append(b_x_similarity.cpu())
    
    all_scores = torch.cat(all_scores)
    all_a_x_similarities = torch.cat(all_a_x_similarities)
    all_b_x_similarities = torch.cat(all_b_x_similarities)

    # 6. Log Results to WandB
    wandb.log({
        'accuracy': all_scores.float().mean().item(),
        'a_x_similarity': all_a_x_similarities.mean().item(),
        'b_x_similarity': all_b_x_similarities.mean().item(),
    })