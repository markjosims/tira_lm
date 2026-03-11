"""
Inference script for generating contextual word embeddings for Abx
sentences and computing distance metrics between them.
"""

import hydra
import torch
import wandb
import os
from omegaconf import DictConfig, OmegaConf
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
)
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F
from scripts.constants import abx_sentence_list, device
from typing import List

class AbxDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        sentence_columns: List[str],
        word_columns: List[str],
        tokenizer: AutoTokenizer,
        max_length: int,
        device: torch.device = torch.device('cpu'),
    ):
        self.df = df
        self.sentence_columns = sentence_columns
        self.word_columns = word_columns
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = {}
        for col in self.sentence_columns:
            item[col] = self.tokenizer(
                self.df.iloc[idx][col],
                return_tensors='pt',
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
            ).to(self.device)
        for col in self.word_columns:
            slice_str = self.df.iloc[idx][col]
            start_index, end_index = slice_str.split(':')
            start_index, end_index = int(start_index), int(end_index)
            word_range = torch.tensor([start_index, end_index], device=self.device)
            item[col] = word_range
        breakpoint()
        return item

def collate_batch(batch_list: List[Dict[str, Any]]):
    ...
    
def get_word_token_indices(token_encoding, word_range):
    """
    Get the indices of the word tokens within the sentence tokens.
    Checks that the word tokens are a contiguous subsequence of the
    sentence tokens and that they only occur once.
    """
    word_start, word_end = word_range
    word_ids = token_encoding.word_ids
    target_word_indices = []
    for i, word_id in enumerate(word_ids):
        if word_id is None:
            continue
        if (word_id >= word_start) and (word_id <= word_end):
            target_word_indices.append(i)

    if not target_word_indices:
        raise ValueError("Word tokens not found in the sentence.")

    return target_word_indices

def get_batch_embeddings(model, batch):
    """
    Compute embeddings for each sentence in the batch, then
    get contextual word embeddings by averaging the token embeddings
    for the word tokens.
    """
    embeddings = {}
    for item in ['a', 'b', 'x']:
        sentence = 'sentence_' + item
        word_idx = f'word_{item}_index'
        with torch.no_grad():
            outputs = model(input_ids=batch[sentence]['input_ids'].squeeze())
        sentence_embeddings = outputs.encoder_last_hidden_state
        batch_embeddings = []
        for i, word_indices in enumerate(batch[word_idx]):
            breakpoint()
            # TODO: only one batch encoding is getting returned per batch of 64! investigate
            token_encoding = batch[sentence][i]
            word_token_indices = get_word_token_indices(token_encoding, word_indices)
            record_embeddings = sentence_embeddings[i].squeeze()
            word_embedding = record_embeddings[word_token_indices].mean(dim=0)
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
    # Setup WandB
    print(f"Using WandB project: {cfg.wandb.project}")
    os.environ["WANDB_PROJECT"] = cfg.wandb.project
    
    # Load Model & Tokenizer
    print(f"Loading model and tokenizer from: {cfg.model.local_path}")
    print(f"For base model {cfg.model.name}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.local_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model.local_path)

    # Put model to device
    if hasattr(cfg.model, 'device'):
        device = torch.device(cfg.model.device)
    model.to(device)
    model.eval()

    # Load Dataset
    data_path = getattr(cfg.data, 'path', None) or abx_sentence_list
    print(f"Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)

    # Tokenize Dataset and define DataLoader
    print("Initializing dataset and dataloader...")
    sentence_columns = [
        'sentence_a', 'sentence_b', 'sentence_x',
    ]
    word_columns = [
        'word_a_index', 'word_b_index', 'word_x_index',
    ]

    dataset = AbxDataset(
        df,
        sentence_columns,
        word_columns,
        tokenizer,
        cfg.model.max_length,
        device=device,
    )
    dataloader = DataLoader(dataset, batch_size=cfg.inference.batch_size)
    
    # Compute Embeddings and Distances
    print("Computing embeddings and distances...")

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

if __name__ == '__main__':
    main()
