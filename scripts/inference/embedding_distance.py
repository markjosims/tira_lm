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
        text_columns: List[str],
        tokenizer: AutoTokenizer,
        max_length: int,
        device: torch.device = torch.device('cpu'),
    ):
        self.df = df
        self.text_columns = text_columns
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = {}
        for col in self.text_columns:
            item[col] = self.tokenizer(
                self.df.iloc[idx][col],
                return_tensors='pt',
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
            )['input_ids']
            # remove batch dimension and move to device
            item[col] = item[col].squeeze(0).to(self.device) 
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
        breakpoint()
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

    # Load Dataset
    data_path = getattr(cfg.data, 'path', None) or abx_sentence_list
    print(f"Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)

    # Tokenize Dataset and define DataLoader
    print("Initializing dataset and dataloader...")
    text_columns = [
        'sentence_a', 'sentence_b', 'sentence_x',
        'word_a', 'word_b', 'word_x',
    ]

    dataset = AbxDataset(
        df,
        text_columns,
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
