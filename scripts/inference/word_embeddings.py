"""
Inference script for generating word embeddings from Tira language dataset.
"""

import hydra
import torch
import os
from omegaconf import DictConfig
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
)
import pandas as pd
from scripts.constants import tira_word_list
from typing import List
from tqdm import tqdm
import re

from scripts.data_utils import TextDataset
from scripts.data_utils import HybridDataLoader
from scripts.inference.embedding_utils import get_encoder_outputs

@hydra.main(version_base="1.3", config_path="../../conf/mbart", config_name="embedding")
def main(cfg: DictConfig):
    
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
    data_path = getattr(cfg.data, 'path', None) or tira_word_list
    print(f"Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)

    # Tokenize Dataset and define DataLoader
    print("Initializing dataset and dataloader...")
    dataset = TextDataset(
        df=df,
        text_col='word',
        tokenizer=tokenizer,
        max_length=cfg.model.max_length,
        device=device
    )
    dataloader = HybridDataLoader(
        torch_dataset=dataset,
        string_dataset=df,
        batch_size=cfg.inference.batch_size,
    )
    
    # Compute Embeddings and Distances
    print("Computing embeddings...")

    word_embeddings = []
    for batch in tqdm(dataloader):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']

        with torch.no_grad():
            encoder_outputs = get_encoder_outputs(model, input_ids, attention_mask)
        
        word_embeddings.append(encoder_outputs.cpu())
        
    # Save embeddings locally
    output_path = os.path.join(cfg.outputs.save_dir, 'word_embeddings.pt')
    os.makedirs(cfg.outputs.save_dir, exist_ok=True)
    print(f"Saving embeddings to: {output_path}")
    torch.save(word_embeddings, output_path)


if __name__ == '__main__':
    main()
