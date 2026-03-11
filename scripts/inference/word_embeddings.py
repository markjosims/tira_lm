"""
Inference script for generating contextual word embeddings for Abx
sentences and computing distance metrics between them.
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
from scripts.constants import abx_sentence_list
from typing import List
from tqdm import tqdm
import re

from scripts.data_utils import AbxDataset
from scripts.data_utils import HybridDataLoader

def get_word_token_indices(batch_index, batch, record_type, tokenizer):
    """
    Get the indices of the word tokens within the sentence tokens.
    Calls either `get_word_token_indices_slow_tokenizer` or
    `get_word_token_indices_fast_tokenizer` depending on whether the
    tokenizer provides word_ids.
    """
    sentence_key = f'sentence_{record_type}'
    word_key = f'word_{record_type}'
    word_index_key = f'word_{record_type}_index'
    if batch[sentence_key].encodings is None:
        assert not tokenizer.is_fast,\
        "Tokenizer does not provide encodings but is a fast tokenizer, check tokenizer configuration."
        sentence = batch['strings'][sentence_key][batch_index]
        word = batch['strings'][word_key][batch_index]
        return get_word_token_indices_slow_tokenizer(sentence, word)
    
    sentence_encoding = batch[sentence_key].encoding[batch_index]
    word_range = batch[word_index_key][batch_index].tolist()
    return get_word_token_indices_fast_tokenizer(sentence_encoding, word_range)    

def get_word_token_indices_slow_tokenizer(sentence, word) -> List[int]:
    """
    Gets the expected indices of the word in the sentence by using their
    utf-8 bytes.
    """
    sentence_bytes = sentence.encode('utf-8')
    word_bytes = word.encode('utf-8')

    matches = re.finditer(word_bytes, sentence_bytes)
    matches = list(matches)
    assert len(matches)==1, f"Expected one occurrence of {word} in {sentence} but got {len(matches)}"
    start = matches[0].start()
    end = matches[0].end()
    return list(range(start, end))


def get_word_token_indices_fast_tokenizer(token_encoding, word_range):
    """
    Use the word_ids provided by the fast tokenizer to find the indices of the
    word tokens in the sentence tokens.
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

def get_encoder_outputs(model, input_ids) -> torch.Tensor:
    """
    Get the encoder outputs for the given input_ids.
    Syntax for accessing encoder outputs differs based on whether
    the model exposes the encoder as an attribute (e.g. ByT5)
    or not (BART).
    """
    if hasattr(model, 'encoder'):
        with torch.no_grad():
            outputs = model.encoder(input_ids=input_ids)
        encoder_out = outputs.last_hidden_state.to('cpu')
    else:
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
        encoder_out = outputs.encoder_last_hidden_state.to('cpu')
    del outputs
    return encoder_out

def get_batch_embeddings(model, tokenizer, batch):
    """
    Compute embeddings for each sentence in the batch, then
    get contextual word embeddings by averaging the token embeddings
    for the word tokens.
    """

    embeddings = {}
    for item in ['a', 'b', 'x']:
        sentence = 'sentence_' + item
        sentence_embeddings = get_encoder_outputs(model, batch[sentence]['input_ids'])
        del outputs
        batch_embeddings = []
        batch_size = batch[sentence]['input_ids'].shape[0]
        for i in range(batch_size):
            word_token_indices = get_word_token_indices(i, batch, item, tokenizer)
            record_embeddings = sentence_embeddings[i].squeeze()
            word_embedding = record_embeddings[word_token_indices].mean(dim=0)
            batch_embeddings.append(word_embedding)
        embeddings[item] = torch.stack(batch_embeddings)
    return embeddings

@hydra.main(version_base="1.3", config_path="../../conf/mbart", config_name="embedding_comparison")
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
    data_path = getattr(cfg.data, 'path', None) or abx_sentence_list
    print(f"Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)

    # Tokenize Dataset and define DataLoader
    print("Initializing dataset and dataloader...")
    sentence_columns = []
    word_columns = []
    word_index_columns = []
    for item in ('a', 'b', 'x'):
        sentence_columns.append(f'sentence_{item}')
        word_columns.append(f'word_{item}')
        word_index_columns.append(f'word_{item}_index')

    dataset = AbxDataset(
        df,
        sentence_columns,
        word_columns,
        word_index_columns,
        tokenizer,
        cfg.model.max_length,
        device=device,
    )
    dataloader = HybridDataLoader(
        torch_dataset=dataset,
        string_dataset=df,
        batch_size=cfg.inference.batch_size,
    )
    
    # Compute Embeddings and Distances
    print("Computing embeddings...")

    embeddings = {}
    for batch in tqdm(dataloader):
        batch_embeddings = get_batch_embeddings(model, tokenizer, batch)
        for key in batch_embeddings:
            if key not in embeddings:
                embeddings[key] = []
            embeddings[key] = torch.cat(
                [embeddings[key], batch_embeddings[key].cpu()],
                dim=0
            )
        
    # Save embeddings locally
    output_path = os.path.join(output_path, 'abx_word_embeddings.pt')
    os.makedirs(output_path, exist_ok=True)
    print(f"Saving embeddings to: {output_path}")
    torch.save(embeddings, output_path)


if __name__ == '__main__':
    main()
