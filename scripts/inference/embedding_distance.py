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
    BatchEncoding,
)
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F
from scripts.constants import abx_sentence_list, device
from typing import List, Dict, Any, Union
from tqdm import tqdm
import re

class AbxDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        sentence_columns: List[str],
        word_columns: List[str],
        word_index_columns: List[str],
        tokenizer: AutoTokenizer,
        max_length: int,
        device: torch.device = torch.device('cpu'),
    ):
        self.df = df
        self.sentence_columns = sentence_columns
        self.word_columns = word_columns
        self.word_index_columns = word_index_columns
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = {}
        for col in self.sentence_columns + self.word_columns:
            item[col] = self.tokenizer(
                self.df.iloc[idx][col],
                return_tensors='pt',
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
            ).to(self.device)
        for col in self.word_index_columns:
            slice_str = self.df.iloc[idx][col]
            start_index, end_index = slice_str.split(':')
            start_index, end_index = int(start_index), int(end_index)
            word_range = torch.tensor([start_index, end_index], device=self.device)
            item[col] = word_range
        return item

class HybridDataLoader(DataLoader):
    """
    Custom DataLoader that yields batches of Torch tensors along with the
    corresponding string literals from the original dataset. This allows us to
    compute embeddings using the tensors while also having access to the original
    sentences and words.
    """
    def __init__(
            self,
            torch_dataset,
            string_dataset: pd.DataFrame,
            batch_size: int,
            **kwargs
    ):
        if shuffle := kwargs.get('shuffle', False):
            raise ValueError(
                "Shuffling is not supported in HybridDataLoader to maintain alignment between"\
                " tensors and strings."
            )
        super().__init__(
            torch_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=HybridDataLoader.collate_batch,
            **kwargs
        )

        self.batch_size = batch_size
        self.string_dataset = string_dataset

    def __iter__(self):
        for i, batch in enumerate(super().__iter__()):
            start_idx = i * self.batch_size
            end_idx = start_idx + self.batch_size
            string_batch = self.string_dataset.iloc[start_idx:end_idx]
            string_batch = string_batch.to_dict(orient='list')
            batch['strings'] = string_batch
            yield batch, string_batch

    @staticmethod
    def collate_batch(
            batch_list: List[Dict[str, Any]]
    ) -> Dict[str, Union[BatchEncoding, torch.Tensor]]:
        collated_batch = {}
        for key in batch_list[0].keys():
            # sentence_(abx) and word_(abx)
            if key.startswith('sentence_') or (key.startswith('word_') and not key.endswith('_index')):
                data = {
                    'input_ids': torch.stack([item[key]['input_ids'].squeeze() for item in batch_list]),
                    'attention_mask': torch.stack([item[key]['attention_mask'].squeeze() for item in batch_list]),
                }
                encodings = None
                if hasattr(batch_list[0][key], 'encodings') and batch_list[0][key].encodings:
                    encodings = [item[key].encodings[0] for item in batch_list]
                collated_batch[key] = BatchEncoding(data=data, encoding=encodings)
            # word_(abx)_index
            else:
                collated_batch[key] = torch.stack([item[key] for item in batch_list])
        return collated_batch
    
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

def get_batch_embeddings(model, tokenizer, batch):
    """
    Compute embeddings for each sentence in the batch, then
    get contextual word embeddings by averaging the token embeddings
    for the word tokens.
    """

    embeddings = {}
    for item in ['a', 'b', 'x']:
        sentence = 'sentence_' + item

        with torch.no_grad():
            outputs = model.encoder(input_ids=batch[sentence]['input_ids'].squeeze())
        sentence_embeddings = outputs.last_hidden_state.to('cpu')
        del outputs
        batch_embeddings = []
        batch_size = batch[sentence]['input_ids'].shape[0]
        for i in tqdm(range(batch_size), total=batch_size):
            word_token_indices = get_word_token_indices(i, batch, item, tokenizer)
            record_embeddings = sentence_embeddings[i].squeeze()
            word_embedding = record_embeddings[word_token_indices].mean(dim=0)
            batch_embeddings.append(word_embedding)
        embeddings[item] = torch.stack(batch_embeddings)
    return embeddings

def score_batch(model, tokenizer, batch):
    """
    Compute embeddings for the batch and then compute cosine similarity
    between the contextual word embeddings of sentence_x and sentence_a,
    and sentence_x and sentence_b. Return a boolean tensor indicating whether
    sentence_x is closer to sentence_a than sentence_b, as well as the similarity scores.
    """
    embeddings = get_batch_embeddings(model, tokenizer, batch)
    a_x_similarity = F.cosine_similarity(embeddings['a'], embeddings['x'])
    b_x_similarity = F.cosine_similarity(embeddings['b'], embeddings['x'])
    scores = a_x_similarity > b_x_similarity
    return scores, a_x_similarity, b_x_similarity

@hydra.main(version_base="1.3", config_path="../../conf/mbart", config_name="embedding_comparison")
def main(cfg: DictConfig):
    # Setup WandB
    print(f"Using WandB project: {cfg.wandb.project}")
    os.environ["WANDB_PROJECT"] = cfg.wandb.project
    wandb.init()
    
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
    print("Computing embeddings and distances...")

    all_scores = []
    all_a_x_similarities = []
    all_b_x_similarities = []

    for batch in tqdm(dataloader):
        scores, a_x_similarity, b_x_similarity = score_batch(model, tokenizer, batch)
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
