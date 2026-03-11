import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, BatchEncoding
from scripts.constants import device
from typing import List, Dict, Union, Any


class AbxDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        sentence_columns: List[str],
        word_columns: List[str],
        word_index_columns: List[str],
        tokenizer: AutoTokenizer,
        max_length: int,
        device: torch.device = device,
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
    
class TextDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        text_col: str,
        tokenizer: AutoTokenizer,
        max_length: int,
        device: torch.device = device,
    ):
        self.df = df
        self.text_col = text_col
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.tokenizer(
            self.df.iloc[idx][self.text_col],
            return_tensors='pt',
            truncation=True,
                padding='max_length',
                max_length=self.max_length,
        )
        # remove batch dimension and put to device
        for key, value in item.items():
            item[key] = value.squeeze().to(self.device)]
        return item

class EmbeddingDataset(Dataset):
    """
    Dataset for storing pre-computed embeddings.
    """
    def __init__(
            self,
            embedding_path: str,
            device: torch.device = device,
        ):
        self.embeddings: Dict[str, torch.Tensor] = torch.load(
            embedding_path,
            map_location=device,
        )
        self.keys = list(self.embeddings.keys())
    
    def __len__(self):
        return self.embeddings[self.keys[0]].shape[0]
    
    def __getitem__(self, idx):
        return {key: self.embeddings[key][idx] for key in self.keys}
    
class SklearnDataset:
    """
    Load embedding dictionary and create positive and negative records
    by subtracting embeddings, such that word_a - word_x is positive and
    word_b - word_x is negative. This allows us to train a logistic regression
    classifier on the resulting dataset.
    """
    def __init__(
            self,
            embedding_path: str,
            device: torch.device = device,
        ):
        self.embeddings: Dict[str, torch.Tensor] = torch.load(
            embedding_path,
            map_location=device,
        )
        self.keys = list(self.embeddings.keys())
        X = []
        y = []
        positive_samples = self.embeddings['a'] - self.embeddings['x']
        negative_samples = self.embeddings['b'] - self.embeddings['x']
        for pos, neg in zip(positive_samples, negative_samples):
            X.append(pos.cpu().numpy())
            y.append(1)
            X.append(neg.cpu().numpy())
            y.append(0)
        self.X = np.array(X)
        self.y = np.array(y)


class HybridDataLoader:
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
            collate_fn=None,
            **kwargs
    ):
        if shuffle := kwargs.get('shuffle', False):
            raise ValueError(
                "Shuffling is not supported in HybridDataLoader to maintain alignment between"\
                " tensors and strings."
            )

        if isinstance(torch_dataset, AbxDataset):
            collate_fn = AbxDataset.collate_batch

        self.dataloader = DataLoader(
            torch_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            **kwargs
        )

        self.batch_size = batch_size
        self.string_dataset = string_dataset

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        for i, batch in enumerate(self.dataloader):
            start_idx = i * self.batch_size
            end_idx = start_idx + self.batch_size
            string_batch = self.string_dataset.iloc[start_idx:end_idx]
            string_batch = string_batch.to_dict(orient='list')
            batch['strings'] = string_batch
            yield batch