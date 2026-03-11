import os

import hydra
from omegaconf import DictConfig
from tqdm import tqdm
import wandb
import pandas as pd
import torch
from torch.nn import functional as F
from scripts.data_utils import EmbeddingDataset, HybridDataLoader

def score_batch(batch) -> pd.DataFrame:
    """
    Compute embeddings for the batch and then compute cosine similarity
    between the contextual word embeddings of sentence_x and sentence_a,
    and sentence_x and sentence_b. Return a boolean tensor indicating whether
    sentence_x is closer to sentence_a than sentence_b, as well as the similarity scores.
    """
    a_x_similarity = F.cosine_similarity(batch['a'], batch['x'])
    b_x_similarity = F.cosine_similarity(batch['b'], batch['x'])
    scores = a_x_similarity > b_x_similarity
    
    report = pd.DataFrame(batch['strings'])
    report['a_x_similarity'] = a_x_similarity.cpu().numpy()
    report['b_x_similarity'] = b_x_similarity.cpu().numpy()
    report['score'] = scores.cpu().numpy()
    return report

@hydra.main(version_base="1.3", config_path="../../conf/mbart", config_name="embedding_comparison")
def main(cfg: DictConfig):
    # Setup WandB
    print(f"Using WandB project: {cfg.wandb.project}")
    os.environ["WANDB_PROJECT"] = cfg.wandb.project
    wandb.init()

    # Load embeddings and text data
    print(f"Loading embeddings from: {cfg.data.embed_path}")
    embedding_dataset = EmbeddingDataset(cfg.data.embed_path)

    print(f"Loading text data from: {cfg.data.csv_path}")
    text_data = pd.read_csv(cfg.data.csv_path)

    # Initialize dataloader
    print("Initializing dataloader...")
    dataloader = HybridDataLoader(
        torch_dataset=embedding_dataset,
        string_dataset=text_data,
        batch_size=cfg.inference.batch_size,
    )

    # Compute scores for each batch
    reports = []
    for batch in tqdm(dataloader):
        report = score_batch(batch)
        reports.append(report)
    
    # Log Results to WandB
    final_report = pd.concat(reports, ignore_index=True)
    wandb.log({"final_report": final_report})
    mean_acc = final_report['score'].astype(float).mean()
    mean_ax_similarity = final_report['a_x_similarity'].mean()
    mean_bx_similarity = final_report['b_x_similarity'].mean()
    print(f"Mean Accuracy: {mean_acc:.4f}")
    print(f"Mean A-X Similarity: {mean_ax_similarity:.4f}")
    print(f"Mean B-X Similarity: {mean_bx_similarity:.4f}")
    wandb.summary["mean_accuracy"] = mean_acc
    wandb.summary["mean_a_x_similarity"] = mean_ax_similarity
    wandb.summary["mean_b_x_similarity"] = mean_bx_similarity
    wandb.finish()