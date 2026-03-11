from torchmetrics.functional.pairwise import pairwise_cosine_similarity
import torch
from sklearn.linear_model import LogisticRegression
import Levenshtein
import argparse
import pandas as pd
import pickle

def pairwise_levenshtein_ratio(words: list) -> torch.Tensor:
    n = len(words)
    dist_mat = torch.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist = Levenshtein.ratio(words[i], words[j])
            dist_mat[i, j] = dist
            dist_mat[j, i] = dist
    return dist_mat

def pairwise_logreg_scores(embeddings: torch.Tensor, logreg: LogisticRegression) -> torch.Tensor:
    embeddings = embeddings.cpu().numpy()
    scores = []
    for query_embed in embeddings:
        query_scores = logreg.decision_function(query_embed - embeddings)
        scores.append(query_scores)
    return torch.tensor(scores)

def main():
    args = get_args()

    if args.strategy == 'edit_distance':
        print("Computing pairwise Levenshtein distances...")
        df = pd.read_csv(args.data_path)
        dist_mat = pairwise_levenshtein_ratio(df['word'].tolist())
    elif args.strategy == 'cosine_similarity':
        print("Computing pairwise cosine similarities...")
        embeddings = torch.load(args.data_path)
        avg_embeddings = torch.stack([emb.mean(dim=0) for emb in embeddings])
        dist_mat = pairwise_cosine_similarity(avg_embeddings, avg_embeddings)
    elif args.strategy == 'logreg':
        print("Computing pairwise logistic regression scores...")
        embeddings = torch.load(args.data_path)
        avg_embeddings = torch.stack([emb.mean(dim=0) for emb in embeddings])
        logreg = pickle.load(open(args.logreg, 'rb'))
        if not isinstance(logreg, LogisticRegression):
            raise ValueError("Loaded model is not a LogisticRegression instance.")
        dist_mat = pairwise_logreg_scores(avg_embeddings, logreg)

    print(f"Saving pairwise similarity results to {args.output_path}...")
    torch.save(dist_mat, args.output_path)

def get_args():
    parser = argparse.ArgumentParser(description="Evaluate pairwise similarity between word embeddings.")
    parser.add_argument("--strategy", type=str, choices=["edit_distance", "cosine_similarity", "logreg"])
    parser.add_argument("--data_path", type=str, required=True, help="Path to the saved words or embeddings.")
    parser.add_argument("--logreg", type=str, help="Path to logistic regression model (if used)")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the pairwise similarity results.")
    return parser.parse_args()

if __name__ == "__main__":
    main()