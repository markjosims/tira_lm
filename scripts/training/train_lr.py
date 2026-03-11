import os

import hydra
from omegaconf import DictConfig
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from scripts.data_utils import SklearnDataset
from scripts.constants import random_seed
import wandb
import pickle


@hydra.main(version_base="1.3", config_path="../../conf/logreg", config_name="mbart_ft")
def main(cfg: DictConfig):
    os.environ["WANDB_PROJECT"] = cfg.wandb.project
    wandb.init()

    embeddings_path = cfg.data.embed_path
    print(f"Loading embeddings from: {embeddings_path}")
    dataset = SklearnDataset(embeddings_path)
    X, y = dataset.X, dataset.y

    # Initialize K-fold and train logistic regression model
    print(
        f"Training Logistic Regression model with K={cfg.training.kfold.k}"\
        " fold cross-validation..."
    )
    kf = KFold(n_splits=cfg.training.kfold.k, shuffle=True, random_state=random_seed)
    fold_accuracies = []
    for fold, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model = LogisticRegression(max_iter=cfg.training.max_iter, random_state=random_seed)
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        fold_accuracies.append(accuracy)
        print(f"Fold {fold + 1} Accuracy: {accuracy:.4f}")
        wandb.log({f"accuracy": accuracy})

    avg_accuracy = sum(fold_accuracies) / len(fold_accuracies)
    wandb.summary["average_accuracy"] = avg_accuracy
    print(f"Average Accuracy across {cfg.training.kfold.k} folds: {avg_accuracy:.4f}")

    save_dir = cfg.outputs.save_dir
    os.makedirs(save_dir, exist_ok=True)
    model_save_path = os.path.join(save_dir, "logistic_regression_model.pkl")
    with open(model_save_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to: {model_save_path}")
    

if __name__ == "__main__":
    main()