import hydra
from omegaconf import DictConfig
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from scripts.data_utils import SklearnDataset
from scripts.constants import random_seed

@hydra.main(version_base="1.3", config_path="../../conf/logreg", config_name="mbart_ft")
def main(cfg: DictConfig):
    embeddings_path = cfg.embeddings.path
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


if __name__ == "__main__":
    main()