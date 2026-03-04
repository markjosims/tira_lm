from datasets import load_dataset
import os
import pandas as pd

dataset_uri = 'tira-parsing/tira-parsing'
sentences_list = 'data/sentences.csv'

def main():
    print(f'Loading dataset from: {dataset_uri}')
    ds = load_dataset(dataset_uri)
    rows = []
    for split in ['train', 'test', 'validation']:
        print(f'Processing split: {split}')

        def make_sentence_list(example, i):
            sentence = example['orig_text']
            translation = example['translation']
            rows.append({
                'sentence': sentence,
                'translation': translation,
                'split': split,
                'index': str(i)
            })

            return None

        ds[split].map(make_sentence_list, with_indices=True)

    print(f'Writing sentences to: {sentences_list}')
    os.makedirs(
        os.path.dirname(sentences_list),
        exist_ok=True,
    )
    df = pd.DataFrame(rows)
    df=df.set_index('index')
    df.to_csv(sentences_list, index=True)

if __name__ == '__main__':
    main()