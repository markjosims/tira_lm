from datasets import load_dataset

dataset_uri = 'tira-parsing/tira-parsing'
sentences_list = 'data/sentences.txt'

def main():
    ds = load_dataset(dataset_uri)
    lines = []
    for split in ['train', 'test', 'validation']:
        print(f'Processing split: {split}')

        def make_sentence_list(example, i):
            sentence = example['orig_text']
            translation = example['translation']
            lines.append(','.join([sentence, translation, split, str(i)]))

            return None

        ds[split].map(make_sentence_list, with_indices=True)

    with open(sentences_list, 'w') as f:
        f.write('\n'.join(lines))