from datasets import load_dataset
import os
import pandas as pd
import re
import argparse

dataset_uri = 'tira-parsing/tira-parsing'
sentences_list = 'data/sentences.csv'

def preprocess_transcription(sentence: str, _):
    """
    Remove leading and trailing whitespace, commas and newlines.
    Currently does not use any arguments, but we include the args
    parameter to allow for future preprocessing options that may
    depend on command-line arguments.
    """
    sentence = sentence.strip()
    sentence = sentence.replace('\n', ' ')
    sentence = sentence.replace(',', '')
    
    return sentence

def preprocess_translation(sentence: str, args: argparse.Namespace):
    """
    Remove all non-alphanumeric characters, set to lowercase,
    and optionally remove parenthetical comments, e.g. "(away from)".
    """

    sentence = preprocess_transcription(sentence, args)

    if not args.include_parentheticals:
        # Remove parenthetical comments, e.g. "(away from)"
        sentence = re.sub(r'\s*\(.*?\)\s*', ' ', sentence)

    sentence = re.sub(r'[^a-zA-Z0-9\s]', '', sentence)
    sentence = sentence.lower()

    return sentence

def filter_sentences(sentence: str):
    """
    Filter out sentences which are fragments of other sentences.
    These sentences are marked with "Parts of:" in the translation column,
    and should be ignored.
    """
    part_of_pattern = re.compile(r'^parts? of', re.IGNORECASE)
    return not part_of_pattern.match(sentence)

def main():
    args = get_args()
    print(f'Loading dataset from: {args.dataset_uri}')
    ds = load_dataset(args.dataset_uri)
    rows = []
    for split in ['train', 'test', 'validation']:
        print(f'Processing split: {split}')

        def make_sentence_list(example, i):
            sentence = preprocess_transcription(example['orig_text'], args)
            translation = preprocess_translation(example['translation'], args)
            rows.append({
                'sentence': sentence,
                'translation': translation,
                'split': split,
                'index': str(i)
            })

            return None

        ds[split].map(make_sentence_list, with_indices=True)

    print(f'Writing sentences to: {args.sentences_list}')
    os.makedirs(
        os.path.dirname(args.sentences_list),
        exist_ok=True,
    )
    df = pd.DataFrame(rows)
    df=df.set_index('index')
    df.to_csv(args.sentences_list, index=True)

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Make sentence list from Hugging Face dataset')
    parser.add_argument('--dataset_uri', type=str, default=dataset_uri, help='Hugging Face dataset URI')
    parser.add_argument('--sentences_list', type=str, default=sentences_list, help='Path to output sentences list CSV file')
    parser.add_argument(
        '--include_parentheticals',
        action='store_true',
        help='Whether to include parenthetical comments in translations, e.g. "(away from)"'
    )
    return parser.parse_args() 

if __name__ == '__main__':
    main()