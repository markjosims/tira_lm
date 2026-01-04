"""
WIP: This script will eventually be used to interface between the
tira-parsing HuggingFace dataset and glosses from both FST and
human annotators.

At the moment, it fetches the highest-probability gloss from the FST
for each sign in the dataset and adds it to the 'updated_gloss' field.
"""

from datasets import load_dataset
import yaml
from typing import *
from tqdm import tqdm

dataset_uri = 'tira-parsing/tira-parsing'
sentence_file_template = 'data/full_lists/{split}_sentences.yaml'

def get_fst_output_for_sentence(sentence_obj: Dict[str, Any]) -> Tuple[str, str]:
    gloss_parts = []
    words = []
    for word in sentence_obj['words']:
        # take the highest-probability parse, if present
        parse = word['parses'].get(0, None)
        if parse is None:
            words.append(word['original_str'])
            gloss_parts.append("<NOPARSE>")
            continue
        # parse is a list [form, segmented_form, gloss, inverse probability]
        word = parse[0]
        gloss = parse[2]
        words.append(word)
        gloss_parts.append(gloss)
    new_sentence = ' '.join(words)
    gloss = ' '.join(gloss_parts)
    return new_sentence, gloss

def main():
    ds = load_dataset(dataset_uri)
    for split in ['train', 'test', 'validation']:
        print(f'Processing split: {split}')
        sentence_file = sentence_file_template.format(split=split)
        print(f'Loading sentences from: {sentence_file}')
        with open(sentence_file, 'r') as f:
            sentences = yaml.safe_load(f)

        sentence2updated = {}
        sentence2gloss = {}
        for sentence_obj in tqdm(sentences, desc=f'Getting FST output for sentences...'):
            old_sentence = sentence_obj['sentence']
            sentence, gloss = get_fst_output_for_sentence(sentence_obj)
            sentence2updated[old_sentence] = sentence
            sentence2gloss[old_sentence] = gloss
        def add_fst_glosses(example):
            sentence = example['orig_text']
            example['updated_gloss'] = sentence2gloss.get(sentence, '')
            example['updated_text'] = sentence2updated.get(sentence, '')
            return example

        ds[split] = ds[split].map(add_fst_glosses)

    column_mapper = {
        'orig_text': 'orig_text',
        'translation': 'translation',
        'updated_gloss': 'FST_gloss',
        'updated_text': 'FST_text'
    }
    columns_to_keep = column_mapper.values()
    ds = ds.rename_columns(column_mapper)
    ds = ds.remove_columns([col for col in ds['train'].column_names if col not in columns_to_keep])

    ds.push_to_hub(
        'tira-parsing/fst-output',
        commit_message='Added FST glosses to dataset',
        private=True
    )

if __name__ == '__main__':
    main()