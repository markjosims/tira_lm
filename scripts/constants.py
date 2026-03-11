import os
import torch

# filepaths
seed_words = os.path.abspath('doc/abx_word_seeds.csv')
edited_wordlist = os.path.abspath('data/edited_abx_words.csv')

documentation_dir = os.path.abspath('doc/')
frame_config = os.path.abspath('doc/frames.yaml')
frame_list = os.path.abspath('data/abx_frames.csv')

dataset_uri = 'tira-parsing/tira-parsing'
abx_sentence_list = os.path.abspath('data/abx_sentences.csv')
tira_sentence_list = os.path.abspath('data/tira_sentences.csv')
tira_word_list = os.path.abspath('data/tira_words.csv')
word2sentence_list = os.path.abspath('data/word2sentence.csv')
parenthetical_list = os.path.abspath('doc/parentheticals.tsv')

# misc
random_seed = os.environ.get('RANDOM_SEED', 42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')