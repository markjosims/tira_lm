import os
import torch

# filepaths
seed_words = os.path.abspath('doc/abx_word_seeds.csv')
edited_wordlist = os.path.abspath('data/edited_abx_words.csv')

documentation_dir = os.path.abspath('doc/')
frame_config = os.path.abspath('doc/frames.yaml')
frame_list = os.path.abspath('data/abx_frames.csv')

abx_sentence_list = os.path.abspath('data/abx_sentences.csv')

parenthetical_list = os.path.abspath('doc/parentheticals.tsv')

# misc
random_seed = os.environ.get('RANDOM_SEED', 42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')