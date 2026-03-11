import os
import torch

project_root = os.path.dirname(os.path.dirname(__file__))

# filepaths
seed_words = os.path.join(project_root, 'doc/abx_word_seeds.csv')
edited_wordlist = os.path.join(project_root, 'data/edited_abx_words.csv')

documentation_dir = os.path.join(project_root, 'doc/')
frame_config = os.path.join(project_root, 'doc/frames.yaml')
frame_list = os.path.join(project_root, 'data/abx_frames.csv')

dataset_uri = 'tira-parsing/tira-parsing'
abx_sentence_list = os.path.join(project_root, 'data/abx_sentences.csv')
tira_sentence_list = os.path.join(project_root, 'data/tira_sentences.csv')
tira_word_list = os.path.join(project_root, 'data/tira_words.csv')
word2sentence_list = os.path.join(project_root, 'data/word2sentence.csv')
parenthetical_list = os.path.join(project_root, 'doc/parentheticals.tsv')

# misc
random_seed = os.environ.get('RANDOM_SEED', 42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')