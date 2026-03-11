from scripts.constants import tira_word_list, tira_sentence_list, word2sentence_list
import argparse
import torch
import pandas as pd
import os

def main():
    args = get_args()

    sentence_df = pd.read_csv(tira_sentence_list)
    word_df = pd.read_csv(tira_word_list, index_col='word_index')
    word2sentence = pd.read_csv(word2sentence_list)

    word = args.word
    try:
        word_index = int(word)
        word = word_df.at[word_index, 'word']
    except ValueError:
        word_index = word_df[word_df['word']==word].iloc[0].name

    sentence_indices = word2sentence[word2sentence['word_index']==word_index]['sentence_index'].tolist()
    hits = sentence_df.iloc[sentence_indices]
    hits['word'] = word
    hits['rank'] = 'self'

    dist_mat = torch.load(args.distance_matrix)
    row = dist_mat[word_index]
    row[word_index] = -float('inf')
    top_indices = torch.topk(row, k=10).indices.tolist()
    top_words = word_df.iloc[top_indices]
    top_words = top_words.reset_index()
    
    top_sentences = []
    for i, row in top_words.iterrows():
        word_index = row['word_index']
        word = row['word']
        sentence_indices = word2sentence[
            word2sentence['word_index']==word_index
        ]['sentence_index'].tolist()
        sentences = sentence_df.iloc[sentence_indices]
        sentences['rank'] = str(i)
        top_sentences.append(sentences)
    hits = pd.concat([hits]+top_sentences)
    print(hits)
    distance_matrix_stem = os.path.splitext(os.path.basename(args.distance_matrix))[0]
    hits.to_csv(f'{word}_{distance_matrix_stem}_hits.csv', index=False)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--word', '-w')
    parser.add_argument('--distance_matrix', '-d')
    return parser.parse_args()

if __name__ == '__main__':
    main()