"""
ABX triplet selection functions.
"""

import logging
import os

import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
from typing import List, Dict, Tuple
from scripts.constants import (
    frame_list, edited_wordlist,
    abx_sentence_list, random_seed
)
import re
import random
import Levenshtein

log_level = os.environ.get('PYTHON_LOG_LEVEL', 'DEBUG')
logging.basicConfig(level=log_level)

seed = os.environ.get('RANDOM_SEED', random_seed)
random.seed(seed)
np.random.seed(seed)

levenshtein_cache: Dict[Tuple[str, str], int] = {}

def cached_levenshtein_distance(word1: str, word2: str) -> int:
    """
    Compute the Levenshtein distance between two words, using a cache to avoid redundant computations.
    """
    key = tuple(sorted((word1, word2)))
    if key not in levenshtein_cache:
        distance = Levenshtein.distance(word1, word2)
        levenshtein_cache[key] = distance
    return levenshtein_cache[key]

def sample_b_and_x(
        sentence_a_target: str,
        candidate_b_words: List[str],
        candidate_x_words: List[str]
) -> Tuple[str, str]:
    """
    Sample a sentence B and X such that Levenshtein(A, X) > Levenshtein(B, X).
    """
    if not candidate_b_words:
        raise ValueError(f"No candidate B words available for target '{sentence_a_target}'")
    if not candidate_x_words:
        raise ValueError(f"No candidate X words available for target '{sentence_a_target}'")

    max_attempts = 1000
    num_attempts = 0

    while True:
        # sample word X first, then compute the Levenshtein distance to filter candidate B words
        # and resample if no suitable B words are found
        num_attempts += 1
        if num_attempts > max_attempts:
            logging.debug(
                f"No valid BX words found for target '{sentence_a_target} after {max_attempts} attempts."
            )
            return None, None

        sentence_x_target = random.choice(candidate_x_words)
        ax_distance = cached_levenshtein_distance(sentence_a_target, sentence_x_target)
        bx_distances = [
            cached_levenshtein_distance(b_word, sentence_x_target)
            for b_word in candidate_b_words
        ]
        valid_b_words = [
            b_word for b_word, bx_distance in zip(candidate_b_words, bx_distances)
            if bx_distance < ax_distance
        
        ]
        if not valid_b_words:

            continue
        sentence_b_target = random.choice(valid_b_words)
        return sentence_b_target, sentence_x_target
    
def sample_abx(edited_word_df, target_ax_mask, target_b_mask, candidate_a_rows):
    # shuffle candidate A rows with fixed seed for reproducibility
    candidate_a_rows = candidate_a_rows.sample(frac=1, random_state=random_seed)
    for _, sentence_a_row in candidate_a_rows.iterrows():
        sentence_a_target = sentence_a_row['edited_word']

        candidate_x_mask = target_ax_mask
        candidate_x_mask.loc[sentence_a_row.name] = False  # exclude the selected A word from candidates for X
        candidate_x_words = edited_word_df.loc[candidate_x_mask, 'edited_word'].tolist()

        candidate_b_words = edited_word_df.loc[target_b_mask, 'edited_word'].tolist()

        sentence_b_target, sentence_x_target = sample_b_and_x(
                sentence_a_target,
                candidate_b_words,
                candidate_x_words
            )
        
        if sentence_b_target is None and sentence_x_target is None:
            logging.debug(
                f"No valid BX words found for target '{sentence_a_target}' after max attempts. "
                f"Skipping this A word and trying another one."
            )
            continue
    
        return sentence_a_target,sentence_b_target,sentence_x_target
    raise ValueError(f"No valid A words found for target '{sentence_a_row['word']}'")

def get_target_word(sentence: str) -> str:
    """
    Extract the target word from a sentence template.
    The target word is denoted by [$tgt=WORD].
    The word may contain spaces.
    """
    match = re.search(r'\[\$tgt=([^\]]+)\]', sentence)
    if match:
        return match.group(1)
    else:
        raise ValueError(f"Could not find target word in sentence template: {sentence}")


def generate_sentences_for_frame(frame_row, edited_word_df) -> List[Dict[str, str]]:
    """
    First select all edited words that match the target word for sentence A.
    Then, for each available value of $k$ (i.e. num_edits), select one edited word
    for sentence A, then sample a random edited word from sentences B and X such that
    Levenshtein(A, X) > Levenshtein(B, X).
    """
    sentence_a_target = get_target_word(frame_row['sentence_a'])
    sentence_b_target = get_target_word(frame_row['sentence_b'])
    sentence_x_target = get_target_word(frame_row['sentence_x'])

    assert sentence_a_target == sentence_x_target,\
        f"Target word for sentence A and X must be the same, but got '{sentence_a_target}' "\
        f"and '{sentence_x_target}'"
    assert sentence_b_target != sentence_a_target,\
        f"Target word for sentence B must be different from sentence A, but got "\
        f"'{sentence_b_target}' and '{sentence_a_target}'"

    target_ax_mask = edited_word_df['word'] == sentence_a_target
    target_b_mask = edited_word_df['word'] == sentence_b_target
    k_values = edited_word_df[target_ax_mask]['k'].unique()
    
    frame_sentences = []
    for k in k_values:
        k_mask = edited_word_df['k'] == k
        candidate_a_rows = edited_word_df[target_ax_mask & k_mask]
        if candidate_a_rows.empty:
            logging.warning(f"No edited words found for target '{sentence_a_target}' with k={k}")
            continue
        try:
            target_a_edited, target_b_edited, target_x_edited = sample_abx(edited_word_df, target_ax_mask, target_b_mask, candidate_a_rows)
        except ValueError as e:
            logging.warning(f"Error occurred while sampling ABX triplets for target '{sentence_a_target}' with k={k}: {e}")
            continue

        sentence_a = frame_row['sentence_a'].replace(f"[$tgt={sentence_a_target}]", target_a_edited)
        sentence_b = frame_row['sentence_b'].replace(f"[$tgt={sentence_b_target}]", target_b_edited)
        sentence_x = frame_row['sentence_x'].replace(f"[$tgt={sentence_x_target}]", target_x_edited)

        row_data = frame_row.to_dict()
        row_data.update({
            'sentence_a': sentence_a,
            'sentence_b': sentence_b,
            'sentence_x': sentence_x,
            'sentence_a_target': sentence_a_target,
            'sentence_b_target': sentence_b_target,
            'sentence_x_target': sentence_x_target,
            'k': k,
        })
        frame_sentences.append(row_data)
    return frame_sentences

def main():
    args = get_args()

    print("Loading edited word list and sentence frames...")
    edited_word_df = pd.read_csv(args.edited_wordlist, index_col='edited_word_index')
    frame_df = pd.read_csv(args.frame_list, index_col='frame_index')

    print("Populating sentence frames with edited words...")
    sentence_rows = []
    for _, frame_row in tqdm(frame_df.iterrows(), total=len(frame_df), desc="Processing frames"):
        sentence_rows.extend(generate_sentences_for_frame(frame_row, edited_word_df))

    print("Saving generated sentences...")
    sentence_df = pd.DataFrame(sentence_rows)
    sentence_df.to_csv(args.abx_sentence_list, index_label='index')

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ABX triplets for edited words")
    parser.add_argument(
        "--edited-wordlist",
        type=str,
        default=edited_wordlist,
        help="Path to edited word list"
    )
    parser.add_argument(
        "--abx-sentence-list",
        type=str,
        default=abx_sentence_list,
        help="Path to output ABX sentence list"
    )
    parser.add_argument(
        "--frame-list",
        type=str,
        default=frame_list,
        help="Path to frame list with sentence templates"
    )
    return parser.parse_args()

if __name__ == "__main__":
    main()