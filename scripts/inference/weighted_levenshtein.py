from scripts.data_preparation.abx_words import vowels, tone, dental_bridge
from numba import jit
import numpy as np
from typing import List, Tuple
import pandas as pd
from tqdm import tqdm

tqdm.pandas()

"""
Functions for weighted Levenshtein edit distance, where edit
weights are expert-coded.
"""

abx_wordlist = 'data/abx_words.csv'

def get_edit_costs() -> np.array:

    reduced_edit_cost = 0.5
    minor_edit_cost = 0.05
    insertion_costs = [
        ('ə', minor_edit_cost),
        (vowels, reduced_edit_cost),
        (tone, minor_edit_cost),
        (dental_bridge, minor_edit_cost),
    ]
    deletion_costs = [
        ('ə', minor_edit_cost),
        (vowels, reduced_edit_cost),
        (tone, minor_edit_cost),
    ]
    substitution_costs = [
        ('ə', vowels, minor_edit_cost),  # underlying vowel reduced to schwa
        ('ɛ', 'ə', minor_edit_cost),    # underlying schwa fronted to /ɛ/

        ('ɜ', 'ɛ', minor_edit_cost),    # ɜ~ɛ interchange
        ('ɛ', 'ɜ', minor_edit_cost),

        ('ɜ', 'a', minor_edit_cost),    # ɜ~a interchange
        ('a', 'ɜ', minor_edit_cost),

        ('ɜ', 'ə', minor_edit_cost),    # ɜ~ə interchange
        ('ə', 'ɜ', minor_edit_cost),

        ('ɛ', 'e', minor_edit_cost),    # ɛ~e interchange
        ('e', 'ɛ', minor_edit_cost),

        ('ɪ', 'e', minor_edit_cost),    # ɪ~ɛ interchange
        ('e', 'ɪ', minor_edit_cost),

        ('o', 'u', minor_edit_cost),    # o~u interchange
        ('u', 'o', minor_edit_cost),

        ('ɔ', 'o', minor_edit_cost),    # o~ɔ interchange
        ('o', 'ɔ', minor_edit_cost),

        ('ɔ', 'u', minor_edit_cost),    # o~u interchange
        ('u', 'ɔ', minor_edit_cost),

        ('ʊ', 'o', minor_edit_cost),    # o~ʊ interchange
        ('o', 'ʊ', minor_edit_cost),

        ('ʊ', 'u', minor_edit_cost),    # u~ʊ interchange
        ('u', 'ʊ', minor_edit_cost),

        ('g', 'k', minor_edit_cost),    # g~k interchange
        ('k', 'g', minor_edit_cost),

        ('r', 'ɾ', minor_edit_cost),    # tap written as trill

        ('u', 'w', minor_edit_cost),    # glide~vowel interchange
        ('w', 'u', minor_edit_cost),
        ('o', 'w', minor_edit_cost),
        ('w', 'o', minor_edit_cost),
        ('i', 'j', minor_edit_cost),
        ('j', 'i', minor_edit_cost),

        (tone, tone, minor_edit_cost),
    ]

    substitution_costs_expanded = []
    insertion_costs_expanded = []
    deletion_costs_expanded = []


    for intab, outtab, cost in substitution_costs:
        for in_char in intab:
            for out_char in outtab:
                substitution_costs_expanded.append((in_char, out_char, cost))

    for intab, cost in insertion_costs:
        for in_char in intab:
            insertion_costs_expanded.append((in_char, cost))

    for intab, cost in deletion_costs:
        for in_char in intab:
            deletion_costs_expanded.append((in_char, cost))

    return substitution_costs_expanded, insertion_costs_expanded, deletion_costs_expanded

@jit(nopython=True, cache=True)
def weighted_levenshtein(
    string_a: str,
    string_b: str,
    substitution_costs: List[Tuple[str, str, float]],
    insertion_costs: List[Tuple[str, float]],
    deletion_costs: List[Tuple[str, float]],

) -> float:
    """
    Computes Levenshtein edit distance between two strings using custom costs,
    as defined by substitution_costs, insertion_costs and deletion_costs.

    Arguments:
        string_a: The first string.
        string_b: The second string.
        substitution_costs: A list of tuples (char1, char2, cost) defining the cost of substituting char1 with char2.
        insertion_costs: A list of tuples (char, cost) defining the cost of inserting a character.
        deletion_costs: A list of tuples (char, cost) defining the cost of deleting a character.
            the actual lengths of the reference sequences.

    Returns:
        edit_distance: The weighted Levenshtein distance between string_a and string_b.
    """
    def get_insertion_cost(char, insertion_costs):
        for in_char, cost in insertion_costs:
            if char == in_char:
                return cost
        return 1.0

    def get_deletion_cost(char, deletion_costs):
        for in_char, cost in deletion_costs:
            if char == in_char:
                return cost
        return 1.0

    def get_substitution_cost(char_a, char_b, substitution_costs):
        for char1, char2, cost in substitution_costs:
            if char_a == char1 and char_b == char2:
                return cost
        return 1.0



    levenshtein_matrix = np.zeros((len(string_a) + 1, len(string_b) + 1), dtype=np.float64)
    
    # Initialize the first row and column of the matrix
    for i in range(len(string_a) + 1):
        levenshtein_matrix[i, 0] = get_deletion_cost(string_a[i - 1], deletion_costs) if i > 0 else 0

    for j in range(len(string_b) + 1):
        levenshtein_matrix[0, j] = get_insertion_cost(string_b[j - 1], insertion_costs) if j > 0 else 0

    for i in range(1, len(string_a) + 1):
        for j in range(1, len(string_b) + 1):
            # Compute costs for each operation
            insert_cost = levenshtein_matrix[i - 1, j]
            delete_cost = levenshtein_matrix[i, j - 1]
            match_cost = levenshtein_matrix[i - 1, j - 1]

            insert_cost += get_insertion_cost(string_b[j - 1], insertion_costs)
            delete_cost += get_deletion_cost(string_a[i - 1], deletion_costs)
            if string_a[i - 1] != string_b[j - 1]:
                match_cost += get_substitution_cost(string_a[i - 1], string_b[j - 1], substitution_costs)

            # Update the matrix with the minimum cost of all operations
            levenshtein_matrix[i, j] = min(match_cost, insert_cost, delete_cost)

    return levenshtein_matrix[len(string_a), len(string_b)]


@jit(nopython=True, cache=True)
def levenshtein(string_a: str, string_b: str) -> int:
    """
    TODO: test against Levenshtein library for speed and correctness
    Computes the standard Levenshtein edit distance between two strings.

    Arguments:
        string_a: The first string.
        string_b: The second string.

    Returns:
        edit_distance: The Levenshtein distance between string_a and string_b.
    """
    if len(string_a) < len(string_b):
        return levenshtein(string_b, string_a)

    if len(string_b) == 0:
        return len(string_a)

    previous_row = range(len(string_b) + 1)
    for i, c1 in enumerate(string_a):
        current_row = [i + 1]
        for j, c2 in enumerate(string_b):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

def main():
    abx_df = pd.read_csv(abx_wordlist)

    # for now, randomly sample 1k words
    num_sample=1_000
    abx_df = abx_df.sample(num_sample)

    substitution_costs, insertion_costs, deletion_costs = get_edit_costs()

    a_x_dist_pred = abx_df.progress_apply(
        lambda row: weighted_levenshtein(
            row['word_a'],
            row['word_x'],
            substitution_costs,
            insertion_costs,
            deletion_costs,
        ),
        axis=1,
    )
    b_x_dist_pred = abx_df.progress_apply(
        lambda row: weighted_levenshtein(
            row['word_b'],
            row['word_x'],
            substitution_costs,
            insertion_costs,
            deletion_costs,
        ),
        axis=1,
    )
    a_x_closer = a_x_dist_pred < b_x_dist_pred
    hits = a_x_closer.sum()
    total = len(abx_df)
    accuracy = hits / total
    print(f'ABX accuracy: {accuracy:.4f} ({hits}/{total})')

if __name__ == '__main__':
    main()
