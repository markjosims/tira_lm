from typing import List, Tuple
import re
from numba import jit
import numpy as np
import random
import os

seed = os.environ.get('RANDOM_SEED', 1337)
random.seed(seed)

"""
phoneme constants
"""

front_vowels = 'iɪeɛ'
central_vowels = 'əɜa'
back_vowels = 'uʊoᴐ'

vowels = front_vowels + central_vowels + back_vowels

stops = 'ptkbdg'
fricatives = 'svð'
affricates = 'cɟ'
nasals = 'mnɲŋ'
sonorants = 'rɾɽljw'
consonants = stops + fricatives + affricates + nasals + sonorants

high_tone = '\u0301'
low_tone = '\u0300'
fall_tone = '\u0302'
rise_tone = '\u030c'
fallrise_tone = '\u1dc9'

tone = high_tone + low_tone + fall_tone + rise_tone + fallrise_tone

dental_bridge = '\u032a'

"""
Functions for weighted Levenshtein edit distance, where edit
weights are expert-coded.
"""

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

"""
Constants describing common edits in transcriptions
"""

# frequent bidirectional edits
# e.g. both e > ɛ and ɛ > e are observed

interchange_sets = [
    front_vowels,
    back_vowels,
]

# unidirectional edits
# e.g. a > ə but not ə > a

unidirectional_edits = {
    'ɜ': 'aəɛ',
    dental_bridge: '',
    'ð': 'd',
    'ɾɽ': 'r',
}

"""
Helper functions for tokenization
"""

# multichar tokens
# all other single characters are assigned to a token
# note: t̪ and d̪ are not tone-bearing, so no character
# will have both a tone and a bridge

consonant_regex = rf'[{consonants}]{dental_bridge}?'
segment_w_tone = rf'[{consonants+vowels}][{tone}]'


def tokenize_string(untokenized_str: str) -> Tuple[str]:
    """
    Tokenizes a string into a tuple where each member is a single segment
    or segment + tone. Returns tuple rather than list as safeguard against
    side-effects.

    Arguments:
        untokenized_str:    Tira word to be tokenized
    Returns:
        tokens:             Tuple of segments w/ tone (if applicable)

    """
    i = 0
    tokens = []
    while i < len(untokenized_str):
        suffix = untokenized_str[i:]
        for multichar_regex in [consonant_regex, segment_w_tone]:
            match = re.match(multichar_regex, suffix)
            if match is not None:
                i+=match.end
                tokens+=match.group()
                break
        else:
            i+=1
            tokens+=untokenized_str[0]
    return tuple(tokens)

"""
Helper functions for matching tokens based on some condition
"""

def find_tokens_w_char(tokens: Tuple[str, ...], char: str) -> List[int]:
    """
    Returns list of indices corresponding to tokens containing
    the specified character.

    Arguments:
        tokens:     tuple of token strings
        char:       character to search for
    Returns:
        indices:    list of token indices
    """

    token_indices = [i for i, token in enumerate(tokens) if char in token]
    return token_indices

def token_is_vowel(token: str) -> bool:
    return any(v in token for v in vowels)

def token_is_consonant(token: str) -> bool:
    return any(c in token for c in consonants)

def find_interconsonantal_vowels(tokens: Tuple[str, ...]) -> List[int]:
    """
    Returns list of indices for any token corresponding to a vowel between
    two consonants.

    Arguments:
        tokens:     tuple of token strings
    Returns:
        indices:    list of token indices  
    """
    if len(tokens) < 3:
        # must be minimally CVC
        return tokens
    
    token_indices = []

    for i, curr_token in enumerate(
        tokens[1:-1],
        stat=1
    ):
        prev_token = tokens[i-1]
        next_token = tokens[i+1]
        if (
            token_is_consonant(prev_token) and
            token_is_vowel(curr_token) and
            token_is_consonant(next_token)
        ):
            token_indices.append(i)

    return token_indices

def find_first_vowel_in_hiatus(tokens: Tuple[str, ...]) -> List[int]:
    """
    Returns list of indices for any token corresponding to a vowel before
    another vowel.

    Arguments:
        tokens:     tuple of token strings
    Returns:
        indices:    list of token indices  
    """
    if len(tokens) < 2:
        # must be minimally VV
        return tokens
    
    token_indices = []

    for i, curr_token in enumerate(tokens[:-1]):
        next_token = tokens[i+1]
        if (
            token_is_vowel(curr_token) and
            token_is_vowel(next_token)
        ):
            token_indices.append(i)

    return token_indices

def tokens_have_same_segment(token_1: str, token_2: str) -> bool:

    # order shouldn't matter as every token will be a single segment
    # or a geminate segment (e.g. ɛ̀ɛ̌)
    vowels_1 = set(c for c in token_1 if c in vowels+consonants)
    vowels_2 = set(c for c in token_2 if c in vowels+consonants)
    return vowels_1 == vowels_2

def tokens_have_same_tone(token_1: str, token_2: str) -> bool:

    # order shouldn't matter: even if both characters of a geminate
    # have different tone (e.g. ɛ̀ɛ̌), no rule will cause *metathesis*
    # of tones
    tones_1 = set(c for c in token_1 if c in tone)
    tones_2 = set(c for c in token_2 if c in tone)
    return tones_1 == tones_2

"""
Functions for performing edits to simulate transcription noise.
"""

def swap_char(intab: str, outtab: str, tokens: Tuple[str, ...]) -> Tuple[str, ...]:
    token_indices = find_tokens_w_char(intab)
    if not token_indices:
        return tokens
    
    sampled_index = random.choice(token_indices)

    if intab in tone:
        # count = 0 so we don't change the tone twice on a geminate
        # e.g. ìǐ changes to íǐ not *íí
        new_token = tokens[sampled_index].replace(intab, outtab, 0)
    else:
        # if swap doesn't involve tone, change both
        # e.g. ɛ̀ɛ̌ > èě
        new_token = tokens[sampled_index].replace(intab, outtab)
    
    return update_token(tokens, sampled_index, new_token)

def centralize_interconsontal_vowel(tokens: Tuple[str, ...]) -> Tuple[str, ...]:
    token_indices = find_interconsonantal_vowels(tokens)
    if not token_indices:
        return tokens

    sampled_index = random.choice(token_indices)
    new_token = tokens[sampled_index]

    for v in vowels:
        # token should only have one vowel
        # break once we find it
        if v in new_token:
            new_token = new_token.replace(v, 'ə')
            break
    
    return update_token(tokens, sampled_index, new_token)

def delete_first_vowel_in_hiatus(tokens: Tuple[str, ...]) -> Tuple[str, ...]:
    token_indices = find_first_vowel_in_hiatus(tokens)
    if not token_indices:
        return tokens
    
    sampled_index = random.choice(token_indices)
    # set to empty str
    return update_token(tokens, sampled_index, '')

def update_token(tokens, sampled_index, new_token):
    preceding_tokens = list(tokens[:sampled_index]) if sampled_index > 0 else []
    following_tokens = list(tokens[sampled_index+1:]) if sampled_index < len(tokens) else []
    return tuple(preceding_tokens + [new_token] + following_tokens)

"""
Create list of curried functions for each possible type of edit.
"""

edit_list = [
    centralize_interconsontal_vowel,
    delete_first_vowel_in_hiatus,
]

# TODO add all non-vacuous changes to list
for charset in interchange_sets:
    ...

for intab, outtab in unidirectional_edits.items():
    ...