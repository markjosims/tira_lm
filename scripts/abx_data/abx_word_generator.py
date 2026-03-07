import itertools
from typing import Dict, List, Literal, Tuple, Callable
import re
from numba import jit
import numpy as np
import random
import os
from dataclasses import dataclass, field
from collections import defaultdict
import logging
from functools import partial
import pandas as pd
from argparse import ArgumentParser
from Levenshtein import distance as levenshtein_distance
from tqdm import tqdm

wordlist = 'doc/keywords.csv'
edited_wordlist = 'data/edited_keywords.csv'
abx_wordlist = 'data/abx_words.csv'

log_level = os.environ.get('PYTHON_LOG_LEVEL', 'DEBUG')
logging.basicConfig(level=log_level)

seed = os.environ.get('RANDOM_SEED', 42)
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
Constants describing common edits in transcriptions
"""

# frequent bidirectional edits
# e.g. both e > ɛ and ɛ > e are observed

interchange_sets = [
    'oᴐ',
    'uʊo',
    'iɪe',
    'ɛe',
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

@dataclass
class Edit:
    edit_type: Literal['vowel_change', 'consonant_change', 'tone_change', 'epenthesis', 'deletion']
    token_index: int
    original_token: str
    new_token: str

    def __str__(self):
        return f'{self.edit_type} at index {self.token_index} ({self.original_token} > {self.new_token})'

class EditHistory:
    def __init__(self):
        self.history = defaultdict(tuple)

    def add_edit(self, token_index: int, edit: 'Edit'):
        self.history[token_index] += (edit,)

    def copy(self) -> 'EditHistory':
        new_history = EditHistory()
        for token_index, edits in self.history.items():
            new_history.history[token_index] = edits
        return new_history
    
    def __getitem__(self, key):
        return self.history[key]

    def __str__(self):
        # str representation of Edit object already includes index
        return ';'.join(str(edit) for edits in self.history.values() for edit in edits)

@dataclass
class EditFunction:
    name: str
    edit_function: Callable[['TokenArray'], 'TokenArray']

    def __call__(self, tokens: 'TokenArray') -> 'TokenArray':
        return self.edit_function(tokens)

@dataclass
class TokenArray:
    data: Tuple[str, ...]
    token_history: 'EditHistory' = field(default_factory=EditHistory)

    def __getitem__(self, index):
        return self.data[index]
    
    def __str__(self):
        return ''.join(self.data)
    
    def __len__(self):
        return len(self.data)

    def replace_token(self, index: int, new_token: str, edit: 'Edit') -> 'TokenArray':
        """
        Construct new TokenArray with token at index replaced by new_token,
        and edit added to token_history.
        """
        preceding_tokens = list(self.data[:index]) if index > 0 else []
        following_tokens = list(self.data[index+1:]) if index < len(self.data) else []
        data = tuple(preceding_tokens + [new_token] + following_tokens)
        token_history = self.token_history.copy()
        token_history.add_edit(index, edit)
        return TokenArray(data=data, token_history=token_history)

    @classmethod
    def tokenize_string(cls, untokenized_str: str) -> 'TokenArray':
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
                    i+=match.end()
                    tokens.append(match.group())
                    break
            else:
                i+=1
                tokens.append(suffix[0])
        return cls(data=tuple(tokens))

"""
Helper functions for matching tokens based on some condition
"""

def find_tokens_w_char(tokens: TokenArray, char: str) -> List[int]:
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

def find_tokens_without_edit(tokens: TokenArray, edit_type: str) -> List[int]:
    """
    Returns list of indices corresponding to tokens that have not been edited
    by the specified edit type.

    Arguments:
        tokens:     tuple of token strings
        edit_type:  type of edit to check for (e.g. 'vowel_change')
    Returns:
        indices:    list of token indices
    """

    token_indices = [
        i for i, token in enumerate(tokens)
        if all(
            edit.edit_type != edit_type for edit in tokens.token_history[i]
        )
    ]
    return token_indices

def get_index_intersection(*index_lists: List[List[int]]) -> List[int]:
    """
    Returns list of indices corresponding to tokens that satisfy all conditions
    specified by index_lists.

    Arguments:
        index_lists: list of lists of token indices, where each list corresponds
                     to a condition (e.g. contains char, has not been edited by
                     a certain edit type)
    Returns:
        indices:    list of token indices
    """
    if not index_lists:
        return []
    return list(set.intersection(*[set(lst) for lst in index_lists]))

def token_is_vowel(token: str) -> bool:
    return any(v in token for v in vowels)

def token_is_consonant(token: str) -> bool:
    return any(c in token for c in consonants)

def find_interconsonantal_vowels(tokens: TokenArray) -> List[int]:
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
        start=1
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

def find_first_vowel_in_hiatus(tokens: TokenArray) -> List[int]:
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

def swap_char(intab: str, outtab: str, tokens: TokenArray) -> TokenArray:
    logging.debug(f"Function swap_char called with intab={intab} and outtab={outtab}")
    token_indices = find_tokens_w_char(tokens, intab)
    if intab in tone:
        edit_type = 'tone_change'
    elif intab in consonants:
        edit_type = 'consonant_change'
    else:
        edit_type = 'vowel_change'
    valid_tokens = find_tokens_without_edit(tokens, edit_type)
    token_indices = get_index_intersection(token_indices, valid_tokens)

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
    

    edit = Edit(
        edit_type=edit_type,
        token_index=sampled_index,
        original_token=tokens[sampled_index],
        new_token=new_token,
    )
    return tokens.replace_token(sampled_index, new_token, edit)

def centralize_interconsontal_vowel(tokens: TokenArray) -> TokenArray:
    token_indices = find_interconsonantal_vowels(tokens)
    valid_tokens = find_tokens_without_edit(tokens, 'vowel_change')
    token_indices = get_index_intersection(token_indices, valid_tokens)

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
    
    edit = Edit(
        edit_type='vowel_change',
        token_index=sampled_index,
        original_token=tokens[sampled_index],
        new_token=new_token,
    )
    return tokens.replace_token(sampled_index, new_token, edit)

def delete_first_vowel_in_hiatus(tokens: TokenArray) -> TokenArray:
    token_indices = find_first_vowel_in_hiatus(tokens)
    valid_tokens = find_tokens_without_edit(tokens, 'vowel_change')
    token_indices = get_index_intersection(token_indices, valid_tokens)

    if not token_indices:
        return tokens
    
    sampled_index = random.choice(token_indices)
    # set token to empty str
    edit = Edit(
        edit_type='vowel_change',
        token_index=sampled_index,
        original_token=tokens[sampled_index],
        new_token='',
    )
    return tokens.replace_token(sampled_index, '', edit)

def delete_interconsonantal_schwa(tokens: TokenArray) -> TokenArray:
    """
    Deletes schwa in interconsonantal position. Only change if the
    vowel is originally a schwa, not if it was changed to schwa by
    another rule.
    """
    token_indices = find_interconsonantal_vowels(tokens)
    schwa_indices = find_tokens_w_char(tokens, 'ə')
    unchanged_vowel_indices = find_tokens_without_edit(tokens, 'vowel_change')
    token_indices = get_index_intersection(
        token_indices,
        schwa_indices,
        unchanged_vowel_indices,
    )
    if not token_indices:
        return tokens
    
    sampled_index = random.choice(token_indices)
    # set token to empty str
    edit = Edit(
        edit_type='vowel_change',
        token_index=sampled_index,
        original_token=tokens[sampled_index],
        new_token='',
    )
    return tokens.replace_token(sampled_index, '', edit)

def add_intonational_rise(tokens: TokenArray) -> TokenArray:
    """
    Intonational rise is often transcribed as a rising tone on the vowel before a pause,
    where the vowel is written as a geminate.
    """
    token_indices = [i for i, token in enumerate(tokens) if token_is_vowel(token)]
    # intonational rise can only occur on last syllable
    last_vowel_index = token_indices[-1]
    tokens_without_epenthesis = find_tokens_without_edit(tokens, edit_type='epenthesis')

    if last_vowel_index not in tokens_without_epenthesis:
        return tokens

    vowel = tokens[last_vowel_index][0]
    assert vowel in vowels, f"Expected vowel token, got {tokens[last_vowel_index]}"
    new_token = tokens[last_vowel_index] + vowel + rise_tone

    edit = Edit(
        edit_type='epenthesis',
        token_index=last_vowel_index,
        original_token=tokens[last_vowel_index],
        new_token=new_token,
    )
    return tokens.replace_token(last_vowel_index, new_token, edit)

def first_high_to_rise(tokens: TokenArray) -> TokenArray:
    """
    The first high tone in a sequence of high tones is sometimes
    transcribed as a rising tone.
    """
    high_tone_indices = find_tokens_w_char(tokens, high_tone)
    # only include tokens where the following vowel token is also high
    # (which would be two tokens later, i.e. V́, C, V́)
    high_tone_indices = [i for i in high_tone_indices if i + 2 in high_tone_indices]

    valid_tokens = find_tokens_without_edit(tokens, edit_type='tone_change')
    token_indices = get_index_intersection(high_tone_indices, valid_tokens)

    if not token_indices:
        return tokens
    
    sampled_index = random.choice(token_indices)
    new_token = tokens[sampled_index].replace(high_tone, rise_tone, 1)
    edit = Edit(
        edit_type='tone_change',
        token_index=sampled_index,
        original_token=tokens[sampled_index],
        new_token=new_token,
    )
    return tokens.replace_token(sampled_index, new_token, edit)

def high_before_low_to_fall(tokens: TokenArray) -> TokenArray:
    """
    A high tone before a low tone is sometimes transcribed as a falling tone.
    """
    high_tone_indices = find_tokens_w_char(tokens, high_tone)
    low_tone_indices = find_tokens_w_char(tokens, low_tone)
    # only include tokens where the following vowel token has a low tone
    # (which would be two tokens later, i.e. V́, C, V̀)
    high_before_low_indices = [i for i in high_tone_indices if i + 2 in low_tone_indices]

    valid_tokens = find_tokens_without_edit(tokens, edit_type='tone_change')
    token_indices = get_index_intersection(high_before_low_indices, valid_tokens)

    if not token_indices:
        return tokens
    
    sampled_index = random.choice(token_indices)
    new_token = tokens[sampled_index].replace(high_tone, fall_tone, 1)
    edit = Edit(
        edit_type='tone_change',
        token_index=sampled_index,
        original_token=tokens[sampled_index],
        new_token=new_token,
    )
    return tokens.replace_token(sampled_index, new_token, edit)

def delete_space(tokens: TokenArray) -> TokenArray:
    """
    Deletes space between a verbal aux and stem, simulating a common transcription error.
    """
    token_indices = find_tokens_w_char(tokens, ' ')

    if not token_indices:
        return tokens
    
    sampled_index = random.choice(token_indices)
    # set token to empty str
    edit = Edit(
        edit_type='deletion',
        token_index=sampled_index,
        original_token=tokens[sampled_index],
        new_token='',
    )
    return tokens.replace_token(sampled_index, '', edit)

"""
Create list of curried functions for each possible type of edit.
"""

emptyset = '\u2205' # ∅ used to represent empty string in edit descriptions

edit_list = [
    EditFunction(
        'V > ə / C_C',
        centralize_interconsontal_vowel
    ),
    EditFunction(
        'V1V2 > V2',
        delete_first_vowel_in_hiatus
    ),
    EditFunction(
        f'ə > {emptyset} / C_C',
        delete_interconsonantal_schwa
    ),
    EditFunction(
        'V > VV̌',
        add_intonational_rise
    ),
    EditFunction(
        'H > R / _ H',
        first_high_to_rise
    ),
    EditFunction(
        'H > F / _ L',
        high_before_low_to_fall
    ),
    EditFunction(
        'delete space',
        delete_space
    )
]

for charset in interchange_sets:
    for char in charset:
        for other_char in charset:
            if char != other_char:
                edit_callable = partial(swap_char, char, other_char)
                edit_name = f'{char} > {other_char}'
                if other_char == '':
                    edit_name = f'{char} > {emptyset}'
                edit_list.append(EditFunction(edit_name, edit_callable))

for intab_set, outtab_set in unidirectional_edits.items():
    # don't strictly need to iterate through intabs since every intab
    # is a single charm, but this is robust to the possibility of multichar
    # intabs in the future
    for intab in intab_set:
        for outtab in outtab_set:
            edit_callable = partial(swap_char, intab, outtab)
            edit_name = f'{intab} > {outtab}'
            if outtab == '':
                edit_name = f'{intab} > {emptyset}'
            edit_list.append(EditFunction(edit_name, edit_callable))

def apply_k_edits(tira_word: str, k: int) -> str:
    """
    Randomly select $k$ edit functions from `edit_list`
    and apply to the string `tira_word`. Only apply
    non-vacuous edits: If a function returns the same word,
    pick a different function.
    """
    word_tokens = TokenArray.tokenize_string(tira_word)
    prev_word_tokens = word_tokens
    for i in range(k):
        remaining_edit_functs = edit_list[:]

        logging.debug(f"Applying {i}^th edit out of k={k}...")
        no_edit_made = True
        while no_edit_made and remaining_edit_functs:
            edit_funct = random.choice(remaining_edit_functs)
            logging.debug(f"Applying edit {edit_funct.name} to string {prev_word_tokens}")
            word_tokens = edit_funct(prev_word_tokens)

            no_edit_made = str(word_tokens) == str(prev_word_tokens)
            if no_edit_made:
                logging.debug("Function was vacuous/ineffectual, reattempting...")
                remaining_edit_functs.remove(edit_funct)
        prev_word_tokens = word_tokens
    if no_edit_made and not remaining_edit_functs:
        logging.debug(f"No more edit functions to apply, final string is {word_tokens}, i={i} out of requested k={k} edits")
        return
    return word_tokens


def test_k_edits():
    k=os.environ.get('k', 5)
    tira_word = 'və̀lɛ̀ðᴐ́'
    tokens = TokenArray.tokenize_string(tira_word)
    print(apply_k_edits(tira_word=tira_word, k=k))
    breakpoint()


"""
ABX triplet selection functions.
"""

def get_all_abx_triples(
        word_df: pd.DataFrame,
        distance_matrix: np.ndarray
):
    """
    Calls `get_abx_triple` for each original word in `word_df` and concatenates
    results into a single dataframe.
    """

    # reset index so word_df indices correspond to distance matrix indices
    word_df = word_df.reset_index(drop=True)

    abx_dfs = []

    for word_a_index in tqdm(word_df.index, desc="Selecting ABX triplets for each original word"):
        abx_df = get_abx_triple(word_a_index, word_df, distance_matrix)
        abx_dfs.append(abx_df)
    return pd.concat(abx_dfs, ignore_index=True)

def get_abx_triple(
        word_a_index: int,
        word_df: pd.DataFrame,
        distance_matrix: np.ndarray
) -> pd.DataFrame:
        """
        For a given original word A, select one edited word B and one edited word X
        such that Lev(A,X) > Lev(B,X) and A,X have the same features as each other, but different from B.
    
        Arguments:
            word_a: original word A
            word_df: dataframe containing all edited variants of the original words, with columns for features
            distance_matrix: pairwise Levenshtein distance matrix for all edited variants of the original words
        Returns:
            abx_df: dataframe with one row containing the selected A,B,X triplet, with columns for the words and their features
        """

        word_a = word_df.loc[word_a_index, 'edited_word']
        word_a_features = word_df.loc[word_a_index, 'features']

        word_b_candidate_mask = word_df['features'] != word_a_features
        word_x_candidate_mask = (word_df['features'] == word_a_features)\
            & (word_df.index != word_a_index)
        
        # iterate through all word_b candidates
        # select all word_x candidates that satisfy Lev(A,X) > Lev(B,X)
        # and append all candidates to daaframe
        abx_rows = []
        common_keys = ['root', 'gloss', 'part_of_speech']
        for word_b_index, word_b_candidate in word_df.loc[word_b_candidate_mask].iterrows():
            word_b = word_b_candidate['edited_word']
            word_b_features = word_b_candidate['features']

            common_data = {key: word_b_candidate[key] for key in common_keys}

            for word_x_index, word_x_candidate in word_df.loc[word_x_candidate_mask].iterrows():
                word_x = word_x_candidate['edited_word']
                word_x_features = word_x_candidate['features']
                a_x_distance = distance_matrix[word_a_index, word_x_index]
                b_x_distance = distance_matrix[word_b_index, word_x_index]
                if a_x_distance > b_x_distance:
                    abx_rows.append({
                        'word_a': word_a,
                        'word_b': word_b,
                        'word_x': word_x,
                        'word_a_features': word_a_features,
                        'word_b_features': word_b_features,
                        'word_x_features': word_x_features,
                        'a_x_distance': a_x_distance,
                        'b_x_distance': b_x_distance,
                        **common_data,
                    })
        abx_df = pd.DataFrame(abx_rows)
        return abx_df
        

def main():
    args = get_args()

    df = pd.read_csv(wordlist)
    edit_rows = []
    edited_words = set()

    for _, row in df.iterrows():
        word = row['word']
        index = row.name

        # include row for original word
        edit_rows.append({
            'word_index': index,
            'edited_word': word,
            'k': 0,
            'edits': '',
            **row,
        })

        for k in range(1, args.k + 1):
            for _ in range(args.num_variants):
                num_attempts = 0
                while num_attempts < 5:
                    num_attempts += 1
                    edited_word_tokens = apply_k_edits(word, k)
                    if not edited_word_tokens:
                        continue
                    edited_word = str(edited_word_tokens)
                    if edited_word in edited_words:
                        logging.debug(f"Edited word {edited_word} already generated, reattempting...")
                        continue
                    break
                if not edited_word_tokens:
                    logging.debug(f"Could not apply {k} edits to word {word} after exhausting all edit functions.")
                    continue
                edited_words.add(edited_word)
                edit_rows.append({
                    'word_index': index,
                    'edited_word': edited_word,
                    'k': k,
                    'edits': str(edited_word_tokens.token_history),
                    **row,
                })
        
    edited_df = pd.DataFrame(edit_rows)
    edited_df.to_csv(edited_wordlist, index_label='edited_word_index')

    # compute pairwise Levenshtein distance for all original and edited words
    # for one root at a time
    unique_roots = edited_df['root'].unique()
    abx_df_list = []
    for root in tqdm(unique_roots, desc="Computing ABX triplets for each root"):
        root_mask = edited_df['root'] == root
        words_w_root = edited_df.loc[root_mask, 'edited_word'].tolist()
        distance_matrix = np.zeros((len(words_w_root), len(words_w_root)), dtype=np.int16)
        for i, word_i in tqdm(list(enumerate(words_w_root)), desc="Computing pairwise distances"):
            for j, word_j in enumerate(words_w_root[i+1:], start=i+1):
                distance_matrix[i, j] = levenshtein_distance(word_i, word_j)
                distance_matrix[j, i] = distance_matrix[i, j]

        # now that we have the pairwise edit distances, we can select
        # triplets of A,B,X words
        # where Lev(A,X) > Lev(B,X)
        # and A,X have the same features as each other, but different from B

        # for now, try to generate one triplet per original word
        # but this can be scaled up in the future
        abx_df = get_all_abx_triples(edited_df.loc[root_mask], distance_matrix)
        abx_df_list.append(abx_df)
    abx_df = pd.concat(abx_df_list, ignore_index=True)
    abx_df.to_csv(abx_wordlist, index=False)


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--k', type=int, default=5, help='Maximum number of edits to apply to each word')
    parser.add_argument(
        '--num-variants',
        type=int,
        default=10,
        help='Number of edited variants to generate for each word for each value '
             'of k from 1 to k. For example, if k=5 and num-variants=10, then for each word, '
             '10 variants will be generated with 1 edit, 10 variants with 2 edits, ..., and 10 variants with 5 edits.')
    return parser.parse_args()

if __name__ == '__main__':
    main()