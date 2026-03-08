"""
Loads sentence frames from a YAML file and ABX words from a CSV file,
and generates sentences by filling the frames with the ABX words.
"""

import yaml
import pandas as pd
import argparse

frames_file = 'doc/frames.yaml'

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

# TODO: the current script should just generate edited words
# triple selection should be taken care of by `sentence_frame_builder.py`
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
        # and append all candidates to dataframe
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


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate sentences from frames and ABX words.")
    parser.add_argument("--frames_file", type=str, required=True, help="Path to the YAML file containing sentence frames.")
    parser.add_argument("--abx_words_file", type=str, required=True, help="Path to the CSV file containing ABX words.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output file where generated sentences will be saved.")
    return parser.parse_args()