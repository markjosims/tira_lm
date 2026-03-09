# tira_lm
Code for training and prompting LMs for Tira text normalization and grammatical parsing

## Investigating grammatical discrimination with learned word embeddings
### Generating ABX eval sets
Source files stored in the `./doc` folder.
- `abx_word_seeds.csv`: List of words to generate spelling variants from, organized into sets.
- `{adjective/adverb/noun}.csv`: List of non-target words used for fill ABX frames but not to be predicted by the model during ABX evaluation.
- `frames.yaml`: Description of each set of frames for generating ABX sentences along with the logic for selecting words (both target and non-target).


Stages and data flow:
1. `scripts/abx_data/edited_word_generator.py`: Generates file `data/abx_words.csv`, which contains a list of seed words with random edits applied.
For each seed word attempts to generate a set of $n=10$ edited words containing $k$ edits for $k\in[1,5]$ if possible.
For some words, less than $n$ edited variants may be possible for a given value of $k$.
2. `scripts/sentence_frame_builder.py`: Generates all possible permutations of frames from `doc/frames.yaml` filled with target and non-target words from lists in the `doc/` folder.
At this point no edits have been applied to the target words: rather, they are given in their canonical form.
Saves frames to `data/abx_frames.csv`.
3. `scripts/populate_sentence_frames.py`: For every frame in `data/abx_frames.csv`, samples one edited word corresponding to the target word $a_\mathrm{target}$ in sentence $\mathbf{a}$ for every $k$ value $k\in[1,5]$.
Then samples an edited word for sentence $\mathbf{b}$ and $\mathbf{x}$ where  $\mathrm{Levenshtein}(b_\mathrm{target},x_\mathrm{target})<\mathrm{Levenshtein}(a_\mathrm{target},x_\mathrm{target})$.
Saves the populate frame sentences to `data/abx_sentences.csv`.


### TODO
- [ ] Programmatically create ABX sentence sets for eval
    - [X] Generate edited target words
    - [X] Generate ABX sentence frames
    - [ ] Associate edited words to sentence frames
- [ ] Write script for pre-training BERT encoder on sentences
    - Use character-level tokenization (each character is it's own token)
- [ ] Write script for evaluating encoding model on ABX task
- [X] Adapt mBART script for updated dataset
- [ ] Add option to use LoRA with encoder or decoder on mBART