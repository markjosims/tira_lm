# tira_lm
Code for training and prompting LMs for Tira text normalization and grammatical parsing

## TODO
- [ ] Programmatically create ABX sentence sets for eval
- [ ] Write script for pre-training BERT encoder on sentences
    - Use character-level/unigram tokenization (each character is it's own token)
- [ ] Write script for evaluating encoding model on ABX task
- [ ] Adapt mBART script for updated dataset
- [ ] Add option to use LoRA with encoder or decoder on mBART