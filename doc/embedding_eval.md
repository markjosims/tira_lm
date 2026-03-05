# embedding_eval
Sentences for evaluating Tira embeddings on an ABX task.
The task is structured as follows:
Sentences $\mathbf{a}$, $\mathbf{b}$ and $\mathbf{x}$ have words $a_i$, $b_j$ and $x_k$ respectively.
$a_i$ is the same lemma and form as the word $x_k$, and $b_j$ is either a different form of the same word or else a different word that is similar in spelling to $a_i$ and $x_k$.

## B has different inflection than A
### Imperative vs. imperfective.
Sentences A and X have the same TAM, sentences B and X have the same object noun.
If embedding A is reliably closer to X, then embedding model is correctly modeling the TAM difference rather than being influenced by the object noun.

- A. **və̀lɛ̀ðá** ápɾíɲá
- B. já *və́lɛ̀ðà* ðàŋàlà
- X. **və̀lɛ̀ðá** ðáŋàlà

Create examples with different verbs, deixis values, object nouns, agreement classes for the imperfective verb and spelling differences between A and X target words.
Include some 'hard' confounds where all words have the same tone melody.

### Nominative vs. accusative
- A. **ŋɛ́n** ŋàŋt̪ᴐ́ múðᴐ̀
- B. múðù kàŋt̪ᴐ́ *ŋɛ́nɛ́*
- X. múðù kàŋt̪ᴐ́ **ŋɛ́n**

Tests that the embedding is more sensitive to the theta-role of the noun than to its surface position in the sentence.
Try with various nouns, including cases where the subject and object nouns differ in class agreement and where they have the same class, and also vary whether the confound noun has a distinct accusative form or not.
Also test different misspellings of the target word in the B and X sentences.

## A and X share root, B and X share inflection
- A. ùɾnᴐ̀ **kàɾᴐ́** t̪ólí t̪ícò
- B. ùɾnᴐ̀ *ká rᴐ́* t̪ólíɲá t̪ícò
- X. ùɾnᴐ̀ **ká àɾᴐ́** àpɾí jícə̀lò

Vary spelling of verb in A and X, e.g. whether it has the same consonant as 'stab'.

## B and A are different parts of speech
### Verb vs. noun
The verb in A resembles the noun in B.
The verb in X has less string similarity to the verb in A.

- A. **rᴐ́** múðᴐ̀
- B. *rô* rɛ́ múðù
- X. kúkù **kə̀rᴐ́** múðᴐ̀

This tests whether the embedding is capturing the verb/noun distinction or is more sensitive to spelling and surface configuration.
Vary the spelling of the verb and noun, and whether the sentence-final noun has a distinct accusative or not.
Also vary the inflectional value of the verb in X to modulate how closely it resembles the string of the verb in A.

Verb/noun pairs to try:
- r 'stab' vs rô 'sticks'
- ð 'slice' vs. ðô 'stick'
- r 'stab' vs. lᴐ́rᴐ́ 'back'

### Noun vs. verb
Similar to the case above but now we're trying to distinguish nouns with and without accusative case from a near-homophonous verb.

- A. **ðô** nɛ̀ ðár
- B. *ðᴐ́* ðár
- X. káɾᴐ́ **ðówɛ̀**