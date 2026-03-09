"""
Loads sentence frames from a YAML file and ABX words from a CSV file,
and creates a list of frames for each unique seed word combination
that is eligible for the given frame.
"""

from typing import (
    Any, Dict, List, Tuple,
    NamedTuple, Optional
)

import yaml
import pandas as pd
import numpy as np
import argparse
from scripts.constants import (
    frame_config, frame_list, documentation_dir
)
from tqdm import tqdm
from pathlib import Path
import itertools
from dataclasses import dataclass, field
import re

@dataclass
class SourceWord:
    """
    Attributes:
        word:           string form of the word
        index:          the index of the word in its source dataframe, used to look up features
                        and metadata for the word
        source:         name of the dataframe where the word was found, e.g. 'seed_words'
        set_member_id:  an identifier for the set of words that this word belongs to, e.g.
                        'nom', 'acc', 'IPFV.VENT' where the specific value is dependent on
                        the set type of the frame. This is used to determine which words
    """
    word: str
    index: int
    source: str
    set_member_id: Optional[str] = None

    def __post_init__(self):
        if type(self.word) is not str:
            raise ValueError(f"Expected word to be of type str, but got {type(self.word)}")
        if not isinstance(self.index, (int, np.integer)):
            raise ValueError(f"Expected index to be of type int, but got {type(self.index)}")
        if type(self.source) is not str:
            raise ValueError(f"Expected source to be of type str, but got {type(self.source)}")
        if self.set_member_id is not None and type(self.set_member_id) is not str:
            raise ValueError(f"Expected set_member_id to be of type str, but got {type(self.set_member_id)}")

    def __str__(self):
        return self.word

class SlotDict(dict):
    def __set__(self, key, value):
        """
        Check value is a SourceWord before setting the slot value
        """
        if not isinstance(value, SourceWord):
            raise ValueError(f"Expected value of type SourceWord, but got {type(value)}")
        super().__setitem__(key, value)

    def update(self, **kwargs):
        for value in kwargs.values():
            if isinstance(value, dict):
                for sub_value in value.values():
                    if not isinstance(sub_value, SourceWord):
                        raise ValueError(f"Expected value of type SourceWord, but got {type(sub_value)}")
            elif not isinstance(value, SourceWord):
                raise ValueError(f"Expected value of type SourceWord, but got {type(value)}")
        return super().update(**kwargs)
    
    def copy(self):
        new_slot_dict = SlotDict()
        for key, value in self.items():
            new_slot_dict[key] = value
        return new_slot_dict

class AbxSentenceTripletFilled(NamedTuple):
    sentence_a: str
    sentence_b: str
    sentence_x: str
    set_type: str
    word_set: str

    def to_dict(self):
        return {
            'sentence_a': self.sentence_a,
            'sentence_b': self.sentence_b,
            'sentence_x': self.sentence_x,
            'set_type': self.set_type,
            'word_set': self.word_set,
        }

@dataclass
class AbxSentenceTriplet:
    set_type: str
    a_template: str
    b_template: str
    x_template: str
    word_set: str = ''
    a_slots: SlotDict = field(default_factory=SlotDict)
    b_slots: SlotDict = field(default_factory=SlotDict)
    x_slots: SlotDict = field(default_factory=SlotDict)

    def __post_init__(self):
        # check that all slots are SlotDict instances
        if not isinstance(self.a_slots, SlotDict):
            raise ValueError(f"Expected a_slots to be of type SlotDict, but got {type(self.a_slots)}")
        if not isinstance(self.b_slots, SlotDict):
            raise ValueError(f"Expected b_slots to be of type SlotDict, but got {type(self.b_slots)}")
        if not isinstance(self.x_slots, SlotDict):
            raise ValueError(f"Expected x_slots to be of type SlotDict, but got {type(self.x_slots)}")


    def items(self) -> List[Tuple[str, str, Dict[str, SourceWord]]]:
        return [
            ('a', self.a_template, self.a_slots),
            ('b', self.b_template, self.b_slots),
            ('x', self.x_template, self.x_slots),
        ]

    def copy(self) -> 'AbxSentenceTriplet':
        return AbxSentenceTriplet(
            a_template=self.a_template,
            b_template=self.b_template,
            x_template=self.x_template,
            a_slots=self.a_slots.copy(),
            b_slots=self.b_slots.copy(),
            x_slots=self.x_slots.copy(),
            set_type=self.set_type,
            word_set=self.word_set,
        )
    
    def update_data(
        self,
        **data: Dict[str, Any]
    ) -> 'AbxSentenceTriplet':
        new_instance = self.copy()
        for new_attr, value in data.items():
            if new_attr in ['a_slots', 'b_slots', 'x_slots']:
                # update the existing slots dict with the new values
                current_slots = getattr(new_instance, new_attr)
                current_slots.update(**value)
            else:
                setattr(new_instance, new_attr, value)
        return new_instance

    def fill_slots(self) -> AbxSentenceTripletFilled:
        # fill the templates for sentences A, B, and X using the slot values
        sentence_a = self.format_sentence(self.a_template, self.a_slots)
        sentence_b = self.format_sentence(self.b_template, self.b_slots)
        sentence_x = self.format_sentence(self.x_template, self.x_slots)

        for sentence in [sentence_a, sentence_b, sentence_x]:
            assert '{' not in sentence and '}' not in sentence,\
                f"Not all slots were filled in sentence: {sentence}. "\
                f"Remaining slots: {[slot for slot in [self.a_slots, self.b_slots, self.x_slots] if slot]}"

        return AbxSentenceTripletFilled(
            sentence_a=sentence_a,
            sentence_b=sentence_b,
            sentence_x=sentence_x,
            set_type=self.set_type,
            word_set=self.word_set,
        )

    def format_sentence(self, template_str: str, slots: Dict[str, SourceWord]) -> str:
        """
        Replace the slot placeholders in the template string with the
        corresponding words from the slots dict. Use `str.replace` as
        `str.format` does not work well with slot names containing punctuation
        """
        formatted_str = template_str
        for key, value in slots.items():
            if key == '$tgt':
                # we'll be replacing the target word with an edited version of the word
                # at a later point, so keep the target clearly identified for now
                formatted_str = formatted_str.replace(
                    '{' + key + '}', f'[{key}={value.word}]'
                )
            else:
                formatted_str = formatted_str.replace('{'+key+'}', value.word)
        return formatted_str

def load_source_data(args: argparse.Namespace) -> Dict[str, pd.DataFrame]:
    # load all csv files in documentation directory
    source_word_data = {}
    docs_dir = Path(args.docs_dir)
    for csv_file in docs_dir.glob('*.csv'):
        df = pd.read_csv(csv_file, index_col='index')
        source_word_data[csv_file.stem] = df

    return source_word_data

def generate_abx_frames(
        frame: Dict[str, Any],
        source_word_data: Dict[str, pd.DataFrame]
) -> List[AbxSentenceTripletFilled]:
    """
    For a given sentence frame, select all eligible seed words and generate sentences.
    The frame config specifies constraints on how seed words can be selected. For example,
    a constraint 'ab_not_equal' indicates that any word may fill the target slot so long as
    the word is different for sentences A and B. The constraint 'ax_nom' and 'b_acc', on
    the other hand, specifies that the words filling the target slot in sentences A and X
    should both be nominative case forms of the same noun, whereas the word filling the target
    slot in sentence B should be an accusative form of the same noun.

    Arguments:
        frame: a dictionary containing the sentence frame template strings
            and a list of constraints for each arg of the template
            which define the logic for how words may be combined to fill the template.
        source_word_data: a dictionary containing dataframes of source words, including
            the seed words and other words used to fill slots in sentence frames.
    Returns:
        frame_sentences: a list of dictionaries, each containing a generated sentence and its metadata
    """
    template_dict = frame['sentence_templates']

    # if a 'generic' template is provided, use it for all three sentences
    if 'generic' in template_dict:
        assert len(template_dict) == 1,\
            "If 'generic' template is provided, it should be the only template."
        template_dict = {
            'a': template_dict['generic'],
            'b': template_dict['generic'],
            'x': template_dict['generic'],
        }

    # initialize main data structure to hold generated sentences and metadata
    main_template = AbxSentenceTriplet(
        a_template=template_dict['a'],
        b_template=template_dict['b'],
        x_template=template_dict['x'],
        set_type=frame['set_type'],
    )

    # first select $tgt words that satisfy the constraints for the frame
    # start by filtering seed words that match the word set specified in the frame config
    seed_words = source_word_data['abx_word_seeds']
    set_type_mask = seed_words['set_type'] == frame['set_type']
    eligible_seed_words = seed_words.loc[set_type_mask]
    assert not eligible_seed_words.empty, f"No eligible seed words found for frame {frame['name']} with word set {frame['word_set']}"
    
    set_member_ids = eligible_seed_words['set_member_id'].unique().tolist()
    constraints = frame['constraints']
    target_constraints = constraints['$tgt']
    valid_target_combinations = get_valid_target_combinations(
        set_member_ids,
        target_constraints,
    )
    sentences_with_targets = get_sentences_with_targets(
        main_template,
        eligible_seed_words,
        valid_target_combinations,
    )
    filled_sentences = []
    for sentence in sentences_with_targets:
        filled_sentences.extend(fill_nontarget_slots(
            sentence,
            source_word_data,
            constraints,
        ))
    
    return filled_sentences

def get_sentences_with_targets(
    main_template: AbxSentenceTriplet,
    eligible_seed_words: pd.DataFrame,
    valid_target_combinations: List[Tuple[str, str, str]],
) -> List[AbxSentenceTriplet]:
    """
    populate the frame templates with the valid target combinations
    there should be one set per value of 'word_set' in the seed words dataframe
    """
    sentences_with_targets = []
    for word_set in eligible_seed_words['word_set'].unique():
        word_set_mask = eligible_seed_words['word_set'] == word_set
        word_set_seed_words = eligible_seed_words.loc[word_set_mask]
        for a_id, b_id, x_id in valid_target_combinations:
            # expect exactly one word for each set_member_id in the combination
            slot2set_member_id = {
                'a_slots': a_id,
                'b_slots': b_id,
                'x_slots': x_id,
            }
            slots = {}
            for slot_name, set_member_id in slot2set_member_id.items():
                word_mask = word_set_seed_words['set_member_id'] == set_member_id
                words_for_slot = word_set_seed_words.loc[word_mask]
                if words_for_slot.empty:
                    break
                assert len(words_for_slot) == 1,\
                    f"Expected exactly one word for set_member_id {set_member_id} in word set {word_set}, "\
                    f"but found {len(words_for_slot)}. Words: {words_for_slot['word'].tolist()}"
                word_for_slot = words_for_slot.iloc[0]
                word_for_slot = SourceWord(
                    word=word_for_slot['word'],
                    index=word_for_slot.name,
                    source='abx_word_seeds',
                    set_member_id=set_member_id,
                )
                slots[slot_name] = {'$tgt': word_for_slot}
            else:
                # only append sentence instance if all three target words were found for the combination
                sentence_instance = main_template.update_data(word_set=word_set, **slots)
                sentences_with_targets.append(sentence_instance)
    return sentences_with_targets

def fill_nontarget_slots(
    sentence_instance: AbxSentenceTriplet,
    source_word_data: Dict[str, pd.DataFrame],
    constraint_config: Dict[str, List[str]],
) -> List[AbxSentenceTripletFilled]:
    """
    For a given sentence instance with target slots filled, fill the non-target slots
    according to the constraints specified in the frame config. Whereas target slots
    are all taken from 'abx_word_seeds', non-target slots are filled based on the part
    of speech specified in their key in the template.
    
    Like target words, non-target slots are selected based on various constraints specified
    in the frame config. For example, if the constraint 'ab_not_equal' is specified for a
    non-target slot, then any word from the respective dataframe may be selected to fill that
    slot so long as it is different from the word filling the same slot in the other sentence.
    If the constraint 'ax_nom' is specified for a non-target slot, then the word filling that
    slot should be a nominative case form of the same noun as the word filling the same slot
    in sentence X.

    Arguments:
        sentence_instance:  an instance of AbxSentenceTriplet with target slots filled.
        source_word_data:   a dictionary containing dataframes of source words, including
                            the seed words and other words used to fill slots in sentence
                            frames.
    Returns:
        filled_sentences:   a list of instances of AbxSentenceTripletFilled with all slots
                            filled and sentences generated.
    """
    brace_regex = r'\{([^}]+)\}'
    unfilled_slots_by_sentence = [
        re.findall(brace_regex, sentence_instance.a_template),
        re.findall(brace_regex, sentence_instance.b_template),
        re.findall(brace_regex, sentence_instance.x_template),
    ]

    # condense into one list since constraints will determine
    # how slots are filled across sentences
    all_unfilled_slots = set()
    for slot_list in unfilled_slots_by_sentence:
        all_unfilled_slots.update(slot_list)
    
    # for each unfilled slot, get the constraints that apply to that slot
    # and fill the slot according to those constraints
    
    sentence_list = [sentence_instance]

    if 'adverb' in all_unfilled_slots:
        sentence_list = fill_adverb_slots(
            sentence_list,
            source_word_data,
            constraint_config
        )
        all_unfilled_slots.remove('adverb')

    if 'noun' in all_unfilled_slots:
        sentence_list = fill_single_noun_slots(
            sentence_list,
            source_word_data,
            constraint_config
        )
        all_unfilled_slots.remove('noun')

    if 'noun.1' in all_unfilled_slots:
        sentence_list = fill_double_noun_slots(
            sentence_list,
            source_word_data,
            constraint_config,
        )
        all_unfilled_slots.remove('noun.1')
        all_unfilled_slots.remove('noun.2')
    
    if 'class' in all_unfilled_slots:
        sentence_list = fill_class_slots(
            sentence_list,
            source_word_data,
            constraint_config
        )
        all_unfilled_slots.remove('class')

    if 'adjective' in all_unfilled_slots:
        sentence_list = fill_adjective_slots(
            sentence_list,
            source_word_data,
            constraint_config
        )
        all_unfilled_slots.remove('adjective')

    return [sentence.fill_slots() for sentence in sentence_list]
        
_cached_word_data = {}

def _get_source_words_from_dataframe(
        word_data: pd.DataFrame,
        source: str,
) -> List[SourceWord]:
    source_words = []
    for index, row in word_data.iterrows():
        source_word = SourceWord(
            word=row['word'],
            index=index,
            source=source,
            set_member_id=row.get('set_member_id', None),
        )
        source_words.append(source_word)
    return source_words

def _get_feature2adverb(adverb_data: pd.DataFrame) -> Dict[str, SourceWord]:
    # get dictionary mapping verb feature str to matching adverb
    if 'feature2adverb' in _cached_word_data:
        return _cached_word_data['feature2adverb']

    feature2adverb = {}
    for feature_str in adverb_data['constraint'].unique():
        feature_mask = adverb_data['constraint'] == feature_str
        assert feature_mask.sum() == 1,\
            f"Expected exactly one adverb for constraint {feature_str}, but found {feature_mask.sum()}. Adverbs: {adverb_data.loc[feature_mask, 'word'].tolist()}"
        feature2adverb[feature_str] = _get_source_words_from_dataframe(
            adverb_data.loc[feature_mask],
            source='adverb',
        )[0]
    _cached_word_data['feature2adverb'] = feature2adverb
    return feature2adverb

def fill_adverb_slots(
    sentence_list: List[AbxSentenceTriplet],
    source_word_data: Dict[str, pd.DataFrame],
    constraint_config: Dict[str, List[str]],
) -> List[AbxSentenceTriplet]:
    adverb_constraints = constraint_config.get('adverb', [])
    adverb_constraints = set(adverb_constraints)
    # currently only supported behavior for adverbs is to match the aspect
    # of the target verb
    assert adverb_constraints == {'match:aspect'}
    adverb_data = source_word_data['adverb']

    feature2adverb = _get_feature2adverb(adverb_data)
    new_sentence_instances = []

    # for each sentence triplet, get the aspect of the target verb and select the adverb that
    # matches that aspect, then update the slots
    for sentence in sentence_list:
        new_slot_data = {}
        for sentence_name, sentence_template, slots in sentence.items():
            target_word = slots.get('$tgt')
            target_word_features = target_word.set_member_id.split('.')
            target_aspect = target_word_features[0]
            adverb_for_aspect = feature2adverb.get(target_aspect)
            new_slot_data[sentence_name + '_slots'] = {'adverb': adverb_for_aspect}
        new_sentence_instance = sentence.update_data(**new_slot_data)
        new_sentence_instances.append(new_sentence_instance)
    
    return new_sentence_instances

def get_noun_role_mask(
    noun_data: pd.DataFrame,
    role: str
) -> pd.Series:
    if f'noun.role.{role}' in _cached_word_data:
        return _cached_word_data[f'noun.role.{role}']
    role_mask = noun_data['role'] == role
    _cached_word_data[f'noun.role.{role}'] = role_mask
    return role_mask

def get_noun_word_set_mask(
    noun_data: pd.DataFrame,
    word_set: str
) -> pd.Series:
    if f'noun.word_set.{word_set}' in _cached_word_data:
        return _cached_word_data[f'noun.word_set.{word_set}']
    # word set column may be wildcard '*'
    word_set_mask = (
        (noun_data['word_set'] == word_set) |
        (noun_data['word_set'] == '*')
    )
    _cached_word_data[f'noun.word_set.{word_set}'] = word_set_mask
    return word_set_mask

def get_noun_set_type_mask(
    noun_data: pd.DataFrame,
    set_type: str,
    noun_tag: str = 'noun'
) -> pd.Series:
    if f'noun.set_type.{set_type}' in _cached_word_data:
        return _cached_word_data[f'noun.set_type.{set_type}']
    # nouns may specify multiple word set types, separated by |
    # so check for string containment rather than exact match
    set_type_mask = noun_data['set_type'].str.contains(set_type)
    _cached_word_data[f'noun.set_type.{set_type}'] = set_type_mask
    return set_type_mask

def get_class_for_word(
        source_data: Dict[str, pd.DataFrame],
        word: SourceWord
    ) -> SourceWord:
    word_data = source_data[word.source]
    word_row = word_data.loc[word.index]
    word_class_str = word_row['class']
    word_class_obj = SourceWord(
        word=word_class_str,
        index=word.index,
        source=word.source,
        set_member_id=word.set_member_id,
    )
    return word_class_obj

def fill_single_noun_slots(
    sentence_list: List[AbxSentenceTriplet],
    source_word_data: Dict[str, pd.DataFrame],
    constraint_config: Dict[str, List[str]],
    noun_tag: str = 'noun'
) -> List[AbxSentenceTriplet]:
    noun_constraints = constraint_config.get(noun_tag, [])
    noun_constraints = set(noun_constraints)
    
    noun_mask = pd.Series([True] * len(source_word_data['noun']))

    role_constraint = [c for c in noun_constraints if c.startswith('role:')]
    if role_constraint:
        assert len(role_constraint) == 1, f"Expected at most one role constraint for noun slots, but found {len(role_constraint)}: {role_constraint}"
        role = role_constraint[0].split(':')[1]
        noun_mask &= get_noun_role_mask(source_word_data['noun'], role)
        noun_constraints.remove(role_constraint[0])

    # allowed configurations for remaining constraints are:
    # 1. abx_equal: all noun slots should be filled with the same noun
    # 2. ax_equal, bx_not_equal: noun in sentence A and X should be the same, but different from the noun in sentence B
    # 3. bx_equal, ax_not_equal: noun in sentence B and X should be the same, but different from the noun in sentence A
    # 4. ab_equal, ax_not_equal, bx_not_equal: noun in sentence A and B should be the same, no noun in sentence X
    if noun_constraints == {'abx_equal'}:
        new_sentence_instances = []
        for sentence in sentence_list:
            word_set_mask = get_noun_word_set_mask(source_word_data['noun'], sentence.word_set)
            set_type_mask = get_noun_set_type_mask(source_word_data['noun'], sentence.set_type)
            sentence_noun_mask = noun_mask & word_set_mask & set_type_mask
            nouns_for_sentence = source_word_data['noun'].loc[sentence_noun_mask]
            assert len(nouns_for_sentence) > 0, f"No nouns found for sentence with word set {sentence.word_set} and set type {sentence.set_type} after applying constraints."
            nouns_for_sentence = _get_source_words_from_dataframe(
                nouns_for_sentence,
                source='noun',
            )
            for noun in nouns_for_sentence:
                new_sentence = sentence.update_data(
                    a_slots={noun_tag: noun},
                    b_slots={noun_tag: noun},
                    x_slots={noun_tag: noun},
                )
                new_sentence_instances.append(new_sentence)
    elif noun_constraints == {'ax_equal', 'ab_not_equal'}:
        new_sentence_instances = []
        for sentence in sentence_list:
            word_set_mask = get_noun_word_set_mask(source_word_data['noun'], sentence.word_set)
            set_type_mask = get_noun_set_type_mask(source_word_data['noun'], sentence.set_type)
            sentence_noun_mask = noun_mask & word_set_mask & set_type_mask
            nouns_for_sentence = source_word_data['noun'].loc[sentence_noun_mask]
            assert len(nouns_for_sentence) > 0, f"No nouns found for sentence with word set {sentence.word_set} and set type {sentence.set_type} after applying constraints."
            nouns_for_sentence = _get_source_words_from_dataframe(
                nouns_for_sentence,
                source='noun',
            )
            for noun_ax in nouns_for_sentence:
                new_sentence = sentence.update_data(
                    a_slots={noun_tag: noun_ax},
                    x_slots={noun_tag: noun_ax},
                )
                for noun_b in nouns_for_sentence:
                    if noun_b.word != noun_ax.word:
                        new_sentence_b = new_sentence.update_data(
                            b_slots={noun_tag: noun_b},
                        )
                        new_sentence_instances.append(new_sentence_b)
    elif noun_constraints == {'bx_equal', 'ax_not_equal'}:
        new_sentence_instances = []
        for sentence in sentence_list:
            word_set_mask = get_noun_word_set_mask(source_word_data['noun'], sentence.word_set)
            set_type_mask = get_noun_set_type_mask(source_word_data['noun'], sentence.set_type)
            sentence_noun_mask = noun_mask & word_set_mask & set_type_mask
            nouns_for_sentence = source_word_data['noun'].loc[sentence_noun_mask]
            assert len(nouns_for_sentence) > 0, f"No nouns found for sentence with word set {sentence.word_set} and set type {sentence.set_type} after applying constraints."
            nouns_for_sentence = _get_source_words_from_dataframe(
                nouns_for_sentence,
                source='noun',
            )
            for noun_bx in nouns_for_sentence:
                new_sentence = sentence.update_data(
                    b_slots={noun_tag: noun_bx},
                    x_slots={noun_tag: noun_bx},
                )
                for noun_a in nouns_for_sentence:
                    if noun_a.word != noun_bx.word:
                        new_sentence_a = new_sentence.update_data(
                            a_slots={noun_tag: noun_a},
                        )
                        new_sentence_instances.append(new_sentence_a)
    elif noun_constraints == {'ab_equal'}:
        new_sentence_instances = []
        for sentence in sentence_list:
            word_set_mask = get_noun_word_set_mask(source_word_data['noun'], sentence.word_set)
            set_type_mask = get_noun_set_type_mask(source_word_data['noun'], sentence.set_type)
            sentence_noun_mask = noun_mask & word_set_mask & set_type_mask
            nouns_for_sentence = source_word_data['noun'].loc[sentence_noun_mask]
            assert len(nouns_for_sentence) > 0, f"No nouns found for sentence with word set {sentence.word_set} and set type {sentence.set_type} after applying constraints."
            nouns_for_sentence = _get_source_words_from_dataframe(
                nouns_for_sentence,
                source='noun',
            )
            for noun_ab in nouns_for_sentence:
                new_sentence = sentence.update_data(
                    a_slots={noun_tag: noun_ab},
                    b_slots={noun_tag: noun_ab},
                )
                new_sentence_instances.append(new_sentence)
    else:
        raise ValueError(f"Unsupported noun constraint configuration: {noun_constraints}")

    return new_sentence_instances

def fill_double_noun_slots(
    sentence_list: List[AbxSentenceTriplet],
    source_word_data: Dict[str, pd.DataFrame],
    constraint_config: Dict[str, List[str]],
) -> List[AbxSentenceTriplet]:
    """
    Wraps around fill_single_noun_slots to fill slots for sentences with two noun slots, e.g. {noun.1} and {noun.2}.
    """
    sentence_list = fill_single_noun_slots(
        sentence_list,
        source_word_data,
        constraint_config,
        noun_tag='noun.1',
    )
    sentence_list = fill_single_noun_slots(
        sentence_list,
        source_word_data,
        constraint_config,
        noun_tag='noun.2',
    )
    return sentence_list

def fill_class_slots(
    sentence_list: List[AbxSentenceTriplet],
    source_word_data: Dict[str, pd.DataFrame],
    constraint_config: Dict[str, List[str]],
) -> List[AbxSentenceTriplet]:
    class_constraints = constraint_config.get('class', [])
    assert len(class_constraints) == 1,\
        f"Expected exactly one constraint for class slots, but found {len(class_constraints)}"
    match_constraint = class_constraints[0]
    assert match_constraint.startswith('match:'),\
        f"Expected class constraint to be a match constraint, but found {match_constraint}"
    word_to_match = match_constraint.split(':')[1]

    new_sentence_instances = []
    for sentence in sentence_list:
        new_slots = {}
        for sentence_name, sentence_template, slots in sentence.items():
            # not all templates may have the class slot
            if '{class}' not in sentence_template:
                continue

            word_to_match_in_sentence = slots.get(word_to_match)
            assert word_to_match_in_sentence is not None,\
                f"Word to match for class constraint not found in sentence slots. Expected to find {word_to_match} in slots, but found {slots.keys()}"
            class_prefix = get_class_for_word(source_word_data, word_to_match_in_sentence)
            new_slots[sentence_name + '_slots'] = {'class': class_prefix}
        new_sentence_instance = sentence.update_data(**new_slots)
        new_sentence_instances.append(new_sentence_instance)
    return new_sentence_instances
            

def get_valid_target_combinations(
        set_member_ids: List[str],
        target_constraints: List[str],
) -> List[Tuple[str, str, str]]:
    """
    Get a list of words that match the constraints for the target slot.
    words are determined to be eligible based on the 'set_member_id' column
    at present three constraint configurations are supported:

    1:
        - ab_not_equal: indicates that the words filling the 'a' and 'b' slots should
        not be identical
        - ax_equal: indicates that the words filling the 'a' and 'x' slots should be identical
    2:
        - ax_nom: indicates that the words filling the 'a' and 'x' slots should be nominative
        case forms of the same noun
        - b_acc: indicates that the word filling the 'b' slot should be an accusative form of
        the same noun as the word filling the 'a' slot
    3.
        - ax_noun: indicates that the words filling the 'a' and 'x' slots should be the same noun
        - b_verb: indicates that the word filling the 'b' slot should be a verb which is a (near)
        homophone of the noun filling the 'a' and 'x' slots

    Return a list of valid (A, B, X) combinations.
    """
    target_constraints = set(target_constraints)
    inequality_constraints = {'ab_not_equal', 'ax_equal'}
    inequality_and_aspect_constraints = {'ab_not_equal', 'ax_equal', 'ab_aspect_not_equal'}
    noun_case_constraints = {'ax_nom', 'b_acc'}
    noun_verb_constraints = {'ax_noun', 'b_verb'}

    # generate a list of all 2-permutations of the available set_member_ids
    if target_constraints == inequality_constraints:
        # the first element is the AX word type, and the second is the B word type
        valid_id_combinations = list(itertools.permutations(set_member_ids, 2))
        # rearrange into the format (A, B, X)
        valid_id_combinations = [(ax, b, ax) for ax, b in valid_id_combinations]
    # only one type of combination is allowed: [nominative, accusative]
    # first check that set_member_ids match expected values
    elif target_constraints == inequality_and_aspect_constraints:
        # same as inequality constraints
        # but also require that the aspect of the A and X words is different
        valid_id_combinations = list(itertools.permutations(set_member_ids, 2))
        valid_id_combinations = [(ax, b, ax) for ax, b in valid_id_combinations]
        filtered_aspect = []
        for ax, b, _ in valid_id_combinations:
            ax_aspect = ax.split('.')[0]
            b_aspect = b.split('.')[0]
            if ax_aspect != b_aspect:
                filtered_aspect.append((ax, b, ax))
        valid_id_combinations = filtered_aspect
    elif target_constraints == noun_case_constraints:
        assert set(set_member_ids) == {'nom', 'acc'},\
            f"Invalid set_member_ids for noun case constraint: {set_member_ids}. "\
            f"Expected set_member_ids are 'nom' and 'acc'."
        valid_id_combinations = [('nom', 'acc', 'nom')]
    # only one type of combination is allowed: [noun, verb]
    elif target_constraints == noun_verb_constraints:
        assert set(set_member_ids) == {'noun', 'verb'},\
            f"Invalid set_member_ids for noun-verb constraint: {set_member_ids}. "\
            f"Expected set_member_ids are 'noun' and 'verb'."
        valid_id_combinations = [('noun', 'verb', 'noun')]
    else:
        raise ValueError(
            f"Invalid target constraints: {target_constraints}. "
            f"Expected one of: {inequality_constraints}, {inequality_and_aspect_constraints}, "
            f"{noun_case_constraints}, {noun_verb_constraints}."
        )
    return valid_id_combinations

def fill_adjective_slots(
    sentence_list: List[AbxSentenceTriplet],
    source_word_data: Dict[str, pd.DataFrame],
    constraint_config: Dict[str, List[str]],
) -> List[AbxSentenceTriplet]:
    # no constraints currently implemented for adjectives
    # so just fill all adjective slots with all available adjectives
    assert 'adjective' not in constraint_config, "Constraints for adjectives are not currently supported."
    adjective_data = source_word_data['adjective']
    adjectives = _get_source_words_from_dataframe(
        adjective_data,
        source='adjective',
    )
    new_sentence_instances = []
    for adjective in adjectives:
        for sentence in sentence_list:
            new_slots = {}
            for sentence_name, sentence_template, slots in sentence.items():
                if '{adjective}' not in sentence_template:
                    continue
                new_slots = {sentence_name + '_slots': {'adjective': adjective}}
            new_sentence_instance = sentence.update_data(**new_slots)
            new_sentence_instances.append(new_sentence_instance)
    return new_sentence_instances

def main():
    args = get_args()

    # load sentence frames
    with open(args.frames_file, 'r') as f:
        frame_data = yaml.safe_load(f)
    frames = frame_data['data']

    # load source data
    source_word_data = load_source_data(args)

    # for each frame, select eligible seed words and generate sentences
    sentences = []
    for frame in tqdm(frames, desc="Generating sentences from frames"):
        frame_sentences = generate_abx_frames(frame, source_word_data)
        sentences.extend(frame_sentences)

    # convert to dataframe and save
    sentences_dicts = [sentence.to_dict() for sentence in sentences]
    sentences_df = pd.DataFrame(sentences_dicts)
    sentences_df.to_csv(args.output_file, index_label='frame_index')

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate sentences from frames and seed words.")
    parser.add_argument(
        "--frames-file",
        type=str,
        help="Path to the YAML file containing sentence frames.",
        default=frame_config,
    )
    parser.add_argument(
        "--docs-dir",
        type=str,
        help="Path to the directory where documentation files are stored.",
        default=documentation_dir,
    )
    parser.add_argument(
        "--output-file",
        type=str,
        help="Path to the output file where generated sentences will be saved.",
        default=frame_list,
    )
    return parser.parse_args()

if __name__ == '__main__':
    main()
