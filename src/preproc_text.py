# import os
import string
import re
import collections
# import pandas as pd
from typing import List, Tuple

from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import stopwords

wordnet_lemmatiser = WordNetLemmatizer()

# Constants
BASIC_STOPWORDS = stopwords.words("english")


def tokenise_sent(text: str) -> List[str]:
    """
    Sentence-tokenises a paragraph of text of any sentence length.
    Returns a list of string sentences.

    Args:
        text:  A paragraph of text (str).

    Returns:
        A list of string sentences.
    """
    try:
        return sent_tokenize(text)
    except TypeError as e:
        raise e
    except Exception:  # empty string as input
        return []


def tokenise_word(parag: List[str]) -> List[List[str]]:
    """
    Word-tokenises sentences within a text.
    Requires a list of string sentences as input, e.g. ['I love dogs.', 'Me too!'].
    Returns a list of lists of tokens, e.g. [[ 'I', 'love', 'dogs', '.'],  ['Me', 'too', '!']].

    Args:
        parag:  A paragraph of sentenced-tokenised text as list of strings.

    Returns:
        A list of lists of token word (i.e., sublists preserve sentence boundaries).

    """

    if isinstance(parag, str):
        raise TypeError("parag must be a list (of strings!")

    try:
        return [word_tokenize(sent) for sent in parag]
    except TypeError as e:
        raise e
    except Exception:
        return []


def lemmatise(parag: List[List[Tuple]]) -> List[List[str]]:
    """
    Lemmatises each sentence within a text (inputted as a list of (POS-tag, word) tuples), using Wordnet POS tags.
    When no wordnet POS tag is avalable, it returns the original word (i.e., without lemmatisation).

    Args:
        parag:  A paragraph of POS-tagged sentences as a list of sublists (sentences) of tuples (POS-tag, word).

    Returns:
        A list of sublists of lemmatised tokens (sublists preserve sentence boundaries).
    """

    lemmatised_parag = [[
        wordnet_lemmatiser.lemmatize(wordPOS_tuple[0],
                                     pos=_get_wordnet_pos(wordPOS_tuple[1]))
        if _get_wordnet_pos(wordPOS_tuple[1]) else wordPOS_tuple[0]
        for wordPOS_tuple in sent
    ] for sent in parag]

    return lemmatised_parag


def tag_pos(parag: List[List[str]]) -> List[List[Tuple]]:
    """
    Tags sentences with Part-of-speech (POS) using Penn Treebank.
    For each sentence, it returns a list with (POS-tag, word) tuples.

    Args:
        parag:  A paragraph of word-tokenised sentences as a list of sublists (sentences) of strings (tokens).

    Returns:
        A list of sublists of (POS-tag, token) tuples (sublists preserve sentence boundaries).

    """

    return [pos_tag(sent) if parag else "" for sent in parag]


def _get_wordnet_pos(treebank_tag: str):
    """
    Maps the Peen Treebank tags to WordNet POS names.
    Adapted from: https://stackoverflow.com/questions/15586721/wordnet-lemmatization-and-pos-tagging-in-python

    Args:
        treebank_tag: the POS-tags produced by `tag_pos` using Penn Treebank.

    Returns:
        Wordnet POS names.
    """

    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    elif treebank_tag.startswith(("V", "M")):
        return wordnet.VERB
    elif treebank_tag.startswith("N"):
        return wordnet.NOUN
    elif treebank_tag.startswith("R"):
        return wordnet.ADV
    elif treebank_tag.startswith("S"):
        return wordnet.ADJ_SAT
    else:
        return ""  # could replace as NOUN as default instead?


def remove_stopwords(parag: List[List[str]],
                     stopwords_list: List[str] = BASIC_STOPWORDS,
                     keep_neg: bool = True,
                     words_to_keep: List = None,
                     extra_stopwords: List = None) -> List[List[str]]:
    """
    Removes specified stop-words.

    Args:
        parag:              lowercase paragraph of word-tokenised sentences as a list of word-token sublists.
                            E.g., [[ "i", "do", "n't", "think"],  ["then", "you", "shoud", "n't", "talk"]]
        stopwords_list:     list of stopwords; (default) English stopwords from nltk.corpus
        keep_neg:           whether to remove negations from list of stopwords; (default) True
        words_to_keep:      (optional) list of current stopwords not to be removed from text
        extra_stopwords:    (optional) list of extra ad-hoc stopwords to be removed from text

    Returns:
        The text paragraph cleaned of the specified stopwords.
    """

    if keep_neg:
        parag = _fix_neg_auxiliary(parag)
        stopwords_list = [
            w for w in stopwords_list if w not in ["no", "nor", "not", "n't"]
        ]
    else:
        # add cases of negation missing from list
        stopwords_list.append("n't")

    if words_to_keep:
        stopwords_list = [
            w for w in stopwords_list
            if w not in [w.lower() for w in words_to_keep]
        ]

    if extra_stopwords:
        stopwords_list += [w.lower() for w in extra_stopwords]

    cleaned_parag = [[w for w in sent if w not in stopwords_list]
                     for sent in parag]

    return cleaned_parag


def _fix_neg_auxiliary(parag: List[List[str]]) -> List[List[str]]:
    """
    Replaces contracted negative forms of auxiliary verbs with negation.
    Useful in pipelines that remove stopwords to still mark negation in sentences.

    Args:
        parag:      lowercase paragraph of word-tokenised sentences as a list of word-token sublists.
                    E.g., [[ "i", "won", "go"],  ["i", "haven't", "either"]]

    Returns:
        The same parag of text but with the negative forms of auxiliary verbs replaced.
        E.g., [[ "i", "not", "go"],  ["i", "not", "either"]]
    """

    contracted_neg_auxs = [
        "don't", "didn", "didn't", "doesn", "doesn't", "hadn", "n't", "hadn't",
        "hasn", "hasn't", "haven", "haven't", "isn", "isn't", "mightn",
        "mightn't", "mustn", "mustn't", "needn", "needn't", "shan't",
        "shouldn", "shouldn't", "wasn", "wasn't", "weren", "weren't", "won",
        "won't", "wouldn", "wouldn't", "aren", "aren't", "couldn", "couldn't"
    ]

    output_parag = [["not" if w in contracted_neg_auxs else w for w in sent]
                    for sent in parag]

    return output_parag


def remove_punctuation(parag: List[List[str]],
                       punct_to_keep: str = '') -> List[List[str]]:
    """
    Removes punctuation from a paragraph of text.

    Args:
        parag:          lowercase paragraph of word-tokenised sentences as a list of word-token sublists.
                        E.g., [[ "i", "will", "."],  ["me", "too", "!"]]
        punct_to_keep:  string of punctuation symbols not to remove (e.g., '!?#')

    Returns:
        The same paragraph of text but with punctuation symbols removed. E.g., [[ "i", "will"],  ["me", "too"]]
    """

    # Update string of punctuation symbols
    if len(punct_to_keep) > 0:
        punctuation_list = ''.join(punct for punct in string.punctuation
                                   if punct not in punct_to_keep)
    else:
        punctuation_list = string.punctuation

    # Remove punctuation
    nopunct_parag = [[token.strip(punctuation_list) for token in sent]
                     for sent in parag]

    # Remove extra white spaces left by removing punctuation symbols
    output_parag = [list(filter(None, sent)) for sent in nopunct_parag]

    return output_parag


def clean_tweet_quibbles(text: str) -> str:
    """
    Removes parts of text that are specific to tweets' text:
    - line breaks symbols: '\n', '\r'
    - anything between <>, including '<' and '>'
    - '**', '*'
    - universal character unicodes: '&#nnnn;'
    - URLs and email addresses

    Args:
        text:   A tweet.

    Returns:
        The tweet text cleaned of line breaks symbols, html < > elements, asteriks, universal character unicodes,
        URLs, and user mentioned.
    """

    # ascii encoding
    text = text.encode("ascii", errors="ignore").decode()

    # remove line breaks
    text = ' '.join(text.split())

    # remove html < >
    text = re.sub(r'<[^>]+>', '', text)

    # remove asteriks
    text = re.sub(r'\*?', '', text)

    # remove unicodes
    text = re.sub(r'&#[^;]+;', '', text)

    # remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # remove user mentioned
    text = re.sub(r'@[\S]+.', '', text)

    return ' '.join(text.split())


def flatten_irregular_listoflists(list_lists: List[List]) -> List[str]:
    """
    Flattens a list of lists that is nested /irregular (e.g. [1, 2, [], [[3]]], 4, [5,6]] ).
    Returns a flattened list generator.

    From: https://stackoverflow.com/questions/2158395/flatten-an-irregular-list-of-lists

    Args:
        list_lists:     a (nested) list of lists.

    Returns:
        A flattened list generator.
    """
    for elem in list_lists:
        if isinstance(elem, collections.Iterable) and not isinstance(
                elem, (str, bytes)):
            yield from flatten_irregular_listoflists(elem)
        else:
            yield elem


def detokenise_list(list_strings: List[str]) -> str:
    """
    Concatenates all the strings in a list of strings into a single string of text.

    E.g. ["I think,", "Therefore, I am."] => "I think. Therefore, I am."

    Args:
        list_strings:     a list of strings.

    Returns:
        A string.
    """

    return " ".join(list_strings)
