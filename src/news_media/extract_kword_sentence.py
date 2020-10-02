import re
import os

from typing import List, Union
from nltk.tokenize import word_tokenize

from src.utils import load_config_yaml
from src.news_media.get_keywords_trend import from_ngrams_to_unigrams
from src.preproc_text import tokenise_sent

# Constants
CONFIG_KWORDS_NAME = "keywords.yaml"
# SUBWK_TO_KW_MAP = "subkw_to_kw_map.yaml"
DIR_EXT = os.environ.get("DIR_EXT")

# Load configuration file
CONFIG_KWORDS_FILE = os.path.join(DIR_EXT, CONFIG_KWORDS_NAME)
CONFIG_KWORDS = load_config_yaml(CONFIG_KWORDS_FILE)

KWORDS = CONFIG_KWORDS['Actors'] + CONFIG_KWORDS['BehavSci'] + CONFIG_KWORDS[
    'Behav_ins'] + CONFIG_KWORDS['Behav_chan'] + CONFIG_KWORDS[
        'Behav_pol'] + CONFIG_KWORDS['Behav_anal'] + CONFIG_KWORDS[
            'Psych'] + CONFIG_KWORDS['Econ_behav'] + CONFIG_KWORDS[
                'Econ_irrational'] + CONFIG_KWORDS['Nudge'] + CONFIG_KWORDS[
                    'Nudge_choice'] + CONFIG_KWORDS['Nudge_pater']
OTHER_IMPORTANT_WORDS = CONFIG_KWORDS['Covid'] + CONFIG_KWORDS[
    'Fatigue'] + CONFIG_KWORDS['Immunity']
WORDS_TO_EXTRACT = KWORDS + OTHER_IMPORTANT_WORDS
QUOTED_WORDS = ["'" + word + "'" for word in WORDS_TO_EXTRACT] + [
    "'" + word for word in WORDS_TO_EXTRACT
] + [word + "'" for word in WORDS_TO_EXTRACT] + [
    '"' + word + '"' for word in WORDS_TO_EXTRACT
] + ['"' + word
     for word in WORDS_TO_EXTRACT] + [word + '"' for word in WORDS_TO_EXTRACT]
ALL_WORDS_TO_EXTRACT = WORDS_TO_EXTRACT + QUOTED_WORDS

# print(ALL_WORDS_TO_EXTRACT)


def collect_kword_opinioncontext(
        article: str) -> List[List[Union[str, tuple]]]:
    """
    Extracts the opinion context for each keyword' occurrence in a article.

    The opinion context is defined as the sentence where the keyword is mentioned.

    Args:
        article:    text of the article as string.

    Returns:
        A list of list. Each sublist contains:
        - first element: the keyword (str) that occurred in the article
        - second element: a tuple of (index i of the sentence where the keyword occurs, the sentence text)
    """

    # add a space after dots that have no space afterwards if they are between alpha characters
    # (e.g., 'I know.But' -> 'I know. But')
    # article = re.sub(r'(?<=[.;!?:])(?=[^\s])', r' ', article)
    article = re.sub(r'([,.!?])([a-zA-Z])', r'\1 \2', article)

    sentences = tokenise_sent(from_ngrams_to_unigrams(article.lower()))

    return [[keyw, (idx, sentence)] for idx, sentence in enumerate(sentences)
            for keyw in ALL_WORDS_TO_EXTRACT
            if keyw in word_tokenize(sentence)]


if __name__ == "__main__":

    import os
    import pandas as pd
    import argparse

    data_parser = argparse.ArgumentParser(
        description='Run src.news_media.extract_kword_sentence module')

    data_parser.add_argument(
        '-which_file',
        type=int,
        action='store',
        choices=[1, 2],
        default=2,
        help="choose 1 (sample up to 9-May) or 2 (sample from 10-May)",
        dest="chosen_file",
        required=False)

    args = data_parser.parse_args()

    if args.chosen_file == 1:
        UK_FILENAME = "uk_news.csv"  # TODO: read from config_news_data.yaml instead
        OUTPUT_FILENAME = "news_df.csv"
        # OUTPUT_FILENAME = 'kword_sents1.csv'  # TO TEST FOR DIFFERENCES
    elif args.chosen_file == 2:
        UK_FILENAME = "uk_news_sample2.csv"
        OUTPUT_FILENAME = "kword_sents_2.csv"
    else:
        raise KeyError(
            "Choose either 1 (sample up to 9-May) or 2 (sample from 10-May)")

    DIR_DATA_INT = os.environ.get("DIR_DATA_INTERIM")
    df = pd.read_csv(os.path.join(DIR_DATA_INT, UK_FILENAME))

    assert "full_text" in df.columns

    # Remove rows that mark start of batches
    bool_series = df["title"].str.startswith("Title (", na=False)
    df = df[~bool_series].copy()
    # Remove duplicates
    df.drop_duplicates(subset="title", keep='last', inplace=True)
    # Remove cases where there is no article text
    df = df[[isinstance(text, str) for text in df.full_text]]

    opinion_contexts = [
        collect_kword_opinioncontext(article=text_article)
        for text_article in df.full_text
    ]

    df['kw_opinioncontext'] = opinion_contexts

    # explode a list like column
    df = df.explode('kw_opinioncontext')

    # split keyword and context across two separate columns
    # way-around due to certain kw_opinioncontext = NaN
    for i in [0, 1]:
        df[str(i)] = [
            k[i] if isinstance(k, list) else k for k in df.kw_opinioncontext
        ]

    df.rename(columns={'0': 'kword', '1': 'opinion_context'}, inplace=True)

    df.columns

    # add a unique id per opinion context
    df['opinion_context_id'] = range(1, df.shape[0] + 1)

    # Save
    df[[
        'Unnamed: 0', 'title', 'pub_date', 'full_text', 'kword',
        'opinion_context', 'opinion_context_id'
    ]].to_csv(os.path.join(DIR_DATA_INT, OUTPUT_FILENAME))
