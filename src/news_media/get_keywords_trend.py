"""
This script produces the keywords frequency measures to use to analyse the prominence of keywords in the press
over time.

Input data: TO ADD

Outputs: TO ADD

How to run it to calculate frequency measure for a two-week (fornight) period:
python -m src.news_media.get_keywords_trend -unit_agg "2W-MON"
"""

import re
import os
import pickle
import pandas as pd

from typing import List
from datetime import datetime

from sklearn.feature_extraction.text import CountVectorizer
from nltk import pos_tag
from nltk.util import ngrams
from nltk.tokenize import word_tokenize

from src.utils import load_config_yaml, chain_functions
from src.preproc_text import tokenise_sent, tokenise_word, remove_punctuation, remove_stopwords, flatten_irregular_listoflists

# Constants
UK_FILENAME = "preproc_kword_sent.pickle"
CONFIG_FILE_NAME = "keywords.yaml"
SUBWK_TO_KW_MAP = "subkw_to_kw_map.yaml"
DIR_DATA_INT = os.environ.get("DIR_DATA_INTERIM")
DIR_EXT = os.environ.get("DIR_EXT")

# Load configuration file
CONFIG_FILE = os.path.join(DIR_EXT, CONFIG_FILE_NAME)
CONFIG = load_config_yaml(CONFIG_FILE)

KWORDS = CONFIG['Actors'] + CONFIG['BehavSci'] + CONFIG['Behav_ins'] + CONFIG[
    'Behav_chan'] + CONFIG['Behav_pol'] + CONFIG['Behav_anal'] + CONFIG[
        'Psych'] + CONFIG['Econ_behav'] + CONFIG['Econ_irrational'] + CONFIG[
            'Nudge'] + CONFIG['Nudge_choice'] + CONFIG['Nudge_pater']
OTHER_IMPORTANT_WORDS = CONFIG['Covid'] + CONFIG['Fatigue'] + CONFIG['Immunity']

# as some sub-keywords are quoted
QUOTED_KWORDS = ["'" + word + "'" for word in KWORDS] + [
    "'" + word for word in KWORDS
] + [word + "'" for word in KWORDS] + ['"' + word + '"' for word in KWORDS] + [
    '"' + word for word in KWORDS
] + [word + '"' for word in KWORDS]
ALL_WORDS_TO_EXTRACT = KWORDS + QUOTED_KWORDS

# keywords that are composed by more than one word
NONUNIGRAMS = [
    kword.replace("_", " ") for kword in KWORDS + OTHER_IMPORTANT_WORDS
    if "_" in kword
]

# CONFIG.keys that do not contain keyword groups
NON_KWORD_CONFIG = ["NgramRange"]

# load lookup sub-keywords to keywords
SUBKEYWORDS_TO_KEYWORDS_MAP = load_config_yaml(
    os.path.join(DIR_EXT, SUBWK_TO_KW_MAP))


# preprocess text first: lower, remove punctuation and stopwords, substitute n-gram keywords with their unigram version
def from_ngrams_to_unigrams(text: str) -> str:
    """Substitites n-gram keywords with their underscored unigram version"""
    for kword in NONUNIGRAMS:
        text = text.replace(kword, kword.replace(" ", "_"))
    return text


def remove_special_symbols(text: str) -> str:
    symbols = [
        "©", "\xad", "•", "…", "●", "“", "”", "•'", "\u200b", "£", "'", "'s",
        "·", "»", "com/"
    ]
    for symb in symbols:
        if symb not in text:
            continue
        text = text.replace(symb, "")
    return text


# We only keep NOUNS and keywords
# Assumptions: anything else is not adding to the content of the articles
TEXT_PREPROC_PIPE = chain_functions(
    lambda x: x.lower(),
    remove_special_symbols,
    lambda x: re.sub(r'[.]+(?![0-9])', r' ', x),
    from_ngrams_to_unigrams,
    tokenise_sent,
    tokenise_word,
    lambda x: [[
        word for (word, pos) in pos_tag(sent) if (pos.startswith("N")) or
        (word in KWORDS) or (word in OTHER_IMPORTANT_WORDS)
    ] for sent in x],
    remove_punctuation,
    remove_stopwords,
    flatten_irregular_listoflists,
    list,
    lambda x: ' '.join(x),
)


class NewsArticles:
    """
    """

    __COLS_GROUPBY_DICT = {}

    def __init__(self):

        with open(os.path.join(DIR_DATA_INT, UK_FILENAME), "rb") as input_file:
            df = pickle.load(input_file)

        # ENSURE DATES BEFORE 27-JAN-2020 WERE REMOVED
        # this exclude 6 articles published on 23-JAN and 26-JAN 2020
        print(df.shape)
        print(df.article_id.nunique())
        df = df[~df.pub_date.isin(
            ['January 23, 2020 Thursday', 'January 26, 2020 Sunday'])].copy()
        print(df.shape)
        print(df.article_id.nunique())
        print(df.columns)
        # this version of data has repeasted measure per article (1 row per keyword occurrence)

        # create version of data with only unique info to each article: id, pub_date_dt, title, full_text
        df_uniq_article = df.drop_duplicates('article_id', 'last')[[
            'article_id', 'pub_date_dt', 'title', 'full_text'
        ]]
        print(df_uniq_article.article_id.nunique())

        # Preprocess Text
        df_uniq_article['preproc_text'] = [
            TEXT_PREPROC_PIPE(article) for article in df_uniq_article.full_text
        ]
        # Calculate count of unique nouns per article ('word_count')
        df_uniq_article['word_count'] = [
            NewsArticles.get_num_ngrams(text, 1)
            for text in df_uniq_article.preproc_text
        ]

        # Append 'word_count' to doc-term matrix dataset as well
        df = df.merge(df_uniq_article[['article_id', 'word_count']],
                      on='article_id')

        self.data_raw = df
        self.data_uniq_article = df_uniq_article
        self.__COLS_GROUPBY_DICT = expand_dict(SUBKEYWORDS_TO_KEYWORDS_MAP)
        # private class-instance attributes
        self._allwords_raw_tf = NewsArticles._compute_allwords_raw_tf(
            news_df=df_uniq_article)
        # NOT NEEDED self._unigram_count_perdoc = [
        #    NewsArticles.get_num_ngrams(text, 1)
        #    for text in df_uniq_article.preproc_text
        # ]
        self._kword_rawfreq = NewsArticles.get_kword_rawfreq(news_df=df)

        self.kword_rawfreq_agg = None
        self._kword_relfreq_agg = None
        self._kword_yn_occurrence = None
        self.kword_docfreq_agg = None
        self._kword_reldocfreq_agg = None
        self._kword_rfrdf_agg = None

    @staticmethod
    def _compute_allwords_raw_tf(news_df: pd.DataFrame,
                                 text_col='preproc_text') -> pd.DataFrame:
        """
        Computes the document-term frequency matrix for all the unigram nouns in the preprocessing texts.

        Args:
            news_df: pandas.Dataframe, results of `get_news_articles.IngestNews`.

        Returns:
            The document-term frequency matrix for all the unigrams in the preprocessed corpus of articles.
        """
        vec = CountVectorizer(stop_words=None,
                              tokenizer=word_tokenize,
                              ngram_range=(1, 1),
                              token_pattern=r"(?u)\b\w\w+\b")

        results_mat = vec.fit_transform(news_df[text_col])

        # sparse to dense matrix
        results_mat = results_mat.toarray()

        # get the feature names from the already-fitted vectorizer
        vec_feature_names = vec.get_feature_names()

        # make a table with word frequencies as values and vocab as columns
        out_df = pd.DataFrame(data=results_mat, columns=vec_feature_names)

        # add article id and pub date as indexes
        # we use the property of CountVectorizer to keep the order of the original texts
        out_df["pub_date"] = news_df["pub_date_dt"].tolist()
        out_df["article_id"] = news_df["article_id"].tolist()
        out_df["word_count"] = news_df["word_count"].tolist()

        out_df.set_index(['pub_date', 'article_id', 'word_count'],
                         append=True,
                         inplace=True)

        return out_df

    @property
    def allwords_raw_tf(self):
        return self._allwords_raw_tf

    @staticmethod
    def get_kword_rawfreq(news_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the raw frequency count of the keywords per articles.

        CountVectorizer was missing to detect a few cases (16 out of more than 1600 cases)
        probably because some of these keyword instances were quoted or not recognise as 'noun'
        in the pre-processing step.

        To ensure consistency with number of cases of keyword occurrences in the sentiment-analysis,
        we will use the keyword occurrences extracted for the keyword-sentiment extraction,
        after they have been coded and filtered by researchers. Researchers also deleted a few
        identified cases while manually coding the sentences, as they were duplicates. 
        """

        df_list_kwords = news_df.groupby(
            ['article_id', 'word_count',
             'pub_date_dt'])['kword'].apply(list).reset_index()

        df_kwords = df_list_kwords.join(
            pd.get_dummies(
                pd.DataFrame(
                    df_list_kwords['kword'].tolist()).stack()).sum(level=0))

        # drop 'kword' col containing list of keywords now that each keyword is a separate col
        df_kwords.drop('kword', axis=1, inplace=True)

        # set indices
        df_kwords.set_index(['pub_date_dt', 'article_id', 'word_count'],
                            append=True,
                            inplace=True)

        return df_kwords

    @property
    def kword_rawfreq(self):
        return self._kword_rawfreq

    def get_kword_rawfreq_agg(self, unit="W-MON"):
        """
        Calculates and returns the raw keyword frequency count of each higher-level keyword per week or forthnight.
        A week starts on a Monday (i.e., dates are grouped by week starting the first Monday before the date),
        and by good coincidence, on 23-Mar-2020 the start date of UK lockdown.
        The first week starts on 20 Jan (Mon) so it will have three days less of data as our data series starts on 23 Jan.

        Args:
            unit:  Either "W-MON" or "2W-MON" for weekly or fortnightly estimates, respectively.
        Returns:
            Raw frequency count of each higher-level keyword per week or forthnight
        """
        if unit == "W-MON":
            unit_name = "week"
        elif unit == "2W-MON":
            unit_name = "fortnight"
        else:
            raise KeyError("'unit' must be either 'W-MON' or '2W-MON'")

        try:
            self.kword_rawfreq_agg = self._kword_rawfreq.reset_index(
                'word_count').resample(unit,
                                       level=1,
                                       label='left',
                                       closed='left').sum()
            self.kword_rawfreq_agg.index.names = [f"{unit_name}_starting"]
        except AttributeError:
            raise AttributeError("`kword_rawfreq` must be calculated first!")
        return self.kword_rawfreq_agg

    @property
    def kword_relfreq_agg(self):
        """Calculates the weekly relative keyword frequency by dividing the number of a keyword's occurrences in a week
        by the total number of words published that week."""
        if self._kword_relfreq_agg is None:
            try:
                self._kword_relfreq_agg = self.kword_rawfreq_agg.iloc[:, 1:].div(
                    self.kword_rawfreq_agg.word_count, axis=0)
            except AttributeError:
                raise AttributeError(
                    "`kword_rawfreq_agg` must be calculated first!")
        return self._kword_relfreq_agg

    def get_kword_docfreq_agg(self, unit="W-MON"):
        """
        Calculates document-frequency (df) for each keyword and week or fortnight. For each keyword, k:
        docfreq_k = [{d in D | k in d}] / |D|
        that is, for each week (fortnight), the number of articles that contains k divided by the total number of published articles.

        That is, `docfreq_k` is not calculated with respect to the whole collection of articles, but
        to the collection of articles in a given week. This will allows us to compare `docfreq_k` trends over time.

        Args:
            unit:  Either "W-MON" or "2W-MON" for weekly or fortnightly estimates, respectively.
        """
        if unit == "W-MON":
            unit_name = "week"
        elif unit == "2W-MON":
            unit_name = "fortnight"
        else:
            raise KeyError("'unit' must be either 'W-MON' or '2W-MON'")

        if self.kword_docfreq_agg is None:
            try:
                articles_count = self._kword_yn_occurrence.reset_index(
                    'article_id').article_id.resample(unit,
                                                      level=1,
                                                      label='left',
                                                      closed='left').count()
                kword_doc_freqs = self._kword_yn_occurrence.resample(
                    unit, level=1, label='left', closed='left').sum()
                self.kword_docfreq_agg = kword_doc_freqs.merge(
                    articles_count, on='pub_date_dt')
                self.kword_docfreq_agg.rename(
                    columns={'article_id': 'article_count'}, inplace=True)
                self.kword_docfreq_agg.index.names = [f"{unit_name}_starting"]
            except AttributeError:
                raise AttributeError(
                    "`kword_yn_occurrence` must be calculated first!")
        return self.kword_docfreq_agg

    @property
    def kword_reldocfreq_agg(self):
        """Calculates and returns the keyword's relative document frequency for each week.
        This is the number of articles published that week in which the keyword is mentioned,
        divided by the total number of articles published that week."""

        if self._kword_reldocfreq_agg is None:
            try:
                # iloc[:, :-1] excludes the 'article_count' col
                self._kword_reldocfreq_agg = self.kword_docfreq_agg.iloc[:, :-1].div(
                    self.kword_docfreq_agg.article_count, axis=0)
            except AttributeError:
                raise AttributeError(
                    "`kword_docfreq_agg` must be calculated first!")
        return self._kword_reldocfreq_agg

    @property
    def kword_rfrdf_agg(self):
        """"""
        if self._kword_rfrdf_agg is None:
            try:
                unit_name = self._kword_relfreq_agg.index.name
                print(unit_name)
                long_kword_relfreq = pd.melt(
                    self._kword_relfreq_agg.reset_index(),
                    id_vars=[unit_name],
                    var_name='kword',
                    value_name='rkf')
                long_kword_reldocfreq = pd.melt(
                    self._kword_reldocfreq_agg.reset_index(),
                    id_vars=[unit_name],
                    var_name='kword',
                    value_name='rdf')
                week_rkf_rdf = long_kword_relfreq.merge(
                    long_kword_reldocfreq, on=[unit_name, 'kword'])
                # calculate final metrics
                week_rkf_rdf[
                    "rkf*rdf"] = week_rkf_rdf['rkf'] * week_rkf_rdf['rdf']
                self._kword_rfrdf_agg = week_rkf_rdf[[
                    unit_name, 'kword', 'rkf*rdf', 'rkf', 'rdf'
                ]]

            except AttributeError:
                raise AttributeError(
                    "'kword_relfreq_agg' and 'kword_reldocfreq_agg' must both be calculated first!"
                )
        return self._kword_rfrdf_agg

    @property
    def kword_yn_occurrence(self):
        """ Returns whether a keyword occurs in an article (1) or not (0)."""
        if self._kword_yn_occurrence is None:
            try:
                self._kword_yn_occurrence = self._kword_rawfreq.applymap(
                    lambda cell: 1 if cell > 0 else 0)
            except AttributeError:
                raise AttributeError(
                    "`kword_rawfreq` must be calculated first!")
        return self._kword_yn_occurrence

    @property
    def unigram_count_perdoc(self):
        return self._unigram_count_perdoc

    @staticmethod
    def get_num_ngrams(text: str, ngram: int) -> int:
        """
        Counts number of n-grams in a text.
        """
        return len(list(ngrams(text.split(), ngram)))

    @staticmethod
    def _extract_date(dates_series: pd.Series) -> List[datetime]:
        """
        Encode the dates contained in a pandas.Series in date format.

        Args:
            dates_series: pandas.Series containing dates, results of `get_news_articles.IngestNews`.

        Returns:
            List of dates in date format.
        """
        date_formats = {3: r"%B %d, %Y", 4: r"%B %d, %Y %A"}

        def _get_date_format(date: str):
            return date_formats[len(date.split())]

        list_dates = [
            datetime.strptime(date, _get_date_format(date))
            for date in dates_series
        ]

        return list_dates


# TODO: move to src.utils
def expand_dict(d: dict) -> dict:
    """
    Expands the original sub-keyword to keyword mapping dictionary where keyword are keys and the
    correspoding sub-keywords are their values (as either string - if one - or list - if several),
    so that each sub-keyword is a key and the corresponding keyword is its value.

    Example:
    {'k1': ['sk1', 'sk2', 'sk3]} => {'sk1':'k1', 'sk2':'k1', 'sk3':'k1'}

    Args:
        d:  the original sub-keyword to keyword mapping dictionary

    Returns:
        The expanded dictionary.
    """

    several_subkws_keys = [k for k, v in d.items() if isinstance(v, list)]
    several_subkws_dict = {v: k for k in several_subkws_keys for v in d[k]}
    single_subkw_dict = {v: k for k, v in d.items() if isinstance(v, str)}

    # combine the two (the two dictionaries do not share keys)
    several_subkws_dict.update(single_subkw_dict)

    return several_subkws_dict


if __name__ == "__main__":

    import argparse

    kword_parser = argparse.ArgumentParser(
        description='Run src.news_media.get_keywords_trend module')

    kword_parser.add_argument(
        '-unit_aggregation',
        type=str,
        action='store',
        choices=['W-MON', '2W-MON'],
        help="choose between W-MON' (weekly) or '2W-MON' (forthnight)",
        dest="chosen_unit_agg",
        required=False)

    args = kword_parser.parse_args()

    if args.chosen_unit_agg:
        unit_agg = args.chosen_unit_agg
    else:
        unit_agg = "W-MON"

    print(f"Aggregation method selected: {unit_agg}")

    uk_news = NewsArticles()
    print(uk_news.data_raw.shape)
    print(uk_news.data_uniq_article.shape)

    # calculate all the metrics
    uk_news.kword_rawfreq
    uk_news.get_kword_rawfreq_agg(unit=unit_agg)
    uk_news.kword_relfreq_agg
    uk_news.kword_yn_occurrence
    uk_news.get_kword_docfreq_agg(unit=unit_agg)
    uk_news.kword_reldocfreq_agg
    uk_news.kword_rfrdf_agg

    # save them as csv so that they can be loaded in in R
    uk_news.kword_rawfreq.to_csv(
        os.path.join(DIR_DATA_INT, "kword_rawfreq.csv"))
    uk_news.kword_yn_occurrence.to_csv(
        os.path.join(DIR_DATA_INT, "kword_yn_occurrence.csv"))
    uk_news.kword_rawfreq_agg.to_csv(
        os.path.join(DIR_DATA_INT, f"kword_rawfreq_{unit_agg}.csv"))
    uk_news.kword_docfreq_agg.to_csv(
        os.path.join(DIR_DATA_INT, f"kword_docfreq_{unit_agg}.csv"))
    uk_news.kword_rfrdf_agg.to_csv(
        os.path.join(DIR_DATA_INT, f"kword_rfrdf_{unit_agg}.csv"))
