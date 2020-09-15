"""
# lower case
# sentence tokenise
# work tokenise
# COUNT: extract number of occurrences of <KEY-WORD> in each article [(key_word, count)]
# BINARY: extract whether <KEY-WORD> occurred in article or not [(key_word, 1 or 0)]
# encode date as date
# plot trend over time

"""

import re
import os
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
UK_FILENAME1 = "uk_news.csv"
UK_FILENAME2 = "uk_news_sample2.csv"
USA_FILENAME = "us_news.csv"
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
        "com/"
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

    def __init__(self, country: str = "uk"):

        if country == "usa":
            csv_filepath = os.path.join(DIR_DATA_INT, USA_FILENAME)
            df = pd.read_csv(csv_filepath)
        else:
            csv_filepath1 = os.path.join(DIR_DATA_INT, UK_FILENAME1)
            csv_filepath2 = os.path.join(DIR_DATA_INT, UK_FILENAME2)
            df1 = pd.read_csv(csv_filepath1)
            df2 = pd.read_csv(csv_filepath2)
            # combine the two
            df = df1.append(df2)

        # Remove rows that mark start of batches
        bool_series = df["title"].str.startswith("Title (", na=False)
        df = df[~bool_series].copy()
        # Remove duplicates
        df.drop_duplicates(subset="title", keep='last', inplace=True)
        # Remove cases where there is no article text (are float not str)
        df = df[[isinstance(text, str) for text in df.full_text]]
        # Proprocess Text
        df['preproc_text'] = [
            TEXT_PREPROC_PIPE(article) for article in df.full_text
        ]

        self.data = df
        self.country = country
        self.__COLS_GROUPBY_DICT = expand_dict(SUBKEYWORDS_TO_KEYWORDS_MAP)
        self.dates = NewsArticles._extract_date(dates_series=df.pub_date)
        # private class-instance attributes
        self._allwords_raw_tf = NewsArticles._compute_allwords_raw_tf(
            news_df=df)
        self._unigram_count_perdoc = [
            NewsArticles.get_num_ngrams(text, 1) for text in df.preproc_text
        ]
        self._subkword_raw_tf = None
        self._kword_rawfreq = None
        self.kword_rawfreq_week = None
        self._kword_relfreq_week = None
        self._kword_yn_occurrence = None
        self.kword_docfreq_week = None
        self._kword_reldocfreq_week = None
        self._kword_rfrdf_week = None

    @staticmethod
    def _compute_allwords_raw_tf(news_df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes the document-term frequency matrix for all the unigram words in the preprocessing texts.

        Args:
            news_df: pandas.Dataframe, results of `get_news_articles.IngestNews`.

        Returns:
            The document-term frequency matrix for all the unigrams in the preprocessed corpus of articles.
        """
        vec = CountVectorizer(stop_words=None,
                              tokenizer=word_tokenize,
                              ngram_range=(1, 1),
                              token_pattern=r"(?u)\b\w\w+\b")

        results_mat = vec.fit_transform(news_df['preproc_text'])

        # sparse to dense matrix
        results_mat = results_mat.toarray()

        # get the feature names from the already-fitted vectorizer
        vec_feature_names = vec.get_feature_names()

        # make a table with word frequencies as values and vocab as columns
        out_df = pd.DataFrame(data=results_mat, columns=vec_feature_names)

        # add article id and pub date as indexes
        # we use the property of CountVectorizer to keep the order of the original texts
        out_df["pub_date"] = NewsArticles._extract_date(
            dates_series=news_df["pub_date"])
        out_df["word_count"] = [
            NewsArticles.get_num_ngrams(text=article, ngram=1)
            for article in news_df['preproc_text']
        ]
        out_df.set_index(['pub_date', 'word_count'], append=True, inplace=True)
        out_df.rename_axis(["id", "pub_date", "word_count"], inplace=True)

        return out_df

    @property
    def allwords_raw_tf(self):
        return self._allwords_raw_tf

    @property
    def subkword_raw_tf(self):
        """
        Returns the raw frequency count of the sub-keywords per articles.
        """
        if self._subkword_raw_tf is None:
            self._subkword_raw_tf = self._allwords_raw_tf[[
                col for col in self._allwords_raw_tf.columns if col in KWORDS
            ]]

        return self._subkword_raw_tf

    @property
    def kword_rawfreq(self):
        """
        Calculates and returns the raw frequency count of each higher-level keyword per article.
        """
        if self._kword_rawfreq is None:
            try:
                self._kword_rawfreq = self._subkword_raw_tf.groupby(
                    self.__COLS_GROUPBY_DICT, axis=1).sum()
            except AttributeError:
                raise AttributeError(
                    "`subkword_raw_tf` must be calculated first!")

        return self._kword_rawfreq

    def get_kword_rawfreq_week(self, unit="W-MON"):
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
            self.kword_rawfreq_week = self._kword_rawfreq.reset_index(
                'word_count').resample(unit,
                                       level=1,
                                       label='left',
                                       closed='left').sum()
            self.kword_rawfreq_week.index.names = [f"{unit_name}_starting"]
        except AttributeError:
            raise AttributeError("`kword_rawfreq` must be calculated first!")
        return self.kword_rawfreq_week

    @property
    def kword_relfreq_week(self):
        """Calculates the weekly relative keyword frequency by dividing the number of a keyword's occurrences in a week
        by the total number of words published that week."""
        if self._kword_relfreq_week is None:
            try:
                self._kword_relfreq_week = self.kword_rawfreq_week.iloc[:, 1:].div(
                    self.kword_rawfreq_week.word_count, axis=0)
            except AttributeError:
                raise AttributeError(
                    "`kword_rawfreq_week` must be calculated first!")
        return self._kword_relfreq_week

    def get_kword_docfreq_week(self, unit="W-MON"):
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

        if self.kword_docfreq_week is None:
            try:
                articles_count = self._kword_yn_occurrence.reset_index(
                    'id').id.resample(unit,
                                      level=0,
                                      label='left',
                                      closed='left').count()
                kword_doc_freqs = self._kword_yn_occurrence.resample(
                    unit, level=1, label='left', closed='left').sum()
                self.kword_docfreq_week = kword_doc_freqs.merge(articles_count,
                                                                on='pub_date')
                self.kword_docfreq_week.rename(columns={'id': 'article_count'},
                                               inplace=True)
                self.kword_docfreq_week.index.names = [f"{unit_name}_starting"]
            except AttributeError:
                raise AttributeError(
                    "`kword_rawfreq` must be calculated first!")
        return self.kword_docfreq_week

    @property
    def kword_reldocfreq_week(self):
        """Calculates and returns the keyword's relative document frequency for each week.
        This is the number of articles published that week in which the keyword is mentioned,
        divided by the total number of articles published that week."""

        if self._kword_reldocfreq_week is None:
            try:
                self._kword_reldocfreq_week = self.kword_docfreq_week.iloc[:, :-1].div(
                    self.kword_docfreq_week.article_count, axis=0)
            except AttributeError:
                raise AttributeError(
                    "`kword_docfreq_week` must be calculated first!")
        return self._kword_reldocfreq_week

    @property
    def kword_rfrdf_week(self):
        """"""
        if self._kword_rfrdf_week is None:
            try:
                unit_name = self._kword_relfreq_week.index.name
                print(unit_name)
                long_kword_relfreq = pd.melt(
                    self._kword_relfreq_week.reset_index(),
                    id_vars=[unit_name],
                    var_name='kword',
                    value_name='rkf')
                long_kword_reldocfreq = pd.melt(
                    self._kword_reldocfreq_week.reset_index(),
                    id_vars=[unit_name],
                    var_name='kword',
                    value_name='rdf')
                week_rkf_rdf = long_kword_relfreq.merge(
                    long_kword_reldocfreq, on=[unit_name, 'kword'])
                # calculate final metrics
                week_rkf_rdf[
                    "rkf*rdf"] = week_rkf_rdf['rkf'] * week_rkf_rdf['rdf']
                self._kword_rfrdf_week = week_rkf_rdf[[
                    unit_name, 'kword', 'rkf*rdf'
                ]]

            except AttributeError:
                raise AttributeError(
                    "'kword_relfreq_week' and 'kword_reldocfreq_week' must both be calculated first!"
                )
        return self._kword_rfrdf_week

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

    uk_news = NewsArticles(country="uk")
    print(uk_news.data.shape)

    # calculate all the metrics
    uk_news.subkword_raw_tf
    uk_news.kword_rawfreq
    uk_news.get_kword_rawfreq_week(unit=unit_agg)
    uk_news.kword_relfreq_week
    uk_news.kword_yn_occurrence
    uk_news.get_kword_docfreq_week(unit=unit_agg)
    uk_news.kword_reldocfreq_week
    uk_news.kword_rfrdf_week

    # save them as csv so that they can be loaded in in R
    uk_news.kword_rawfreq.to_csv(
        os.path.join(DIR_DATA_INT, "kword_rawfreq.csv"))
    uk_news.kword_yn_occurrence.to_csv(
        os.path.join(DIR_DATA_INT, "kword_yn_occurrence.csv"))
    uk_news.kword_rawfreq_week.to_csv(
        os.path.join(DIR_DATA_INT, f"kword_rawfreq_{unit_agg}.csv"))
    uk_news.kword_docfreq_week.to_csv(
        os.path.join(DIR_DATA_INT, f"kword_docfreq_{unit_agg}.csv"))
    uk_news.kword_reldocfreq_week.to_csv(
        os.path.join(DIR_DATA_INT, f"kword_reldocfreq_{unit_agg}.csv"))
    uk_news.kword_rfrdf_week.to_csv(
        os.path.join(DIR_DATA_INT, f"kword_rfrdf_{unit_agg}.csv"))
