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

from typing import List, Union
from datetime import datetime

from sklearn.feature_extraction.text import CountVectorizer
from nltk.util import ngrams
from nltk.tokenize import word_tokenize

from math import log10

from src.utils import load_config_yaml, chain_functions
from src.preproc_text import tokenise_sent, tokenise_word, remove_punctuation, remove_stopwords, flatten_irregular_listoflists

# Constants
UK_FILENAME = "uk_news.csv"
USA_FILENAME = "us_news.csv"
CONFIG_FILE_NAME = "keywords.yaml"
DIR_DATA_INT = os.environ.get("DIR_DATA_INTERIM")
DIR_EXT = os.environ.get("DIR_EXT")

# Load configuration file
CONFIG_FILE = os.path.join(DIR_EXT, CONFIG_FILE_NAME)
CONFIG = load_config_yaml(CONFIG_FILE)

KWORDS = CONFIG['Actors'] + CONFIG['BehavSci'] + CONFIG['Nudge'] + CONFIG[
    'Positive'] + CONFIG['Negative'] + CONFIG['Covid'] + CONFIG[
        'Fatigue'] + CONFIG['Immunity']

# keywords that are composed by more than one word
NONUNIGRAMS = [kword.replace("_", " ") for kword in KWORDS if "_" in kword]

# CONFIG.keys that do not contain keyword groups
NON_KWORD_CONFIG = ["NgramRange", "SingularPlural"]


# preprocess text first: lower, remove punctuation and stopwords, substitute n-gram keywords with their unigram version
def from_ngrams_to_unigrams(text: str) -> str:
    """Substitites n-gram keywords with their underscored unigram version"""
    for kword in NONUNIGRAMS:
        text = text.replace(kword, kword.replace(" ", "_"))
    return text


TEXT_PREPROC_PIPE = chain_functions(
    lambda x: x.lower(), tokenise_sent, tokenise_word, remove_punctuation,
    remove_stopwords, flatten_irregular_listoflists, list,
    lambda x: ' '.join(x), lambda x: re.sub(r'[.]+(?![0-9])', r' ', x),
    from_ngrams_to_unigrams)


class NewsArticles:
    """
    """

    __COLS_GROUPBY_DICT = {}

    def __init__(self, country: str = "uk"):

        if country == "usa":
            csv_filepath = os.path.join(DIR_DATA_INT, USA_FILENAME)
        else:
            csv_filepath = os.path.join(DIR_DATA_INT, UK_FILENAME)
        df = pd.read_csv(csv_filepath)

        # Remove rows that mark start of batches
        bool_series = df["title"].str.startswith("Title (", na=False)
        df = df[~bool_series].copy()
        # Proprocess Text
        df['preproc_text'] = [
            TEXT_PREPROC_PIPE(article) for article in df.full_text
        ]

        self.data = df
        self.country = country
        self.__COLS_GROUPBY_DICT = self.expand_dict(CONFIG)
        self.dates = NewsArticles._extract_date(dates_series=df.pub_date)
        # private class-instance attributes
        self._kword_raw_tf = NewsArticles._compute_kword_raw_tf(news_df=df)
        self._kword_yn_occurrence = None
        self._kword_normlen_tf = None
        self._kword_normlog_tf = None
        self._kword_docfreq = None
        self._theme_rawfreq = None
        self._theme_normlen_f = None
        self._theme_normlog_f = None
        self._theme_docfreq = None
        self._unigram_count_perdoc = None

    @staticmethod
    def _compute_kword_raw_tf(news_df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes the document-term frequency matrix for the keywords in the KWORDS list.

        Args:
            news_df: pandas.Dataframe, results of `get_news_articles.IngestNews`.

        Returns:
            The document-term frequency matrix for the keywords.
        """
        vec = CountVectorizer(vocabulary=KWORDS,
                              stop_words=None,
                              tokenizer=word_tokenize,
                              ngram_range=(1, 1))
        results_mat = vec.fit_transform(news_df['preproc_text'])

        # sparse to dense matrix
        results_mat = results_mat.toarray()

        # get the feature names from the already-fitted vectorizer
        vec_feature_names = vec.get_feature_names()

        # test that vec's feature names == vocab
        assert vec_feature_names == KWORDS

        # make a table with word frequencies as values and vocab as columns
        out_df = pd.DataFrame(data=results_mat, columns=vec_feature_names)
        # out_df = NewsArticles._remove_duplicate_counts(out_df)
        out_df = NewsArticles._combined_plur_sing_kwords(out_df)

        # append document id and pub date
        # we use the property of CountVectorizer to keep the order of the original texts
        out_df["pub_date"] = NewsArticles._extract_date(
            dates_series=news_df["pub_date"])
        out_df.set_index('pub_date', append=True, inplace=True)
        out_df.rename_axis(["id", "pub_date"], inplace=True)

        return out_df

    @staticmethod
    def _combined_plur_sing_kwords(df: pd.DataFrame) -> pd.DataFrame:
        """
        Combines the plural/singular forms of a keyword (e.g., "behavioural insight"/"behavioural insights")
        and their raw term frequencies into one single keyword/frequency. The singular form is used.

        Args:
            df:     Dataframe whose columns contain keywords' raw term frequencies.

        Returns:
            The same dataset but with the term frequency of plural/singular forms of the same keyword combined.
        """
        data = df.copy()
        for k, vs in CONFIG['SingularPlural'].items():
            if isinstance(vs, str):
                data[k] = data[k] + data[vs]
                data = data.drop(columns=[vs])
            elif isinstance(vs, list):
                for v in vs:
                    data[k] = data[k] + data[v]
                    data = data.drop(columns=[v])
        return data

    @property
    def kword_raw_tf(self):
        return self._kword_raw_tf

    @property
    def kword_yn_occurrence(self):
        """
        Returns whether a keyword occurs in an article (1) or not (0).
        """
        if self._kword_yn_occurrence is None:
            try:
                self._kword_yn_occurrence = self._kword_raw_tf.applymap(
                    lambda cell: 1 if cell > 0 else 0)
            except AttributeError:
                raise AttributeError(
                    "`kword_raw_tf` must be calculated first!")
        return self._kword_yn_occurrence

    @property
    def kword_normlog_tf(self):
        """
        Returns log-normalised frequency from raw frequency of keyword.
        I.e., log-tf = 1 + log10(tf) if tf > 0, 0 otherwise.
        """
        if self._kword_normlog_tf is None:
            try:
                self._kword_normlog_tf = self._kword_raw_tf.applymap(
                    NewsArticles._normalise_tf_log)
            except AttributeError:
                raise AttributeError(
                    "`kword_raw_tf` must be calculated first!")
        return self._kword_normlog_tf

    @staticmethod
    def _normalise_tf_log(raw_count: int) -> float:
        """
        Calculated log frequency as normalised frequency.
        Ref: https://nlp.stanford.edu/IR-book/html/htmledition/sublinear-tf-scaling-1.html
        """
        norm_freq = 1 + log10(raw_count) if raw_count > 0 else 0
        return norm_freq

    @property
    def kword_normlen_tf(self):
        """
        Returns normalised frequency of keywords by dividing a keyword's raw frequency by the lenght
        of the document in which it occurs.
        The length of the document is calculated as number of unigrams.

        All keywords (also those composed by ngrams
        (e.g., 'nudge unit') have been previously turned into unigram (i.e., 'nudge_unit'). This is because we believe
        these keywords can be considered as a single word. That is, both the words ("nudge" and "unit") can have
        independent meaning, however, when they are together, they express a precise, unique concept.
        """
        if self._kword_normlen_tf is None:
            try:
                self._kword_normlen_tf = self._kword_raw_tf.div([
                    NewsArticles.get_num_ngrams(text, 1)
                    for text in self.data.preproc_text
                ],
                                                                axis=0)
            except AttributeError:
                raise AttributeError(
                    "`kword_raw_tf` must be calculated first!")
        return self._kword_normlen_tf

    @property
    def theme_rawfreq(self):
        """
        Returns the theme raw frequencies (i.e., frequencies of the over-arching themes)
        by summing the raw frequencies of the correspoding keywords.
        """
        try:
            self._theme_rawfreq = self._kword_raw_tf.groupby(
                self.__COLS_GROUPBY_DICT, axis=1).sum()
        except AttributeError:
            raise AttributeError("`kword_raw_tf` must be computed first!")
        return self._theme_rawfreq

    @property
    def theme_normlen_f(self):
        """
        Returns the theme len-normalised frequencies (i.e., len-normed frequencies of the over-arching themes)
        by summing the len-normed frequencies of the correspoding keywords.
        """
        try:
            self._theme_normlen_f = self._kword_normlen_tf.groupby(
                self.__COLS_GROUPBY_DICT, axis=1).sum()
        except AttributeError:
            raise AttributeError("`kword_normlen_tf` must be computed first!")
        return self._theme_normlen_f

    @property
    def theme_normlog_f(self):
        """
        Returns the theme log-normalised frequencies (i.e., log-normed frequencies of the over-arching themes)
        by summing the log-normed frequencies of the correspoding keywords.
        """
        try:
            self._theme_normlog_f = self._kword_normlog_tf.groupby(
                self.__COLS_GROUPBY_DICT, axis=1).sum()
        except AttributeError:
            raise AttributeError("`kword_normlog_tf` must be computed first!")
        return self._theme_normlog_f

    @property
    def kword_docfreq(self):
        """
        Calculates document-frequency (df) for each keyword and date. For each keyword, k:
        df_k = [{d in D | k in d}] / |D|
        that is, for each date, the number of documenet that contains k divided by the total number of documents.

        That is, `df` is not calculated with respect to the whole collection of documents, but
        to the collection of documents on a given date. This will allows us to compare `df` trends over time.
        """
        if self._kword_raw_tf is None:
            raise AttributeError("`kword_raw_tf` must be calculated first!")

        if self._kword_docfreq is None:
            # whether a keyword appear in an article yes or no (regardless of how many times)
            kwc_bin = self._kword_raw_tf.applymap(lambda cell: 1
                                                  if cell > 0 else 0)

            self._kword_docfreq = kwc_bin.groupby(
                kwc_bin.index.get_level_values('pub_date').values).sum(
                ) / kwc_bin.groupby(
                    kwc_bin.index.get_level_values('pub_date').values).count()

        return self._kword_docfreq

    @property
    def theme_docfreq(self):
        """
        Calculates document-frequency for the over-arching themes.
        A theme occurrence is defined regardless of which specific keyword(s) occur and how many occurrences.
        """
        if self._kword_raw_tf is None:
            raise AttributeError("`kword_raw_tf` must be calculated first!")

        # raw freq by theme
        kwc_theme = self._kword_raw_tf.groupby(self.__COLS_GROUPBY_DICT,
                                               axis=1).sum()

        # binary frequency (yes/no)
        kwc_theme_bin = kwc_theme.applymap(lambda cell: 1 if cell > 0 else 0)
        self._theme_docfreq = kwc_theme_bin.groupby(
            kwc_theme_bin.index.get_level_values('pub_date').values
        ).sum() / kwc_theme_bin.groupby(
            kwc_theme_bin.index.get_level_values('pub_date').values).count()
        return self._theme_docfreq

    @property
    def unigram_count_perdoc(self):
        if self._unigram_count_perdoc is None:
            self._unigram_count_perdoc = [
                NewsArticles.get_num_ngrams(text, 1)
                for text in self.data.preproc_text
            ]
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

    @staticmethod
    def expand_dict(d: dict) -> dict:
        """
        Explodes the original configuration-file dictionary where themes are keys and the
        correspoding keywords are their list values in list format,
        so that each keyword is a key and the corresponding theme is its value.

        Example:
        {'theme1': ['k1', 'k2', 'k3]} => {'k1':'theme1', 'k2':'theme2', 'k3':'theme3'}

        Args:
            d:  the original configuration-file dictionary

        Returns:
            The exploded dictionary.
        """

        keys = [
            k for k, v in d.items()
            if (isinstance(v, list)) and (k not in NON_KWORD_CONFIG)
        ]
        return {v: k for k in keys for v in d[k]}


def collect_kword_opinioncontext(
        article: str) -> List[List[Union[str, tuple]]]:
    """
    Extracts the opinion context for each keyword' occurrence in a article.

    The opinion context is defined as the sentence where the keyword occurrence and the
    following two sentences.

    Args:
        article:    text of the article as string.

    Returns:
        A list of list. Each sublist contains:
        - first element: the keyword (str) that occurred in the article
        - second element: a tuple of (index i of first sentence where the keyword occurs, the first senetence)
        - third element: a tuple of (index i+1, text) of the next sentence
        - fourth element: a tuple of (index i+2, text) of the next-next sentence

    """

    # add a space after dots or commas that have no space afterwards
    # (e.g., 'I know.But' -> 'I know. But')
    article = re.sub(r'(?<=[.;!?:])(?=[^\s])', r' ', article)

    sentences = tokenise_sent(article.lower())

    # max_idx = len(sentences)
    results = []
    for keyw in KWORDS:
        for idx, sentence in enumerate(sentences):
            if keyw in sentence:
                context1 = (idx, sentence)
                kw_result = [context1]
                try:
                    context2 = (idx + 1, sentences[idx + 1])
                    kw_result.append(context2)
                    context3 = (idx + 2, sentences[idx + 2])
                    kw_result.append(context3)
                except IndexError:
                    pass
                kw_result = [keyw, kw_result]
                results.append(kw_result)

    return remove_duplicates(results)


# TODO: streamline / refactor
def remove_duplicates(
        list_results: List[list]) -> List[Union[str, List[tuple]]]:
    """
    Removes duplicate results.

    Example: cases that are double counted because both "nudges" and "nudge"
    have been identified for the same sentence (because the string "nudges" contains "nudge").

    Args:
        list_results:   List of results. Each result is also a list whose first element is the keyword (str)
                        and second element is a list of tuples; the first element of the first tuple is the index of the sentence
                        where the keyword appears.

    Returns:
        De-duplicated list of results.
    """

    # extract list of keywords
    # and  # extract index of first sentence for each keyword
    kws_idx_first_sent_dict = {
        result[0]: result[1][0][0]
        for result in list_results
    }
    kws_list = [result[0] for result in list_results]

    # keyword list contains both 'nudges' and 'nudge' for same sentences
    if ("nudges" and "nudge"
            in kws_list) and (kws_idx_first_sent_dict.get('nudge')
                              == kws_idx_first_sent_dict.get('nudges')):
        # get rid of "nudge" results as it is a duplicate
        list_results = [
            result for result in list_results if result[0] != "nudge"
        ]

    if ("psychologists" and "psychologist"
            in kws_list) and (kws_idx_first_sent_dict.get('psychologist')
                              == kws_idx_first_sent_dict.get('psychologists')):
        list_results = [
            result for result in list_results if result[0] != "psychologist"
        ]

    if ("coronavirus" and "corona"
            in kws_list) and (kws_idx_first_sent_dict.get('corona')
                              == kws_idx_first_sent_dict.get('coronavirus')):
        list_results = [
            result for result in list_results if result[0] != "corona"
        ]

    if ("covid19" and "covid"
            in kws_list) and (kws_idx_first_sent_dict.get('covid')
                              == kws_idx_first_sent_dict.get('covid19')):
        list_results = [
            result for result in list_results if result[0] != "covid"
        ]

    if ("covid-19" and "covid"
            in kws_list) and (kws_idx_first_sent_dict.get('covid')
                              == kws_idx_first_sent_dict.get('covid-19')):
        list_results = [
            result for result in list_results if result[0] != "covid"
        ]

    return list_results


if __name__ == "__main__":

    uk_news = NewsArticles(country="uk")
    print(uk_news.data.shape)

    # calculate all the metrics
    uk_news.kword_raw_tf
    uk_news.theme_rawfreq
    uk_news.kword_normlen_tf
    uk_news.theme_normlen_f
    uk_news.kword_normlog_tf
    uk_news.theme_normlog_f
    uk_news.theme_docfreq
    uk_news.kword_docfreq
    uk_news.unigram_count_perdoc

    # save them as csv so that they can be load in in R
    uk_news.kword_raw_tf.to_csv(os.path.join(DIR_DATA_INT, "kword_raw_tf.csv"))
    uk_news.theme_rawfreq.to_csv(
        os.path.join(DIR_DATA_INT, "theme_rawfreq.csv"))
    uk_news.kword_normlen_tf.to_csv(
        os.path.join(DIR_DATA_INT, "kword_normlen_tf.csv"))
    uk_news.theme_normlen_f.to_csv(
        os.path.join(DIR_DATA_INT, "theme_normlen_f.csv"))
    uk_news.theme_docfreq.to_csv(
        os.path.join(DIR_DATA_INT, "theme_docfreq.csv"))
    uk_news.kword_docfreq.to_csv(
        os.path.join(DIR_DATA_INT, "kword_docfreq.csv"))
    pd.DataFrame(uk_news.unigram_count_perdoc).to_csv(
        os.path.join(DIR_DATA_INT, "unigram_count_perdoc.csv"))
