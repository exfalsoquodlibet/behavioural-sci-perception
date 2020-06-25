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

from src.utils import load_config_yaml
from src.preproc_text import tokenise_sent

# Constants
UK_FILENAME = "uk_news.csv"
USA_FILENAME = "us_news.csv"
CONFIG_FILE_NAME = "keywords.yaml"
DIR_DATA_INT = os.environ.get("DIR_DATA_INTERIM")
DIR_EXT = os.environ.get("DIR_EXT")

# Load configuration file
CONFIG_FILE = os.path.join(DIR_EXT, CONFIG_FILE_NAME)
CONFIG = load_config_yaml(CONFIG_FILE)

print(type(CONFIG))

VOCAB = CONFIG['Actors'] + CONFIG['BehavSci'] + CONFIG['Nudge'] + CONFIG[
    'Positive'] + CONFIG['Negative'] + CONFIG['Covid'] + CONFIG[
        'Fatigue'] + CONFIG['Immunity']


class NewsArticles:
    """
    """
    def __init__(self, country: str = "uk"):

        if country == "usa":
            csv_filepath = os.path.join(DIR_DATA_INT, USA_FILENAME)
        else:
            csv_filepath = os.path.join(DIR_DATA_INT, UK_FILENAME)
        df = pd.read_csv(csv_filepath)

        # Remove rows that mark start of batches
        bool_series = df["title"].str.startswith("Title (", na=False)
        df = df[~bool_series].copy()

        self.data = df
        self.country = country

    def count_keywords(self):
        """"""
        vec = CountVectorizer(vocabulary=VOCAB,
                              stop_words=None,
                              ngram_range=(1, CONFIG['NgramRange']))
        results_mat = vec.fit_transform(self.data.full_text)

        # sparse to dense matrix
        results_mat = results_mat.toarray()

        # get the feature names from the already-fitted vectorizer
        vec_feature_names = vec.get_feature_names()

        # test that vec's feature names == vocab
        assert vec_feature_names == VOCAB

        # make a table with word frequencies as values and vocab as columns
        out_df = pd.DataFrame(data=results_mat, columns=vec_feature_names)
        out_df = NewsArticles._remove_duplicate_counts(out_df)

        self.keywords_count = out_df

    @staticmethod
    def _remove_duplicate_counts(df: pd.DataFrame) -> pd.DataFrame:
        """
        'behavioural insights team' contains 'behavioural insights' meaning that
        when 'behavioural insights team' occurs in the text, it is double-counted
        (i.e., 1 x 'behavioural insights team' and 1 x 'behavioural insights').

        This function removes its count from  'behavioural insights'.

        Same for the other cases.
        """
        df['behavioural insights'] = df['behavioural insights'] - df[
            'behavioural insights team']
        df['nudge'] = df['nudge'] - df['nudge unit'] - df['nudge theory']
        return df

    def extract_date(self):
        """
        """
        date_formats = {3: r"%B %d, %Y", 4: r"%B %d, %Y %A"}

        def _get_date_format(date: str):
            return date_formats[len(date.split())]

        list_dates = [
            datetime.strptime(date, _get_date_format(date))
            for date in self.data.pub_date
        ]

        self.data['date'] = list_dates
        self.date = list_dates

    def get_keywords_timetrend(self):
        self.keywords_count['date'] = self.date
        self.keywords_count.groupby('date').apply(
            lambda x: (x > 0).sum()).reset_index(name='count')


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
    for keyw in VOCAB:
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

    Example: cases that are double counted because both "nudge theory" and "nudge"
    have been identified for the same sentence.

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

    # keyword list contains both 'nudge theory' and 'nudge' for same sentences
    if ("nudge theory" and "nudge"
            in kws_list) and (kws_idx_first_sent_dict.get('nudge')
                              == kws_idx_first_sent_dict.get('nudge theory')):
        # get rid of "nudge" results as it is a duplicate
        list_results = [
            result for result in list_results if result[0] != "nudge"
        ]

    # keyword list contains both 'nudge uniit' and 'nudge' for same sentences
    if ("nudge unit" and "nudge"
            in kws_list) and (kws_idx_first_sent_dict.get('nudge')
                              == kws_idx_first_sent_dict.get('nudge unit')):
        # get rid of "nudge" results as it is a duplicate
        list_results = [
            result for result in list_results if result[0] != "nudge"
        ]

    # keyword list contains both 'nudges' and 'nudge' for same sentences
    if ("nudges" and "nudge"
            in kws_list) and (kws_idx_first_sent_dict.get('nudge')
                              == kws_idx_first_sent_dict.get('nudges')):
        # get rid of "nudge" results as it is a duplicate
        list_results = [
            result for result in list_results if result[0] != "nudge"
        ]

    # keyword list contains both 'choice architecture' and 'choice architect' for same sentences
    if ("choice architecture" and "choice architect" in kws_list) and (
            kws_idx_first_sent_dict.get('choice architect')
            == kws_idx_first_sent_dict.get('choice architecture')):
        # get rid of "choice architect" results as it is a duplicate
        list_results = [
            result for result in list_results
            if result[0] != "choice architect"
        ]

    if ("behavioural insights" and "behavioural insight" in kws_list) and (
            kws_idx_first_sent_dict.get('behavioural insight')
            == kws_idx_first_sent_dict.get('behavioural insights')):
        list_results = [
            result for result in list_results
            if result[0] != "behavioural insight"
        ]

    #
    if ("behavioural insights team" and "behavioural insights" in kws_list
        ) and (kws_idx_first_sent_dict.get('behavioural insights')
               == kws_idx_first_sent_dict.get('behavioural insights team')):
        list_results = [
            result for result in list_results
            if result[0] != "behavioural insights"
        ]

    if ("behavioural insights team" and "behavioural insight" in kws_list
        ) and (kws_idx_first_sent_dict.get('behavioural insight')
               == kws_idx_first_sent_dict.get('behavioural insights team')):
        list_results = [
            result for result in list_results
            if result[0] != "behavioural insight"
        ]

    #
    if ("behavioural scientists" and "behavioural scientist" in kws_list) and (
            kws_idx_first_sent_dict.get('behavioural scientist')
            == kws_idx_first_sent_dict.get('behavioural scientists')):
        list_results = [
            result for result in list_results
            if result[0] != "behavioural scientist"
        ]

    if ("behavioural sciences" and "behavioural science" in kws_list) and (
            kws_idx_first_sent_dict.get('behavioural science')
            == kws_idx_first_sent_dict.get('behavioural sciences')):
        list_results = [
            result for result in list_results
            if result[0] != "behavioural science"
        ]

    if ("behavioural economists" and "behavioural economist" in kws_list) and (
            kws_idx_first_sent_dict.get('behavioural economist')
            == kws_idx_first_sent_dict.get('behavioural economists')):
        list_results = [
            result for result in list_results
            if result[0] != "behavioural economist"
        ]

    if ("behavioural analysts" and "behavioural analyst" in kws_list) and (
            kws_idx_first_sent_dict.get('behavioural analyst')
            == kws_idx_first_sent_dict.get('behavioural analysts')):
        list_results = [
            result for result in list_results
            if result[0] != "behavioural analyst"
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

    # 1. Count of all occurrences per article/date
    uk_news.count_keywords()
    print(uk_news.keywords_count.shape)
    print(uk_news.keywords_count.columns)
    print(uk_news.keywords_count.head(2))

    # Get date in date format
    uk_news.extract_date()
    print(uk_news.date[:10])

    # TODO: add the following as class methods

    uk_news.keywords_count['date'] = uk_news.date
    # count number of articles per date
    n_articles_per_date = uk_news.keywords_count.groupby('date').size()
    n_articles_per_date = n_articles_per_date.reset_index()
    n_articles_per_date.rename(columns={0: 'n_articles'}, inplace=True)

    # a) add keyword higher-level grouping
    long_counts_by_date = pd.melt(uk_news.keywords_count,
                                  id_vars=['date', 'title'],
                                  value_vars=VOCAB,
                                  var_name='keyword',
                                  value_name='count_kw_occurrences')
    keywords_groups = CONFIG.copy()
    keywords_groups.pop('NgramRange')
    list_kw_groups = [
        group for group, list_kws_ in keywords_groups.items()
        for kw in long_counts_by_date.keyword if kw in list_kws_
    ]
    long_counts_by_date['keyword_group'] = list_kw_groups
    # attach n of articles per date
    long_counts_by_date = long_counts_by_date.merge(n_articles_per_date,
                                                    how='left',
                                                    on='date')
    # b) sum together by keywords-groupings
    counts_by_kwgroups_date = long_counts_by_date.groupby(
        ['keyword_group', 'date',
         'n_articles']).agg({'count_kw_occurrences': 'sum'})

    # 2. Calculate in how many articles each keyword appeared on a given date
    counts_by_date = uk_news.keywords_count.groupby('date').apply(
        lambda x: x[x > 0].count()).reset_index()

    counts_by_date['n_articles'] = n_articles_per_date.to_list()
    cols_order_1 = ['date', 'n_articles'] + VOCAB
    counts_by_date = counts_by_date[cols_order_1]

    uk_news.keywords_count['title'] = uk_news.data.title
    cols_order = ['date', 'title'] + VOCAB
    uk_news.keywords_count = uk_news.keywords_count[cols_order]

    # export results as CSV
    uk_news.keywords_count.to_csv(
        os.path.join(DIR_DATA_INT, "keywords_count1.csv"))
    counts_by_date.to_csv(
        os.path.join(DIR_DATA_INT, "articles_with_keyword_count_by_date.csv"))
    counts_by_kwgroups_date.to_csv(
        os.path.join(DIR_DATA_INT, "kwgroups_count_by_date.csv"))
