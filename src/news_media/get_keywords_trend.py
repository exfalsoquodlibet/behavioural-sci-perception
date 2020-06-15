"""
# lower case
# sentence tokenise
# work tokenise
# COUNT: extract number of occurrences of <KEY-WORD> in each article [(key_word, count)]
# BINARY: extract whether <KEY-WORD> occurred in article or not [(key_word, 1 or 0)]
# encode date as date
# plot trend over time

"""

import os
import pandas as pd
from datetime import datetime
from sklearn.feature_extraction.text import CountVectorizer

from src.utils import load_config_yaml

# Constants
UK_FILENAME = "uk_news.csv"
USA_FILENAME = "us_news.csv"
CONFIG_FILE_NAME = "keywords.yaml"
DIR_DATA_INT = os.environ.get("DIR_DATA_INTERIM")
DIR_EXT = os.environ.get("DIR_EXT")

# Load configuration file
CONFIG_FILE = os.path.join(DIR_EXT, CONFIG_FILE_NAME)
CONFIG = load_config_yaml(CONFIG_FILE)

BIGRAM_KEYWORDS = CONFIG['Bigrams']
UNIGRAM_KEYWORDS = CONFIG['Unigrams']


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
        VOCAB = BIGRAM_KEYWORDS + UNIGRAM_KEYWORDS
        vec = CountVectorizer(vocabulary=VOCAB,
                              stop_words=None,
                              ngram_range=(1, 2))
        results_mat = vec.fit_transform(self.data.full_text)

        # sparse to dense matrix
        results_mat = results_mat.toarray()

        # get the feature names from the already-fitted vectorizer
        vec_feature_names = vec.get_feature_names()

        # test that vec's feature names == vocab
        assert vec_feature_names == VOCAB

        # make a table with word frequencies as values and vocab as columns
        out_df = pd.DataFrame(data=results_mat, columns=vec_feature_names)

        self.keywords_count = out_df

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


if __name__ == "__main__":

    uk_news = NewsArticles(country="uk")
    print(uk_news.data.shape)

    uk_news.count_keywords()
    print(uk_news.keywords_count.shape)
    print(uk_news.keywords_count.columns)
    print(uk_news.keywords_count.head(2))

    uk_news.extract_date()
    print(uk_news.date[:10])

    uk_news.keywords_count['date'] = uk_news.date

    # calculate in how many articles each keyword appeared on a given date
    counts_by_date = uk_news.keywords_count.groupby('date').apply(
        lambda x: x[x > 0].count()).reset_index()
    # count number of articles per date
    n_articles_per_date = uk_news.keywords_count.groupby('date').size()
    counts_by_date['n_articles'] = n_articles_per_date.to_list()
    cols_order_1 = ['date', 'n_articles'] + BIGRAM_KEYWORDS + UNIGRAM_KEYWORDS
    counts_by_date = counts_by_date[cols_order_1]

    uk_news.keywords_count['title'] = uk_news.data.title
    cols_order = ['date', 'title'] + BIGRAM_KEYWORDS + UNIGRAM_KEYWORDS
    uk_news.keywords_count = uk_news.keywords_count[cols_order]

    # export results as CSV
    uk_news.keywords_count.to_csv(
        os.path.join(DIR_DATA_INT, "keywords_count1.csv"))
    counts_by_date.to_csv(
        os.path.join(DIR_DATA_INT, "articles_with_keyword_count_by_date.csv"))
