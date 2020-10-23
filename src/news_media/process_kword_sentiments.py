"""
This script preprocess the keyword-opinion pairs coded for sentiment by the researchers:

Preprocessing steps:
- aggregate sub-keywords into keywords
- exclude sub-keywords no longer used
- cast date as datetime format
- remove cases whose publication date is before 2020-01-27 (Monday)
"""

import os
import pickle
import pandas as pd
from typing import List
from datetime import datetime
from src.utils import load_config_yaml
from src.news_media.get_keywords_trend import expand_dict

# Input data
DIR_EXT = os.environ.get("DIR_EXT")
CONFIG_DATA_FILE = os.path.join(DIR_EXT, "config_ingest_kword_sentiments.yaml")
CONFIG_DATA = load_config_yaml(CONFIG_DATA_FILE)
FILENAME = CONFIG_DATA['Outputname']
DIR_DATA = os.environ.get("DIR_DATA_INTERIM")

# Configuration files for keywords grouping
DIR_EXT = os.environ.get("DIR_EXT")
CONFIG_FILE_NAME = "keywords.yaml"
SUBWK_TO_KW_MAP = "subkw_to_kw_map.yaml"

# Load configuration file
CONFIG_FILE = os.path.join(DIR_EXT, CONFIG_FILE_NAME)
CONFIG = load_config_yaml(CONFIG_FILE)

KWORDS = CONFIG['Actors'] + CONFIG['BehavSci'] + CONFIG['Behav_ins'] + CONFIG[
    'Behav_chan'] + CONFIG['Behav_pol'] + CONFIG['Behav_anal'] + CONFIG[
        'Psych'] + CONFIG['Econ_behav'] + CONFIG['Econ_irrational'] + CONFIG[
            'Nudge'] + CONFIG['Nudge_choice'] + CONFIG['Nudge_pater']
OTHER_IMPORTANT_WORDS = CONFIG['Covid'] + CONFIG['Fatigue'] + CONFIG['Immunity']

# load lookup sub-keywords to keywords
SUBKWORDS_TO_KWORDS = expand_dict(
    load_config_yaml(os.path.join(DIR_EXT, SUBWK_TO_KW_MAP)))


def preproc_step(batch=FILENAME):

    # Read data
    data = pd.read_csv(os.path.join(DIR_DATA, batch))
    print(f"size of data before removing unnecessary kwords: {data.shape}")
    print(
        f"N articles before removing unnecessary kwords: {data.article_id.nunique()}"
    )

    # check coded keywords that are no longer part of list
    # [kw for kw in data.kword.unique() if kw not in KWORDS+OTHER_IMPORTANT_WORDS]
    # ok: voodoo, bogus (that we had later excluded)

    # exlcude three cases of kwords no longer in list
    data = data[[
        kw in KWORDS + ['behavioural_fatigue', 'herd_immunity']
        for kw in data['kword']
    ]].copy()

    print(f"size of data after removing unnecessary kwords: {data.shape}")
    print(
        f"N articles after removing unnecessary kwords: {data.article_id.nunique()}"
    )

    # group keywords
    # rename current 'kword' as 'subkword'
    data.rename(columns={'kword': 'subkword'}, inplace=True)

    # [kw for kw in data['subkword'] if kw not in SUBKWORDS_TO_KWORDS.keys()]
    # herd_immunity and behavioural_fatigue are no longer in kword

    # create new "kword"
    data['kword'] = [
        SUBKWORDS_TO_KWORDS.get(kw, kw) for kw in data['subkword']
    ]

    data['pub_date_dt'] = _extract_date(data['pub_date'])

    print(
        f"size of data before removing articles before '2020-01-27': {data.shape}"
    )
    print(
        f"N articles before removing articles before '2020-01-27': {data.article_id.nunique()}"
    )
    data = data[data.pub_date_dt >= '2020-01-27'].copy()
    print(
        f"size of data after removing articles before '2020-01-27': {data.shape}"
    )
    print(
        f"N articles after removing articles before '2020-01-27': {data.article_id.nunique()}"
    )

    return data


def _extract_date(dates_series: pd.Series) -> List[datetime]:
    """
    Encode the dates contained in a pandas.Series in date format.

    Args:
        dates_series: pandas.Series containing dates, results of `get_news_articles.IngestNews`.

    Returns:
        List of dates in date format.
    """
    date_formats = {
        2: r"%Y-%m-%d %H:%M:%S",
        3: r"%B %d, %Y",
        4: r"%B %d, %Y %A"
    }

    def _get_date_format(date: str):
        return date_formats[len(date.split())]

    list_dates = [
        datetime.strptime(date, _get_date_format(date))
        for date in dates_series
    ]

    return list_dates


if __name__ == "__main__":
    OUTPUT = "preproc_kword_sent.pickle"
    data = preproc_step(batch=FILENAME)
    with open(os.path.join(DIR_DATA, OUTPUT), "wb") as output_file:
        pickle.dump(data, output_file)
