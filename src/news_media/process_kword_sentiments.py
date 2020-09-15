""" TO ADD """

import os
import pickle
import pandas as pd
from typing import List
from datetime import datetime
from src.utils import load_config_yaml
from src.news_media.get_keywords_trend import expand_dict

# Input data
DIR_DATA = os.environ.get("DIR_DATA_INTERIM")
FILENAME = "batch1_kword_sent.csv"

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

    # check coded keywords that are no longer part of list
    # [kw for kw in data.kword.unique() if kw not in KWORDS+OTHER_IMPORTANT_WORDS]
    # ok: voodoo, bogus (that we had later excluded)

    # exlcude three cases of kwords no longer in list
    data = data[[
        kw in KWORDS + ['behavioural_fatigue', 'herd_immunity']
        for kw in data['kword']
    ]].copy()

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

    return data


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


if __name__ == "__main__":

    BATCH1_FILENAME = "batch1_kword_sent.csv"
    BATCH1_OUTPUT = "batch1_preproc_kword_sent.pickle"
    batch1_data = preproc_step(batch=BATCH1_FILENAME)
    with open(os.path.join(DIR_DATA, BATCH1_OUTPUT), "wb") as output_file:
        pickle.dump(batch1_data, output_file)
