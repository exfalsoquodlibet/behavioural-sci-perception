""" This script ingest the keyword-opinion pairs coded for sentiment by the researchers. """

import os
import pandas as pd
import numpy as np
from typing import Dict
from src.utils import load_config_yaml

# Load configuration file
CONFIG_FILE_NAME = "config_ingest_kword_sentiments.yaml"
DIR_EXT = os.environ.get("DIR_EXT")
CONFIG_FILE = os.path.join(DIR_EXT, CONFIG_FILE_NAME)

CONFIG = load_config_yaml(CONFIG_FILE)

# Input data dir
DIR_DATA_RAW = os.environ.get("DIR_DATA_RAW")
# Specify output filepath
OUTPUT_PATH = os.environ.get("DIR_DATA_INTERIM")


def ingest_data(batch, batch_suffix):

    # get data from all the excel tabs
    data_dict = read_data_xl(os.path.join(DIR_DATA_RAW, batch['FileName']))

    # verify that all tabs have been read in
    if not all([
            expected_sheet in data_dict.keys()
            for expected_sheet in batch['SheetNames']
    ]):
        raise Exception(
            f"Expected sheets are missing from {batch['FileName']}")

    # transform dictionary into a big dataset
    expected_cols = batch['ExpectedCols']

    data_df = pd.DataFrame(columns=expected_cols)
    for tab_name, df in data_dict.items():
        # validate columns
        if not all([col in expected_cols for col in df.columns]):
            raise Exception(f"Missing required columns in {tab_name}")
        data_df = data_df.append(df)

    # remove kword-sentences that were not coded for sentiments
    # because they were not relevant keywords
    data_df = data_df[~np.isnan(data_df['keyword_sentiment'])].copy()

    # add batch number as suffix to article_id
    data_df['article_id'] = [
        batch_suffix + str(id) for id in data_df['article_id']
    ]

    return data_df


def read_data_xl(filename: str) -> Dict[str, pd.DataFrame]:
    """Reads the specified sheet from a xls, xlsx, xlsm, xlsb, or odf file.
    Please not that the function does not accept sheetname as all sheets are read in.

    Args:
        filename:   path to the file

    Returns:
        A dictionary of sheet names (keys) and pandas.DataFrames (values).
    """
    try:
        sheet_df = pd.read_excel(filename, sheet_name=None)
        return sheet_df
    except Exception as e:
        raise type(e)("Could not open file:", filename)


def join_dfs(df1: pd.DataFrame,
             df2: pd.DataFrame,
             duplicates_id=CONFIG['DuplicatesId']) -> pd.DataFrame:
    """Join two batches and remove (manually identified) duplicate articles"""

    df = df1.append(df2)
    assert "article_id" in df.columns
    df = df[~df.article_id.isin(duplicates_id)].copy()

    return df


if __name__ == "__main__":

    BATCH1 = CONFIG['FileBatch1']
    BATCH2 = CONFIG['FileBatch2']

    df_batch1 = ingest_data(batch=BATCH1, batch_suffix='b1_')
    df_batch2 = ingest_data(batch=BATCH2, batch_suffix='b2_')
    print(f"batch1 size: {df_batch1.shape}")
    print(f"batch1 N articles: {df_batch1.article_id.nunique()}")
    print(f"batch2 size: {df_batch2.shape}")
    print(f"batch2 N articles: {df_batch2.article_id.nunique()}")
    df = join_dfs(df_batch1, df_batch2)
    print(f"Joined deduplicates batches size: {df.shape}")
    print(f"Joined deduplicated batches N articles: {df.article_id.nunique()}")

    df.to_csv(os.path.join(OUTPUT_PATH, CONFIG['Outputname']))
