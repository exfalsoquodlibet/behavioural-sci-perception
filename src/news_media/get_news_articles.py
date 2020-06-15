"""Ingest and validate news articles data.

This script allows the user to load and do an initial validation of the news articles,
made available as a .xlsx file.

The name of the files are specified in `config_news_data.yaml` in the `ext/` folder,
under 'NewsFileUK'.'FileName' and 'NewsFileUS'.'Filename'.

It is assumed that the file contains the sheet and columns specified in the
`config_make_data.yaml` under 'SheetName' and 'Fields'.

This file can also be imported as a module and contains the following
classes and functions:

    * IngestNews - the main class of the script
    * read_data_xl - reads a .xlsm or .xlsx file

To run the script, from the root directory:
$ python -m src.news_media.get_news_articles

"""

import os
import inspect
import pandas as pd
from src.utils import load_config_yaml

# Load configuration file
CONFIG_FILE_NAME = "config_news_data.yaml"
DIR_EXT = os.environ.get("DIR_EXT")
CONFIG_FILE = os.path.join(DIR_EXT, CONFIG_FILE_NAME)

CONFIG = load_config_yaml(CONFIG_FILE)

# Specify output filepath
OUTPUT_PATH = os.environ.get("DIR_DATA_INTERIM")


class IngestNews:
    """
    Class to ingest news articles data.

    This class accepts .xlsm/.xlsx files as input format.
    The class uses a series of methods that are defined outside of the class.
    The name of the sheet (sheetname) must be provided on
    top of the name of the workbook (filename).

    ...

    Attributes:
        filename (str):
            the path to the file containing the data
        data (pandas.DataFrame):
            the data extracted from the provided file

    Methods:
        save_data_to_csv(self, output_folder)
            Saves the data to the provided folder path
        _check_file_exists_and_is_not_empty(filename)
            (staticmethod) Checks that filename exists and is not empty
        _validate_columns(df: pd.DataFrame)
            (staticmethod) Checks that the required columns (listed under `Fields` in .yaml) are in the data
        select_columns(self)
            Filters the dataset to only keep the columns specified in 'FieldsToKeep' in the .yaml file
        rename_columns(self)
            Renames the dataset columns as specified in `FieldsNewNames` in .yaml
    """
    def __init__(self, filename: str, sheetname: str = None):
        """
        Args:

        filename (str):         path to the file
        sheetname (str):        (optional) the name of the excel sheet to be read (required if excel file)
        """

        # Apply file validation steps
        print("Checking that file exists and is not empty...")
        IngestNews._check_file_exists_and_is_not_empty(filename)

        # Extract extension of provided file
        extension = os.path.splitext(filename)[1]

        # If extension is one in extension_map, use the corresponding reader,
        # else, throw ERROR
        try:
            data_reader = extension_map.get(extension)
        except KeyError as e:
            raise type(e)("Extension must be a xls, xlsx, xlsm, xlsb, or odf")

        print("Reading the news data in...")
        if not sheetname:
            print(
                "No sheetname was provided; defaulting to read the first sheet in the",
                extension, "file")

        # Assign attributes
        num_args = len(inspect.signature(data_reader).parameters)
        args = [filename, sheetname]
        self.data = data_reader(*args[:num_args])
        self.filename = filename
        print("Data read successfully.")

        # Validate columns
        print("Validating data columns...")
        IngestNews._validate_columns(self.data)

    def save_data_to_csv(self, output_name) -> None:
        """Saves the data as a CSV file in the 'data/intermediate/` folder.
        The output name should be specified in the .yaml file under "SaveAs".

        Args:
            output_name: name of the CSV file.

        Returns:
            None.
        """
        try:
            self.data.to_csv(OUTPUT_PATH + '/' + output_name)
        except Exception as ex:
            raise ex
        else:
            print(
                f"Data saved successfully as {output_name} to 'data/intermediate/`"
            )
            return None

    @staticmethod
    def _check_file_exists_and_is_not_empty(filename: str) -> None:
        """Checks that `filename` exists and is not empty."""
        try:
            if os.path.getsize(filename) > 0:
                print("All good -", filename, "is not empty")
            else:
                print(filename, "is empty")
        except OSError as e:
            raise type(e)(filename, "does not exists or is non accessible.")

    @staticmethod
    def _validate_columns(df: pd.DataFrame) -> None:
        """"Checks that the required columns are in the file.
        Args:
            df (pd.DataFrame):      dataframe whose columns need to be validated

        Returns:
            A message that all the required columns are present, a `ValueError` is raised if not.
        """
        try:
            if not set(CONFIG['Fields']).issubset(set(df.columns)):
                raise ValueError(
                    f"All columns {CONFIG['Fields']} are required")
            else:
                print("Good - All required columns are present.")
        except UnicodeDecodeError:
            raise UnicodeDecodeError("Cannot parse data")

    def select_columns(self) -> None:
        """"Subsets the dataframe by only keeping the columns specified in CONFIG['FieldsToKeep'].

        Returns:
            A message that the dataset has been successfully subsetted.
        """
        self.data = self.data[CONFIG['FieldsToKeep']].copy()
        print("Dataset successfully filtered.")
        return self

    def rename_columns(self) -> None:
        """"Renames the dataset's columns as specified in CONFIG['FieldsNewNames'].

        Returns:
            A message that the dataset's columns have been successfully re-named.
        """
        self.data.columns = CONFIG['FieldsNewNames']
        return self


def read_data_xl(filename: str, sheetname: str) -> pd.DataFrame:
    """Reads the specified sheet from a xls, xlsx, xlsm, xlsb, or odf file.

    Args:
        filename:   path to the file
        sheetname:  name of the sheet (excel tab)

    Returns:
        A pandas.DataFrame
    """
    try:
        sheet_df = pd.read_excel(filename, sheet_name=sheetname)
        return sheet_df
    except Exception as e:
        raise type(e)("Could not open file:", filename)


extension_map = {
    '.xlsm': read_data_xl,
    '.xlsx': read_data_xl,
    '.xls': read_data_xl,
    '.xlsb': read_data_xl,
    '.odf': read_data_xl
}

if __name__ == "__main__":

    print("Reading in UK news data...")
    uk_news = IngestNews(filename=os.path.join(
        os.environ.get("DIR_DATA_RAW"), CONFIG['NewsFileUK']['FileName']),
                         sheetname=CONFIG['NewsFileUK']['SheetName'])

    # Filter columns
    print("Filtering data columns...")
    uk_news.select_columns()

    # Rename columns
    print("Renaming data columns...")
    uk_news.rename_columns()

    uk_news.save_data_to_csv(output_name=CONFIG['NewsFileUK']['SaveAs'])
