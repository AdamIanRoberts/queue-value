"""
Script to process order book data obtained from lobsterdata.com

To process the files, the user must save raw lobster .csv files in <project-directory>/data/raw/TICKER/
with naming convention:
TICKER_YYYY-MM-DD_xxxxxxxx_xxxxxxxx_TYPE_LEVELS

Files will be processed and saved to <project-directory>/data/processed/TICKEROo with naming convention:
TICKER_YYYY-MM-DD_orderbook_LEVELS
"""

import os

from data_processing.lobster.LobsterOrderbookProcessor import LobsterOrderbookProcessor
from definitions import ROOT_DIR

TICKER = "CSCO"
RAW_DATA_DIR = os.path.join(ROOT_DIR, "data/raw")
PROCESSED_DATA_DIR = os.path.join(ROOT_DIR, "data/processed")


def process_lobster_data(ticker: str, dir_from: str, dir_to: str) -> None:
    lob_builder = LobsterOrderbookProcessor(ticker, dir_from)
    lob_builder.save_processed_orderbook(dir_to)


def main():
    process_lobster_data(TICKER, RAW_DATA_DIR, PROCESSED_DATA_DIR)


if __name__ == "__main__":
    main()
    print("Process terminated successfully.")
