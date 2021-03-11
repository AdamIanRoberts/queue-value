from dataclasses import dataclass
from datetime import datetime, timedelta
from tqdm import tqdm

import os

import numpy as np
import pandas as pd


@dataclass
class FileDetails:
    # File name convention: TICKER_YYYY-MM-DD_xxxxxxxx_xxxxxxxx_TYPE_LEVELS
    file_name: str

    def __post_init__(self):
        self.ticker, date, _, _, self.type, levels = self.file_name.split("_")
        self.date = datetime.strptime(date, "%Y-%m-%d")
        levels, _ = levels.split(".")
        self.levels = int(levels)


class LobsterOrderbookProcessor:
    def __init__(self, ticker: str, directory: str) -> None:
        self.directory = os.path.join(directory, ticker)
        self.files = [FileDetails(file_name=name) for name in os.listdir(self.directory) if name.startswith(ticker)]
        assert len(np.unique([file.ticker for file in self.files])) == 1, "Inconsistent stock name in files."
        self.ticker = self.files[0].ticker
        assert len(np.unique([file.levels for file in self.files])) == 1, "Inconsistent orderbook levels in files."
        self.levels = self.files[0].levels
        self.message_files = {file.date: file for file in self.files if file.type == "message"}
        self.book_view_files = {file.date: file for file in self.files if file.type == "orderbook"}
        assert set(self.message_files.keys()) == set(self.book_view_files.keys()), "Inconsistent dates in files."
        self.dates = list(self.message_files.keys())
        self.start_time = 9.5 * 60 * 60  # 09:30:00 in sec after midnight
        self.end_time = 16 * 60 * 60  # 16:30:00 in sec after midnight

    def save_processed_orderbook(self, directory: str) -> None:
        if not os.path.exists(os.path.join(directory, self.ticker)):
            os.makedirs(os.path.join(directory, self.ticker))
        for date in tqdm(self.dates):
            orderbook = self._build_orderbook(date)
            processed_file_name = f"{self.ticker}_{date.date()}_orderbook_{self.levels}.csv"
            orderbook.to_csv(os.path.join(directory, self.ticker, processed_file_name))

    def _build_orderbook(self, date: datetime) -> pd.DataFrame:
        messages = self._build_messages(date)
        book_view = self._build_book_view(date)
        orderbook = pd.merge(messages, book_view, on=["index"])
        orderbook = orderbook[orderbook["event_type"] != 7]
        orderbook = orderbook[orderbook["timestamp"].between(self.start_time, self.end_time)]
        orderbook = orderbook.drop(columns=["timestamp", "broker"])
        return orderbook

    def _build_messages(self, date: datetime) -> pd.DataFrame:
        # Message file information:
        # ----------------------------------------------------------
        #
        # Dimension: (NumberEvents x 6)
        #
        # Structure: Each row:
        # Timestamp (sec after midnight), Event type, Order ID, Size (number of shares), Price, Direction
        #
        # Event types:
        # - '1': Submission new limit order
        # - '2': Cancellation(partial)
        # - '3': Deletion(total order)
        # - '4': Execution of a visible limit order
        # - '5': Execution of a hidden limit order
        # - '7': Trading Halt(Detailed information below)
        #
        # Direction:
        # - '-1': Sell limit order
        # - '1': Buy limit order
        # NOTE: Execution of a sell (buy) limit order corresponds to a buyer - (seller -) initiated trade, i.e. a
        #       BUY (SELL) trade.
        #
        # ----------------------------------------------------------
        messages = self._message_file_to_df(self.message_files[date])
        return messages

    def _message_file_to_df(self, file: FileDetails) -> pd.DataFrame:
        column_names = [
            "timestamp",
            "event_type",
            "order_id",
            "size",
            "price",
            "direction",
            "broker",
        ]
        column_dtypes = [float, int, int, int, int, int, str]
        dtype_mapping = dict(zip(column_names, column_dtypes))
        message_df = pd.read_csv(os.path.join(self.directory, file.file_name), names=column_names, dtype=dtype_mapping)
        message_df = message_df[message_df["timestamp"].between(self.start_time, self.end_time)]
        message_df["datetime"] = [file.date + timedelta(seconds=s) for s in message_df["timestamp"]]
        message_df.index.name = "index"
        return message_df

    def _build_book_view(self, date: datetime) -> pd.DataFrame:
        # Orderbook file information:
        # ----------------------------------------------------------
        #
        # - Dimension: (NumberEvents x (NumberLevels * 4))
        #
        # - Structure: Each row:
        # Ask price 1, Ask volume 1, Bid price 1, Bid volume 1, Ask price 2, Ask volume 2, Bid price 2, ...
        #
        # - Note: Unoccupied bid (ask) price levels are set to -9999999999 (9999999999) with volume 0
        #
        # ----------------------------------------------------------
        book_view = self._book_view_file_to_df(self.book_view_files[date])
        return book_view

    def _book_view_file_to_df(self, file: FileDetails) -> pd.DataFrame:
        column_prefixes = ["ask_price_", "ask_size_", "bid_price_", "bid_size_"]
        column_names = [prefix + str(l) for l in range(file.levels) for prefix in column_prefixes]
        column_dtypes = np.repeat(int, len(column_names))
        dtype_mapping = dict(zip(column_names, column_dtypes))
        book_view_df = pd.read_csv(
            os.path.join(self.directory, file.file_name), names=column_names, dtype=dtype_mapping
        )
        book_view_df.index.name = "index"
        return book_view_df
