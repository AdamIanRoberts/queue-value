from unittest import TestCase

import os
import tempfile

from data_processing.lobster.LobsterOrderbookProcessor import LobsterOrderbookProcessor


class TestLobsterOrderbookProcessor(TestCase):
    def test___init__(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            os.mkdir(tmpdirname + "/CSCO")
            open(os.path.join(tmpdirname, "CSCO/CSCO_2020-01-02_x_x_orderbook_3.csv"), "a").close()
            open(os.path.join(tmpdirname, "CSCO/CSCO_2020-01-02_x_x_message_3.csv"), "a").close()
            processor = LobsterOrderbookProcessor(ticker="CSCO", directory=tmpdirname)
            self.assertEqual(processor.directory, tmpdirname + "/CSCO")
            self.assertEqual(processor.ticker, "CSCO")
            self.assertEqual(processor.levels, 3)
