import ccxt
import numpy as np
import pandas as pd
from pandas import DataFrame
from ta import add_all_ta_features
from ta.trend import EMAIndicator, SMAIndicator
from typing import Any, List

# global variables
PANEL_THRESHOLD = 5
SCREENER_TYPES = [
    "binance",
    "bybit",
]
TIMEFRAMES = [
    "5m",
    "15m",
    "1h",
    "4h",
    "1d",
]


class Screener:
    """Returns a Screener object that can request an Exchange API, calculate TA indicators, and list relevant pairs based on historic price data."""

    def __init__(self, screener_type: str, timeframe: str) -> None:
        """Args:
        screener_type (str): must be 1 of SCREENER_TYPES
        interval (str): must be 1 of SCREENER_TYPES_PROFILES[screener_type]["interval"].keys()
        """
        if not screener_type in SCREENER_TYPES:
            raise Exception(
                f"""Screener type must be one of '{", ".join(t for t in SCREENER_TYPES)}'"""
            )
        self.screener_type = screener_type
        self.exchange: ccxt.Exchange = getattr(ccxt, screener_type)()

        if not timeframe in TIMEFRAMES:
            raise Exception(
                f"""Interval must be one of '{", ".join(i for i in TIMEFRAMES)}'"""
            )
        self.timeframe = timeframe
        self.usdt_pairs_counter: int = 0

    @property
    def open_contracts(self) -> List[str]:
        """returns a list of strings with tradable pairs"""
        if not hasattr(self, "_open_contracts"):
            token_dict = self.exchange.load_markets()
            self._open_contracts = [
                token_dict[i].get("info", {}).get("symbol", "")
                for i in token_dict.keys()
                if token_dict[i].get("info", {}).get("status", "").lower() == "trading"
                and token_dict[i].get("info", {}).get("symbol", "")
            ]
            self._open_contracts = list(set(self._open_contracts))
            self._open_contracts.sort()
        return self._open_contracts

    def get_candles(self, symbol: str) -> List[List[Any]]:
        """returns a list of lists with candle information

        Args:
            symbol (str): like BTCUSDT or ETHUSD

        Returns:
            List[List[Any]]: list of candles where 'a candle' is a list of strings with candle information
        """

        try:
            return self.exchange.fetch_ohlcv(symbol, self.timeframe)[:-1]
        except Exception as e:
            print(str(e))
            return []

    def get_dataframe(self, candles: List[List[Any]]) -> DataFrame | None:
        """returns pandas dataframe or None

        Args:
            candles (List[List[Any]]): candles with candle information

        Returns:
            DataFrame | None: pandas dataframe if everything works, None if else
        """
        # refuse empty candle dict
        if not candles:
            return None

        # load dataframe
        dataframe = pd.DataFrame.from_records(candles)

        # refuse less than 31 candles
        if not len(dataframe.index) or len(dataframe.index) < 31:
            return None

        # column names
        dataframe.columns = ["Time", "Open", "High", "Low", "Close", "Volume"]

        # change al column_name(s) data types to float32
        for column_name in dataframe.columns:
            dataframe = dataframe.astype(
                {
                    column_name: "float32",
                }
            )

        # add all TA features, why not
        dataframe = add_all_ta_features(
            dataframe,
            open="Open",
            high="High",
            low="Low",
            close="Close",
            volume="Volume",
            fillna=True,
        )

        # 8EMA
        dataframe["8ema"] = EMAIndicator(dataframe["Close"], window=8).ema_indicator()
        dataframe["8ema_shift"] = dataframe.shift(periods=1)["8ema"]
        dataframe["close_shift"] = dataframe.shift(periods=1)["Close"]
        dataframe["8ema_shift2"] = dataframe.shift(periods=2)["8ema"]
        dataframe["close_shift2"] = dataframe.shift(periods=2)["Close"]
        dataframe["2_over_8ema"] = np.where(
            np.logical_and(
                dataframe["close_shift"] > dataframe["8ema_shift"],
                dataframe["close_shift2"] > dataframe["8ema_shift2"],
            ),
            1,
            0,
        )
        dataframe["2_under_8ema"] = np.where(
            np.logical_and(
                dataframe["close_shift"] < dataframe["8ema_shift"],
                dataframe["close_shift2"] < dataframe["8ema_shift2"],
            ),
            1,
            0,
        )

        # 21EMA
        dataframe["21ema"] = EMAIndicator(dataframe["Close"], window=21).ema_indicator()
        dataframe["8ema_over_21ema"] = np.where(
            dataframe["8ema"] > dataframe["21ema"], 1, 0
        )
        dataframe["8ema_under_21ema"] = np.where(
            dataframe["8ema"] < dataframe["21ema"], 1, 0
        )

        # 200 SMA
        dataframe["200sma"] = SMAIndicator(
            dataframe["Close"], window=200
        ).sma_indicator()
        dataframe["price_above_200sma"] = np.where(
            dataframe["Close"] > dataframe["200sma"], 1, 0
        )
        dataframe["price_below_200sma"] = np.where(
            dataframe["Close"] < dataframe["200sma"], 1, 0
        )

        # ichimoku cloud color
        dataframe["ichimoku_cloud_green"] = np.where(
            dataframe["trend_ichimoku_a"] > dataframe["trend_ichimoku_b"], 1, 0
        )
        dataframe["ichimoku_cloud_red"] = np.where(
            dataframe["trend_ichimoku_a"] < dataframe["trend_ichimoku_b"], 1, 0
        )

        # ichimoku above / under the cloud
        dataframe["ichimoku_above_cloud"] = np.where(
            np.logical_and(
                dataframe["Close"] > dataframe["trend_visual_ichimoku_a"],
                dataframe["Close"] > dataframe["trend_visual_ichimoku_b"],
            ),
            1,
            0,
        )

        dataframe["ichimoku_beneeth_cloud"] = np.where(
            np.logical_and(
                dataframe["Close"] < dataframe["trend_visual_ichimoku_a"],
                dataframe["Close"] < dataframe["trend_visual_ichimoku_b"],
            ),
            1,
            0,
        )

        # ichimoku conversion / base line
        dataframe["ichimoku_conversion_over_base"] = np.where(
            dataframe["trend_ichimoku_conv"] > dataframe["trend_ichimoku_base"], 1, 0
        )
        dataframe["ichimoku_conversion_under_base"] = np.where(
            dataframe["trend_ichimoku_conv"] < dataframe["trend_ichimoku_base"], 1, 0
        )
        dataframe["ichimoku_above_lag"] = np.where(
            dataframe["Close"] - dataframe.shift(periods=26)["Close"] > 0,
            1,
            0,
        )
        dataframe["ichimoku_beneeth_lag"] = np.where(
            dataframe["ichimoku_above_lag"] == 0, 1, 0
        )

        # MACD
        dataframe["macd_green"] = np.where(dataframe["trend_macd_diff"] > 0, 1, 0)
        dataframe["macd_red"] = np.where(dataframe["macd_green"] == 0, 1, 0)

        # THE PANELS
        # long
        dataframe["panel_long"] = (
            dataframe["ichimoku_cloud_green"]
            + dataframe["ichimoku_above_cloud"]
            + dataframe["ichimoku_conversion_over_base"]
            + dataframe["ichimoku_above_lag"]
            + dataframe["macd_green"]
            + dataframe["2_over_8ema"]
            + dataframe["8ema_over_21ema"]
            + dataframe["price_above_200sma"]
        )

        # short
        dataframe["panel_short"] = (
            dataframe["ichimoku_cloud_red"]
            + dataframe["ichimoku_beneeth_cloud"]
            + dataframe["ichimoku_conversion_under_base"]
            + dataframe["ichimoku_beneeth_lag"]
            + dataframe["macd_red"]
            + dataframe["2_under_8ema"]
            + dataframe["8ema_under_21ema"]
            + dataframe["price_below_200sma"]
        )

        dataframe = dataframe.astype(
            {
                "panel_long": "int",
                "panel_short": "int",
            }
        )
        return dataframe

    @property
    def discord_text(self) -> str:
        return f"""@here {len(self.open_contracts)} {" ".join(self.screener_type.title().split("_"))} pairs scanned in total
{str(self.usdt_pairs_counter)} USDT pairs selected for rating on timeframe `{self.timeframe}` with Python using an API

8/21 EMA cross + {PANEL_THRESHOLD}/8 panel score minimum:
- 2 candle closes above/below 8 EMA
- 8/21 EMA cross
- MAC-D color
- 4 Ichimoku Cloud indicators
- candle close above/below 200 SMA

Displaying it like: score(yesterday's score) token name"""

    def run(self):
        """Puts all class functions together. Prints all pair information as it loops open contracts. Finally prints all interesting pairs with scores."""
        longs = []
        shorts = []

        # print index row
        print("symbol\t\tlong \tshort")

        # loop over tickers
        for symbol in self.open_contracts:

            # remove stuff that somehow doesn"t work
            if any(
                i in symbol
                for i in [
                    "TUSD",
                    "UST",
                    "USDN",
                    "USDJ",
                    "USDC",
                    "SUSD",
                    "OUSD",
                    "DOWN/USDT",
                    "UP/USDT",
                    "BUSD",
                    "1000",
                ]
            ):
                continue

            if not "USDT" in symbol:
                continue

            if symbol.startswith("USD"):
                continue

            # create the dataframe with candles
            dataframe = self.get_dataframe(self.get_candles(symbol))

            if not isinstance(dataframe, pd.DataFrame):
                continue

            self.usdt_pairs_counter += 1

            # get panel scores
            panel_long = dataframe["panel_long"].iloc[-1]
            panel_long_shift = dataframe["panel_long"].iloc[-2]
            panel_short = dataframe["panel_short"].iloc[-1]
            panel_short_shift = dataframe["panel_short"].iloc[-2]
            print(symbol, "\t", panel_long, "\t", panel_short)

            if (
                panel_long >= PANEL_THRESHOLD
                and dataframe["8ema_over_21ema"].iloc[-1] == 1
            ):
                longs.append(
                    (
                        symbol,
                        panel_long,
                        panel_long_shift,
                    )
                )

            if (
                panel_short >= PANEL_THRESHOLD
                and dataframe["8ema_under_21ema"].iloc[-1] == 1
            ):
                shorts.append(
                    (
                        symbol,
                        panel_short,
                        panel_short_shift,
                    )
                )

        # sort by panel score
        longs.sort(key=lambda x: x[1], reverse=True)
        shorts.sort(key=lambda x: x[1], reverse=True)

        print()
        print(self.discord_text)

        if longs:
            print()
            print(
                f"{len(longs)} LONGS:\nscore coin\n"
                + "\n".join("{}({}) {}".format(i[1], int(i[2]), i[0]) for i in longs)
            )

        if shorts:
            print()
            print(
                f"{len(shorts)} SHORTS:\nscore coin\n"
                + "\n".join("{}({}) {}".format(i[1], int(i[2]), i[0]) for i in shorts)
            )


if __name__ == "__main__":
    screener_type = ""
    while not screener_type in SCREENER_TYPES:
        screener_type = input(", ".join(SCREENER_TYPES) + "?\n")

    interval = ""
    while not interval in TIMEFRAMES:
        interval = input(", ".join(TIMEFRAMES) + "?\n")

    Screener(screener_type, interval).run()
