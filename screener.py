from typing import Any, List
from decouple import config
import pandas as pd
from pandas import DataFrame
import math
import datetime as dt
import requests
import numpy as np
import time
import json
import base64
import hmac
import hashlib
from ta import add_all_ta_features
from ta.trend import EMAIndicator


# global variables
PANEL_THRESHOLD = 5
KUCOIN_REGULAR = "kucoin_regular"
KUCOIN_FUTURES = "kucoin_futures"
BINANCE_REGULAR = "binance_regular"
BYBIT = "bybit"

SCREENER_TYPES = [
    KUCOIN_REGULAR,
    KUCOIN_FUTURES,
    BINANCE_REGULAR,
    BYBIT,
]


class Credentials:
    """Credentials to be used for sending API requests"""

    def __init__(self, screener_type: str) -> None:
        s_type = screener_type.upper()
        self.api_key = config(f"{s_type}_API_KEY")
        self.api_secret = config(f"{s_type}_API_SECRET")
        if not screener_type in [
            BINANCE_REGULAR,
            BYBIT,
        ]:
            self.api_passphrase = config(f"{s_type}_API_PASSPHRASE")


SCREENER_TYPES_PROFILES = {
    KUCOIN_REGULAR: dict(
        column_names=[
            "Time",
            "Entry price",
            "Close price",
            "Highest price",
            "Lowest price",
            "Trading volume",
            "Trading amount",
        ],
        interval={
            "5m": "5min",
            "15m": "15min",
            "1h": "1hour",
            "4h": "4hour",
            "d": "1day",
        },
        base_url="https://api.kucoin.com",
        open_contracts_path="/api/v1/symbols",
        candle_request_path="/api/v1/market/candles",
    ),
    KUCOIN_FUTURES: dict(
        column_names=[
            "Time",
            "Entry price",
            "Highest price",
            "Lowest price",
            "Close price",
            "Trading volume",
        ],
        interval={
            "5m": "5",
            "15m": "15",
            "1h": "60",
            "4h": "240",
            "d": "1440",
        },
        base_url="https://api-futures.kucoin.com",
        open_contracts_path="/api/v1/contracts/active",
        candle_request_path="/api/v1/kline/query?",
    ),
    BINANCE_REGULAR: dict(
        column_names=[
            "Time",
            "Entry price",
            "Highest price",
            "Lowest price",
            "Close price",
            "Trading volume",
            "Close time",
            "Quote asset volume",
            "Trading amount",
            "Taker buy base asset volume",
            "Taker buy quote asset volume",
            "Ignore",
        ],
        interval={
            "5m": "5m",
            "15m": "15m",
            "1h": "1h",
            "4h": "4h",
            "d": "1d",
        },
        base_url="https://api.binance.com",
        open_contracts_path="/api/v3/ticker/bookTicker",
        candle_request_path=None,
    ),
    BYBIT: dict(
        column_names=[
            "start",
            "Entry price",
            "Highest price",
            "Lowest price",
            "Close price",
            "Trading volume",
            "Turnover",
        ],
        interval={
            "5m": 5,
            "15m": 15,
            "1h": 60,
            "4h": 240,
            "d": "D",
        },
        base_url="https://api.bybit.com",
        open_contracts_path="/v2/public/tickers",
        candle_request_path="/derivatives/v3/public/kline",
    ),
}


class Screener:
    """Returns a Screener object that can request an Exchange API, calculate TA indicators, and list relevant pairs based on historic price data."""

    def __init__(self, screener_type: str, interval: str) -> None:
        """Args:
        screener_type (str): must be 1 of SCREENER_TYPES
        interval (str): must be 1 of SCREENER_TYPES_PROFILES[screener_type]["interval"].keys()
        """
        if not screener_type in SCREENER_TYPES:
            raise Exception(
                f"""Screener type must be one of '{", ".join(t for t in SCREENER_TYPES)}'"""
            )
        self.screener_type = screener_type
        self.profile = SCREENER_TYPES_PROFILES[self.screener_type]
        self.credentials = Credentials(self.screener_type)

        if not interval in self.profile.get("interval", {}).keys():
            raise Exception(
                f"""Interval must be one of '{", ".join(i for i in self.profile.get("interval", {}))}'"""
            )
        self.interval = interval

        if self.screener_type == BINANCE_REGULAR:
            from binance.spot import Spot

            self.client = Spot(
                key=self.credentials.api_key, secret=self.credentials.api_secret
            )
        self.usdt_pairs_counter: int = 0

    def get_open_contracts(self) -> List[str]:
        """returns a list of strings with tradable pairs"""
        if self.screener_type == BINANCE_REGULAR:
            open_contracts_list = self.client.send_request(
                "GET", self.profile.get("open_contracts_path")
            )
            self.open_contracts = [
                contract.get("symbol") for contract in open_contracts_list
            ]

        else:
            open_contracts_dict = self._api_request(
                method="get",
                base_url=self.profile.get("base_url"),
                path=self.profile.get("open_contracts_path"),
            )
            if "kucoin" in self.screener_type:
                self.open_contracts = [
                    contract.get("symbol", "")
                    for contract in open_contracts_dict.get("data", [])
                ]
            else:
                self.open_contracts = [
                    contract.get("symbol", "")
                    for contract in open_contracts_dict.get("result", [])
                ]
        self.open_contracts.sort()
        # [print(f'\"{i}\",') for i in self.open_contracts if not i.startswith("USDT")]
        return self.open_contracts

    def get_candles(self, symbol: str) -> List[List[Any]]:
        """returns a list of lists with candle information

        Args:
            symbol (str): like BTCUSDT or ETHUSD

        Returns:
            List[List[Any]]: list of candles where 'a candle' is a list of strings with candle information
        """
        if self.screener_type == BINANCE_REGULAR:
            return self.client.klines(symbol, "1d", limit=100)

        now = dt.datetime.now()
        nr_of_candles = 100
        start_dict = {
            "5m": {"minutes": 5 * nr_of_candles},
            "15m": {"minutes": 15 * nr_of_candles},
            "1h": {"minutes": 60 * nr_of_candles},
            "4h": {"minutes": 240 * nr_of_candles},
            "d": {"days": nr_of_candles},
        }
        start = (
            math.floor(
                int(
                    dt.datetime.timestamp(
                        now - dt.timedelta(**start_dict[self.interval])
                    )
                )
            )
            * 1000
        )
        end = math.floor(int(dt.datetime.timestamp(now))) * 1000

        if self.screener_type == KUCOIN_FUTURES:
            params = {
                "symbol": symbol,
                "from": start,
                "to": end,
                "granularity": self.profile["interval"][self.interval],
            }
        if self.screener_type == KUCOIN_REGULAR:
            params = {
                "symbol": symbol,
                "startAt": start,
                "endAt": end,
                "type": self.profile["interval"][self.interval],
                # "1day", # daily candles
                # "type": "4hour", # 4H candles
            }
        if self.screener_type == BYBIT:
            params = {
                "symbol": symbol,
                "start": start,
                "end": end,
                "interval": self.profile["interval"][self.interval],
            }

        return_value = self._api_request(
            method="get",
            base_url=self.profile.get("base_url"),
            path=self.profile.get("candle_request_path"),
            params=params,
        )
        match self.screener_type:
            case "binance_regular":
                return return_value
            case "bybit":
                return return_value.get("result", {}).get("list")
            case _:
                return return_value.get("data")

    def _api_request(self, method, base_url, path, params=None, json_dict=None) -> dict:
        """returns the response of an API request

        Args:
            method (_type_): GET / POST
            base_url (_type_): https base url
            path (_type_): /path
            params (_type_, optional): additional API information. Defaults to None.
            json_dict (_type_, optional). Defaults to None.

        Returns:
            dict: _description_
        """
        # current time in milliseconds timestamp
        now = int(time.time() * 1000)

        # string to sign
        str_to_sign = str(now) + method.upper() + path

        # add the json_dict as json if presented (for post requests)
        if json_dict:
            json_data = json.dumps(json_dict)
            str_to_sign += json_data
        else:
            json_data = None

        # create signature
        signature = base64.b64encode(
            hmac.new(
                self.credentials.api_secret.encode("utf-8"),
                str_to_sign.encode("utf-8"),
                hashlib.sha256,
            ).digest(),
        )

        # create passphrase / headers
        if "kucoin" in self.screener_type:
            passphrase = base64.b64encode(
                hmac.new(
                    self.credentials.api_secret.encode("utf-8"),
                    self.credentials.api_passphrase.encode("utf-8"),
                    hashlib.sha256,
                ).digest(),
            )
            headers = {
                "KC-API-SIGN": signature,
                "KC-API-TIMESTAMP": str(now),
                "KC-API-KEY": self.credentials.api_key,
                "KC-API-PASSPHRASE": passphrase,
                "KC-API-KEY-VERSION": "2",
                "Content-Type": "application/json",
            }
        elif self.screener_type == BYBIT:
            headers = {
                "X-BAPI-SIGN-TYPE": "2",
                "X-BAPI-SIGN": signature,
                "X-BAPI-API-KEY": self.credentials.api_key,
                "X-BAPI-TIMESTAMP": str(now),
                "X-BAPI-RECV-WINDOW": "5000",
            }

        # do the request
        r = requests.request(
            method, base_url + path, headers=headers, params=params, data=json_data
        )
        r.raise_for_status()

        if r.status_code == 200:
            return r.json()
        else:
            print(r.content.decode())
            return {}

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
        dataframe.columns = self.profile.get("column_names")

        # change al column_name(s) data types to float32
        for column_name in dataframe.columns:
            dataframe = dataframe.astype(
                {
                    column_name: "float32",
                }
            )

        # api"s are different (seconds / milliseconds)
        if self.screener_type == KUCOIN_FUTURES:
            dataframe["Time"] = pd.to_datetime(dataframe["Time"], unit="ms")
        if self.screener_type == KUCOIN_REGULAR:
            dataframe["Time"] = pd.to_datetime(dataframe["Time"], unit="s")

            # reverse candles, oldest first
            dataframe = dataframe.loc[::-1]
        if self.screener_type == BYBIT:
            dataframe = dataframe.loc[::-1]

        # add all TA features, why not
        dataframe = add_all_ta_features(
            dataframe,
            open="Entry price",
            high="Highest price",
            low="Lowest price",
            close="Close price",
            volume="Trading volume",
            fillna=True,
        )

        # 8EMA
        dataframe["8ema"] = EMAIndicator(
            dataframe["Close price"], window=8
        ).ema_indicator()
        dataframe["8ema_shift"] = dataframe.shift(periods=1)["8ema"]
        dataframe["close_shift"] = dataframe.shift(periods=1)["Close price"]
        dataframe["8ema_shift2"] = dataframe.shift(periods=2)["8ema"]
        dataframe["close_shift2"] = dataframe.shift(periods=2)["Close price"]
        dataframe["2_over_8ema"] = np.where(np.logical_and(
            dataframe["close_shift"] > dataframe["8ema_shift"], dataframe["close_shift2"] > dataframe["8ema_shift2"]),
            1, 0
        )
        dataframe["2_under_8ema"] = np.where(np.logical_and(
            dataframe["close_shift"] < dataframe["8ema_shift"], dataframe["close_shift2"] < dataframe["8ema_shift2"]),
            1, 0
        )

        # 21EMA
        dataframe["21ema"] = EMAIndicator(
            dataframe["Close price"], window=21
        ).ema_indicator()
        dataframe["8ema_over_21ema"] = np.where(
            dataframe["8ema"] > dataframe["21ema"],
            1, 0
        )
        dataframe["8ema_under_21ema"] = np.where(
            dataframe["8ema"] < dataframe["21ema"],
            1, 0
        )

        # ichimoku cloud color
        dataframe["ichimoku_cloud_green"] = np.where(
            dataframe["trend_ichimoku_a"] > dataframe["trend_ichimoku_b"],
            1, 0
        )
        dataframe["ichimoku_cloud_red"] = np.where(
            dataframe["trend_ichimoku_a"] < dataframe["trend_ichimoku_b"],
            1, 0
        )

        # ichimoku above / under the cloud
        dataframe["ichimoku_above_cloud"] = np.where(np.logical_and(
            dataframe["Close price"] > dataframe["trend_visual_ichimoku_a"],
            dataframe["Close price"] > dataframe["trend_visual_ichimoku_b"]),
            1, 0
        )
        
        dataframe["ichimoku_beneeth_cloud"] = np.where(np.logical_and(
            dataframe["Close price"] < dataframe["trend_visual_ichimoku_a"],
            dataframe["Close price"] < dataframe["trend_visual_ichimoku_b"]),
            1, 0
        )

        # ichimoku conversion / base line
        dataframe["ichimoku_conversion_over_base"] = np.where(
            dataframe["trend_ichimoku_conv"] > dataframe["trend_ichimoku_base"],
            1, 0
        )
        dataframe["ichimoku_conversion_under_base"] = np.where(
            dataframe["trend_ichimoku_conv"] < dataframe["trend_ichimoku_base"],
            1, 0
        )
        dataframe["ichimoku_above_lag"] = np.where(
            dataframe["Close price"] - dataframe.shift(periods=26)["Close price"] > 0,
            1, 0
        )
        dataframe["ichimoku_beneeth_lag"] = np.where(
            dataframe["ichimoku_above_lag"] == 0,
            1, 0
        )

        # MACD
        dataframe["macd_green"] = np.where(
            dataframe["trend_macd_diff"] > 0,
            1, 0
        )
        dataframe["macd_red"] = np.where(
            dataframe["macd_green"] == 0,
            1, 0
        )

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
{str(self.usdt_pairs_counter)} USDT pairs selected for rating on timeframe `{self.interval}` with Python using an API
8/21 EMA cross + 5/7 Assassins panel score minimum #custom-indicator-scripts
displaying it like: score(yesterday's score) token name"""

    def run(self):
        """Puts all class functions together. Prints all pair information as it loops open contracts. Finally prints all interesting pairs with scores."""
        longs = []
        shorts = []

        # print index row
        print("symbol\t\tlong \tshort")

        # get open contracts
        self.get_open_contracts()

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
                    "DOWNUSDT",
                    "UPUSDT",
                    "BUSD",
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
    while not interval in SCREENER_TYPES_PROFILES[screener_type]["interval"].keys():
        interval = input(
            ", ".join(SCREENER_TYPES_PROFILES[screener_type]["interval"].keys()) + "?\n"
        )

    Screener(screener_type, interval).run()
