# Data

The dataset is not included in this repository. It is free to download from HistData.com:

1. Go to <https://www.histdata.com/download-free-forex-historical-data/?/excel/1-minute-bar-quotes/eurusd/2023>
2. Download **EUR/USD, 1-minute bar quotes, 2023** (Excel format).
3. Unzip it and place `DAT_XLSX_EURUSD_M1_2023.xlsx` in this folder.

The ASCII (CSV) version from HistData also works; pass its path with `--data`.

**About the data:** 1-minute OHLC bars of the EUR/USD bid price for 2023, with timestamps in EST without daylight-saving adjustments. HistData omits minutes in which no ticks occurred.
