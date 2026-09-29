"""
Downloads free real 15-minute history (2012-2022) for gold and three FX pairs from
the ejtraderLabs/historical-data repository on GitHub, into data/real/.

    python get_sample_data.py

Then, for example:
    python run_all.py --data data/real/XAUUSDm15.csv --cost 25

Price units in these files: XAUUSD 1 unit = $0.01; FX pairs 1 unit = 0.1 pip.
The --cost values used in the README (XAUUSD 25, USDJPY 10, GBPJPY 18, EURUSD 10) are
rough raw-spread-account costs including commission and a little slippage.
Use your own broker's numbers.
"""
import os
import urllib.request

URL = "https://raw.githubusercontent.com/ejtraderLabs/historical-data/main/{s}/{s}m15.csv"
SYMBOLS = ["XAUUSD", "USDJPY", "GBPJPY", "EURUSD"]

if __name__ == "__main__":
    os.makedirs(os.path.join("data", "real"), exist_ok=True)
    for s in SYMBOLS:
        path = os.path.join("data", "real", f"{s}m15.csv")
        if os.path.exists(path):
            print(f"{path} already there")
            continue
        print(f"downloading {s} ...")
        urllib.request.urlretrieve(URL.format(s=s), path)
        print(f"  saved {path} ({os.path.getsize(path) / 1e6:.1f} MB)")
