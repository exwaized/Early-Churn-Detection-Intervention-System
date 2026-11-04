"""
Convert the UCI "Online Retail II" workbook into the event log this pipeline expects.

Source: https://archive.ics.uci.edu/dataset/502/online+retail+ii (CC BY 4.0), a UK online
retailer's real transactions, 2009-12-01 .. 2011-12-09 (two sheets).

Mapping (one row per purchase = one "session"):
    user_id       <- Customer ID
    event_date    <- invoice date
    session_id    <- Invoice
    actions_count <- number of line items on the invoice
    features_used <- number of distinct products (StockCode) on the invoice
    revenue       <- sum(Quantity * Price), GBP   (extra column, used for the ROI analysis)

Cleaning: drop rows without a Customer ID, cancellations (Invoice starting with "C"),
non-positive quantity/price, and non-product stock codes (POST, DOT, M, BANK CHARGES, ...).

Usage: python data/prepare_retail.py path/to/online_retail_II.xlsx events.csv
"""
import sys
import pandas as pd

NON_PRODUCT = {"POST", "D", "DOT", "M", "BANK CHARGES", "AMAZONFEE", "CRUK", "S", "B", "ADJUST", "ADJUST2", "TEST001", "TEST002"}


def main(src: str, dst: str):
    sheets = pd.read_excel(src, sheet_name=None, dtype={"Invoice": str, "StockCode": str})
    df = pd.concat(sheets.values(), ignore_index=True)
    raw = len(df)
    df = df.dropna(subset=["Customer ID"])
    df = df[~df["Invoice"].str.startswith("C")]
    df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]
    df = df[~df["StockCode"].str.upper().isin(NON_PRODUCT)]
    df["line_value"] = df["Quantity"] * df["Price"]
    ev = (df.groupby(["Customer ID", "Invoice"])
            .agg(event_date=("InvoiceDate", "min"), actions_count=("StockCode", "size"),
                 features_used=("StockCode", "nunique"), revenue=("line_value", "sum"))
            .reset_index()
            .rename(columns={"Customer ID": "user_id", "Invoice": "session_id"}))
    ev["user_id"] = ev["user_id"].astype(int)
    ev["event_date"] = pd.to_datetime(ev["event_date"]).dt.normalize()
    ev = ev[["user_id", "event_date", "session_id", "actions_count", "features_used", "revenue"]]
    ev.to_csv(dst, index=False)
    print(f"{raw:,} raw rows -> {len(ev):,} purchases from {ev['user_id'].nunique():,} customers "
          f"({ev['event_date'].min().date()} .. {ev['event_date'].max().date()})")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
