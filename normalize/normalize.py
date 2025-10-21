from __future__ import annotations
import pandas as pd
import numpy as np
from typing import List, Optional

class Normalize:
    """Cross-sectional neutralization with (t-1) controls."""

    @staticmethod
    def prepare_controls_lagged(
        factors_df: pd.DataFrame,
        controls: List[str],
        group_col: Optional[str] = None,
        weight_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Shift control exposures forward by 1 day within each ticker,
        so that signal_t is regressed on controls_{t-1}.
        """
        df = factors_df.copy().sort_values(["Ticker", "Date"])
        for c in controls:
            df[c + "_lag"] = df.groupby("Ticker")[c].shift(1)
        if group_col is not None and group_col in df.columns:
            df[group_col + "_lag"] = df.groupby("Ticker")[group_col].shift(1)
        if weight_col is not None and weight_col in df.columns:
            df[weight_col + "_lag"] = df.groupby("Ticker")[weight_col].shift(1)
        return df

    @staticmethod
    def neutralize_cross_sectional_lagged(
        signal_df: pd.DataFrame,  # columns: ['Date','Ticker','factorvalue']
        factors_df: pd.DataFrame,  # columns: ['Date','Ticker', controls..., optional group/weight]
        controls: List[str] = ["MarketCap", "Volatility", 'Momentum',"Beta", "Turnover"], 
        signal_col: str = "factorvalue",
        group_col: Optional[str] = None,
        weight_col: Optional[str] = None,
        winsor_clip: float = 5.0,
        zscore_output: bool = False,
        min_obs_per_day: int = 30,
    ) -> pd.DataFrame:
        """
        Neutralizes signal_t using cross-sectional regression on controls_{t-1}.
        Inputs (signal & controls) are normalized cross-sectionally per date.
        Returns a copy of the single-factor table with neutralized values stored
        in the original signal column name.
        """
        # 1) lag controls
        lagged = Normalize.prepare_controls_lagged(
            factors_df, controls, group_col, weight_col
        )

        # 2) merge with signal
        panel = (
            pd.merge(signal_df, lagged, on=["Date", "Ticker"], how="left")
            .sort_values(["Date", "Ticker"])
            .reset_index(drop=True)
        )

        results = []
        for _, sdf in panel.groupby("Date", sort=True):
            sdf = sdf.copy()

            # Build design matrix from lagged controls (cross-sectional z-score)
            Xparts = []
            for c in controls:
                x = sdf[c + "_lag"]
                if x.notna().sum() == 0:
                    continue
                mu = x.mean(skipna=True)
                sd = x.std(ddof=0, skipna=True)
                if sd == 0 or not np.isfinite(sd):
                    Xparts.append(pd.Series(np.nan, index=sdf.index, name=c + "_z"))
                else:
                    Xparts.append(((x - mu) / sd).rename(c + "_z"))

            if not Xparts:
                sdf["neu_raw"] = np.nan
                results.append(sdf[["Date", "Ticker", signal_col, "neu_raw"]])
                continue

            X = pd.concat(Xparts, axis=1)
            X.insert(0, "intercept", 1.0)

            # y = cross-sectional z-score of signal
            y_raw = sdf[signal_col]
            y_sd = y_raw.std(ddof=0, skipna=True)
            y = (y_raw - y_raw.mean(skipna=True)) / (y_sd if np.isfinite(y_sd) and y_sd != 0 else 1.0)

            # Drop NA rows for the day's regression
            good = ~(X.isna().any(axis=1) | y.isna())
            if good.sum() < max(min_obs_per_day, X.shape[1] + 2):
                sdf["neu_raw"] = np.nan
                results.append(sdf[["Date", "Ticker", signal_col, "neu_raw"]])
                continue

            Xg, yg = X.loc[good].values, y.loc[good].values

            # OLS (cross-sectional)
            beta, *_ = np.linalg.lstsq(Xg, yg, rcond=None)
            yhat = X.values @ beta
            resid = y.values - yhat
            sdf["neu_raw"] = resid

            results.append(sdf[["Date", "Ticker", signal_col, "neu_raw"]])

        res = pd.concat(results, ignore_index=True).sort_values(["Date", "Ticker"])

        merged = signal_df.merge(
            res[["Date", "Ticker", "neu_raw"]],
            on=["Date", "Ticker"],
            how="left",
        )
        merged = merged[["Date", "Ticker", "neu_raw"]]
        merged.rename(columns={"neu_raw": signal_col}, inplace=True)
        return merged


    # =========================================================
    # Factor building helpers
    # =========================================================
    def get_benchmark(start: str = "2015-01-01", end: str = "2025-12-31") -> pd.DataFrame:
        """Return IWV (Russell 3000) daily market return."""
        iwv = yf.download("IWV", start=start, end=end, progress=False)
        if isinstance(iwv.columns, pd.MultiIndex):
            iwv.columns = iwv.columns.get_level_values(0)
        iwv.index = iwv.index.tz_localize(None)
        iwv["Ret_mkt"] = iwv["Close"].pct_change()
        return iwv[["Ret_mkt"]]


    def build_factors(
        data_table: pd.DataFrame,
        shares_outstanding_map: Dict[str, float],
        benchmark: pd.DataFrame,
        start: str = "2015-01-01",
        end: str = "2025-12-31",
    ) -> pd.DataFrame:
        """
        Build factor exposures per ticker.
        NOTE: This uses a single (constant) SharesOutstanding per ticker; if that
        value is current SO, it can be a source of look-ahead. Prefer a dated SO series.
        """
        all_factors: List[pd.DataFrame] = []

        for t, df in data_table.groupby("Ticker"):
            try:
                sdf = df.copy()
                sdf["Date"] = pd.to_datetime(sdf["Date"]).dt.tz_localize(None)

                so = shares_outstanding_map.get(t, None)
                if so is None:
                    continue

                sdf["SharesOutstanding"] = so
                sdf["MarketCap"] = sdf["Close"] * so
                sdf["Ret"] = sdf["Close"].pct_change()

                # Factors
                sdf["Momentum"] = sdf["Close"].pct_change(30)
                sdf["Turnover"] = sdf["Volume"] / so
                sdf["Volatility"] = sdf["Ret"].rolling(30).std()

                # Merge with benchmark for rolling beta
                sdf = sdf.merge(benchmark, left_on="Date", right_index=True, how="left")
                cov = sdf["Ret"].rolling(30).cov(sdf["Ret_mkt"])
                var = sdf["Ret_mkt"].rolling(30).var()
                sdf["Beta"] = cov / var

                # Clip date range
                sdf = sdf[(sdf["Date"] >= start) & (sdf["Date"] <= end)]

                all_factors.append(
                    sdf[
                        [
                            "Date",
                            "Ticker",
                            "MarketCap",
                            "SharesOutstanding",
                            "Momentum",
                            "Turnover",
                            "Volatility",
                            "Beta",
                        ]
                    ]
                )
            except Exception as e:
                print(f"⚠️ Error {t}: {e}")

        return pd.concat(all_factors, ignore_index=True) if all_factors else pd.DataFrame()


    def run_in_batches(
        data_table: pd.DataFrame,
        tickers: List[str],
        shares_outstanding_map: Dict[str, float],
        benchmark: pd.DataFrame,
        start: str = "2015-01-01",
        end: str = "2025-12-31",
        batch_size: int = 200,
    ) -> pd.DataFrame:
        """Build factors in batches of tickers to control memory."""
        results: List[pd.DataFrame] = []
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i : i + batch_size]
            print(f"▶️ Processing batch {i}:{i + batch_size} ...")
            subset = data_table[data_table["Ticker"].isin(batch)]
            f = build_factors(subset, shares_outstanding_map, benchmark, start, end)
            if not f.empty:
                results.append(f)
            print(f"✅ Finished batch {i}:{i + batch_size}, got {len(f)} rows")
        return pd.concat(results, ignore_index=True) if results else pd.DataFrame()

