from __future__ import annotations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from typing import Dict, List, Optional, Tuple
from data.get_stock_data import GetStockData
    # =========================================================
    # Backtest (monthly rebalance, MTD compounding)
    # =========================================================
class Backtest:
    def __init__(self, factor_table: pd.DataFrame) -> None:
        self.factor_table = factor_table
        loader = GetStockData()
        self.rawdata = loader._get_russell3000_data()
        self.sectorcode = loader._get_russell3000_sector() or {}

    # ---------- Simple helpers ----------
    def _get_return(self) -> pd.DataFrame:
        """Daily simple return using raw Open price (consider using adjusted open)."""
        ret = self.rawdata.groupby("Ticker", group_keys=False).apply(
            lambda x: x["Open"].pct_change()
        )
        self.rawdata["return"] = ret
        return self.rawdata[["Date", "Ticker", "return"]]

    def _pivot_factor(self, factor_name: str) -> pd.DataFrame:
        """Pivot factor table to Date x Ticker values."""
        return pd.pivot_table(
            self.factor_table, values=factor_name, index="Date", columns="Ticker"
        )

    # ---------- Optimizer plumbing ----------
    @staticmethod
    def _sanitize_mu(mu: pd.Series) -> pd.Series:
        mu = pd.to_numeric(mu, errors="coerce")
        mu = mu.replace([np.inf, -np.inf], np.nan)
        return mu.dropna()

    @staticmethod
    def _align_inputs(mu: pd.Series, sectors: pd.Series, w_cap):
        mu = Backtest._sanitize_mu(mu)
        sectors = sectors.astype(str)

        # --- Drop duplicate tickers in sectors, keeping first occurrence
        if not sectors.index.is_unique:
            sectors = sectors[~sectors.index.duplicated(keep="first")]

        # --- ensure alignment between mu and sectors
        tickers = mu.index.intersection(sectors.index)
        mu = mu.loc[tickers]
        sectors = sectors.reindex(tickers)  # now safe to reindex

        # --- fill missing sector labels
        sectors = sectors.fillna("UNKNOWN")

        # --- build dummy matrix, including UNKNOWN
        sec_dummies = pd.get_dummies(sectors)
        P = sec_dummies.to_numpy().T
        sector_names = sec_dummies.columns.tolist()

        # --- build per-name caps vector
        if np.isscalar(w_cap):
            wmax = np.full(len(mu), float(w_cap))
        else:
            w_cap = pd.Series(w_cap, index=tickers).astype(float)
            w_cap = w_cap.replace([np.inf, -np.inf], np.nan).fillna(np.inf)
            wmax = w_cap.values

        # ✅ Final alignment sanity check
        if P.shape[1] != len(mu):
            print(f"⚠️ Shape mismatch detected: P={P.shape[1]}, mu={len(mu)} — auto-fixing by reindexing")
            # auto-truncate to the smaller dimension to keep run going
            n = min(P.shape[1], len(mu))
            P = P[:, :n]
            mu = mu.iloc[:n]
            sectors = sectors.iloc[:n]
            wmax = wmax[:n]

        return tickers[:len(mu)], mu, sectors, wmax, P, sector_names



    @staticmethod
    def _feasibility_check(
        sectors: pd.Series, wmax: np.ndarray, P: np.ndarray, sector_names: List[str], sector_cap: float
    ) -> Tuple[bool, Dict]:
        name_capacity = float(np.sum(wmax))
        eff_caps = []
        for k in range(P.shape[0]):
            mask = P[k, :] == 1
            eff_caps.append(min(float(sector_cap), float(wmax[mask].sum())))
        sector_capacity = float(np.sum(eff_caps))
        total_capacity = min(name_capacity, sector_capacity if P.size else name_capacity)

        info = {
            "name_capacity": name_capacity,
            "sector_capacity": sector_capacity if P.size else name_capacity,
            "total_capacity": total_capacity,
            "per_sector_effective_caps": dict(zip(sector_names, eff_caps)) if P.size else {},
            "num_sectors_present": int(P.shape[0]),
        }
        feasible = total_capacity >= 1 - 1e-12
        return feasible, info

    @staticmethod
    def optimize_portfolio(
        mu: pd.Series,
        sectors: pd.Series,
        w_cap: float = 0.05,
        sector_cap: float = 0.5,
        verbose: bool = False,
    ) -> Tuple[pd.Series, Optional[pd.Series], float]:
        """
        Maximize mu' w, s.t. sum(w)=1, 0<=w<=w_cap, and P w <= sector_cap (per sector).
        Returns: (w_opt: Series, sector_expo: Series|None, objective_value: float)
        """
        tickers, mu, sectors, wmax, P, sector_names = Backtest._align_inputs(
            mu, sectors, w_cap
        )

        feasible, info = Backtest._feasibility_check(sectors, wmax, P, sector_names, sector_cap)
        if verbose:
            print("Feasibility diagnostics:", info)
        if not feasible:
            raise RuntimeError(
                "Infeasible constraints:\n"
                f"- sum per-name caps = {info['name_capacity']:.4f}\n"
                f"- sum sector capacity = {info['sector_capacity']:.4f}\n"
                f"-> total capacity = {info['total_capacity']:.4f} < 1.0\n"
                "Fix: raise sector_cap and/or w_cap, add more sectors/names, or relax fully invested."
            )

        n = len(mu)
        w = cp.Variable(n)
        objective = cp.Maximize(mu.values @ w)

        cons = [cp.sum(w) == 1, w >= 0, w <= wmax]
        if P.size:
            cons += [P @ w <= sector_cap]

        prob = cp.Problem(objective, cons)

        for solver in (cp.ECOS, cp.OSQP, cp.SCS, cp.CLARABEL):
            try:
                prob.solve(solver=solver, warm_start=True, verbose=verbose)
                if w.value is not None and prob.status in ("optimal", "optimal_inaccurate"):
                    w_opt = pd.Series(np.array(w.value).ravel(), index=tickers)
                    sector_expo = pd.Series(P @ w_opt.values, index=sector_names) if P.size else None

                    # Long-only cleanup & exact normalization
                    w_opt = w_opt.clip(lower=0)
                    s = w_opt.sum()
                    if s <= 0:
                        return pd.Series(dtype=float), sector_expo, float("nan")
                    w_opt = w_opt / s
                    return w_opt, sector_expo, float(prob.value)
            except Exception:
                continue

        # If all solvers fail
        return pd.Series(dtype=float), None, float("nan")





    # ---------- Basket construction ----------
    def implement_optimization(
        self,
        out: pd.DataFrame,
        w_cap: float = 0.05,
        sector_cap: float = 0.26,
        verbose: bool = False,
    ) -> Tuple[Dict[pd.Timestamp, List[pd.Series]], Dict[pd.Timestamp, List[Optional[pd.Series]]]]:
        """
        Create 5 quantile baskets per rebalance date and optimize each basket
        under long-only + sector constraints.
        """
        w_opt_map: Dict[pd.Timestamp, List[pd.Series]] = {}
        sector_expo_map: Dict[pd.Timestamp, List[Optional[pd.Series]]] = {}

        for _, row in out.iterrows():
            date = row[0]
            non_null = row[1:].dropna()

            # 5 baskets by rank
            baskets = pd.qcut(
                non_null.rank(method="first"),
                q=5,
                labels=False,
                duplicates="drop",
            )
            basket_cols = {
                f"Basket_{b + 1}": non_null.index[baskets == b].tolist() for b in range(5)
            }

            # sector names aligned to each basket
            sector_map: Dict[str, List[str]] = {}
            for bname, tick_list in basket_cols.items():
                sector_map[bname] = [self.sectorcode.get(tk, "UNKNOWN") for tk in tick_list]

            weight_basket_list: List[pd.Series] = []
            sector_expo_list: List[Optional[pd.Series]] = []

            for bname, tick_list in basket_cols.items():
                factorvalue = non_null[tick_list]
                sectors = pd.Series(sector_map[bname], index=tick_list, dtype=str)

                w_opt, sector_expo, _ = Backtest.optimize_portfolio(
                    factorvalue, sectors, w_cap=w_cap, sector_cap=sector_cap, verbose=verbose
                )

                if w_opt.empty:
                    continue

                weight_basket_list.append(w_opt)
                sector_expo_list.append(sector_expo)

            w_opt_map[pd.to_datetime(date)] = weight_basket_list
            sector_expo_map[pd.to_datetime(date)] = sector_expo_list

        return w_opt_map, sector_expo_map

    # ---------- Rebalance engine ----------
    @staticmethod
    def chain_mtd_rebalance(
        weight_map: Dict[pd.Timestamp, List[pd.Series]],
        prices_wide: pd.DataFrame,
        num_baskets: int = 5,
        start_value: float = 1.0,
    ) -> Tuple[Dict[str, pd.Series], Dict[str, pd.Series]]:
        """
        Apply monthly MTD compounding using first trading day of each month.
        weight_map: {prev_month_end_date -> [w_B1,...,w_Bk]}
        prices_wide: DataFrame (index=dates, columns=tickers) of price levels.
        """
        px = prices_wide.sort_index().copy()
        ref_dates = sorted(pd.to_datetime(list(weight_map.keys())))

        def fdom(ts: pd.Timestamp) -> Optional[pd.Timestamp]:
            i = px.index.searchsorted(ts)
            return None if i >= len(px.index) else px.index[i]

        value_map = {f"Basket_{i+1}": pd.Series(dtype=float) for i in range(num_baskets)}
        mtd_map = {f"Basket_{i+1}": pd.Series(dtype=float) for i in range(num_baskets)}
        prev_val = {f"Basket_{i+1}": start_value for i in range(num_baskets)}

        month_starts = sorted({(d + pd.offsets.MonthBegin(1)).normalize() for d in ref_dates})

        for mstart in month_starts:
            start = fdom(mstart)
            if start is None:
                continue
            next_start = fdom((mstart + pd.offsets.MonthBegin(1)).normalize())
            days = px.index[
                (px.index >= start)
                & (px.index <= (next_start or px.index.max() + pd.Timedelta(days=1)))
            ]
            if days.empty:
                continue

            # weights for this month = latest ref < mstart
            ref = max([d for d in ref_dates if d < mstart], default=None)
            if ref is None:
                continue
            wlist = weight_map[ref]

            # MTD returns framed off first trading day
            p0 = px.loc[start]
            valid_tickers = p0.dropna().index
            if valid_tickers.empty:
                continue

            p0 = p0.loc[valid_tickers]
            px_valid = px.loc[days, valid_tickers]
            px_valid = px_valid[p0.index]

            mtd = (px_valid.divide(p0, axis=1)) - 1.0

            for b in range(num_baskets):
                name = f"Basket_{b+1}"
                if b >= len(wlist):
                    continue
                w = wlist[b]
                if w is None or len(w) == 0:
                    continue

                tick = w.index.intersection(px_valid.columns)
                if tick.empty:
                    continue

                w = w.loc[tick].clip(lower=0.0).astype(float)
                if w.sum() <= 0:
                    continue

                sub = mtd[tick]
                mask = sub.notna().astype(float)
                denom = (mask * w).sum(axis=1).replace(0.0, np.nan)

                pnl = sub.fillna(0.0).mul(w, axis=1).sum(axis=1)
                mtd_series = (pnl / denom).fillna(0.0)

                val_series = prev_val[name] * (1.0 + mtd_series)
                prev_val[name] = float(val_series.iloc[-1])

                mtd_map[name] = pd.concat([mtd_map[name], mtd_series])
                value_map[name] = pd.concat([value_map[name], val_series])

        # drop accidental duplicate dates
        for d in (value_map, mtd_map):
            for k in d:
                if not d[k].empty:
                    d[k] = d[k][~d[k].index.duplicated(keep="last")]
        return value_map, mtd_map

    # ---------- Plotting ----------
    @staticmethod
    def chart_the_graph(value_map: Dict[str, pd.Series]) -> None:
        """Plot the 5 basket time series contained in `value_map`."""
        if not isinstance(value_map, dict) or len(value_map) == 0:
            raise ValueError("value_map must be a non-empty dict of pandas Series.")

        cleaned: Dict[str, pd.Series] = {}
        for name, ser in value_map.items():
            if ser is None:
                continue
            s = pd.Series(ser).copy()
            try:
                s.index = pd.to_datetime(s.index)
            except Exception:
                pass
            s = s.sort_index()
            if s.index.duplicated().any():
                s = s[~s.index.duplicated(keep="last")]
            s = s.astype(float)
            if s.empty:
                continue

            lbl = name.replace("_", " ").title() if isinstance(name, str) else str(name)
            cleaned[lbl] = s

        if not cleaned:
            raise ValueError("No non-empty Series found in value_map.")

        df = pd.concat(cleaned, axis=1)

        fig, ax = plt.subplots(figsize=(10, 6))
        df.plot(ax=ax, linewidth=1.6)

        # # -------------------------------------------------------------
        # # Add Basket 1 - Basket 5 spread
        # # -------------------------------------------------------------
        # basket_names = list(df.columns)
        # if len(basket_names) >= 2:
        #     first, last = basket_names[0], basket_names[-1]
        #     spread = df[first] - df[last]
        #     spread_label = f"{first} - {last}"
        #     spread.plot(ax=ax, linewidth=2.0, color="black", linestyle="--", label=spread_label)
        #     ax.legend(title="Basket", fontsize=9)

        # Styling
        ax.set_title("Basket Performance")
        ax.set_xlabel("Date")
        ax.set_ylabel("Portfolio Value")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        return value_map


    # ---------- Top-level API ----------
    def compute_monthly_ic(
        factor_table: pd.DataFrame,  # columns: ['Date','Ticker','factorvalue']
        factor_name,
        rawdata: pd.DataFrame,       # columns: ['Date','Ticker','Open'] 
        ret_horizon: int = 30,        # 1-day forward return
        method: str = "spearman"     # 'spearman' or 'pearson'
    ) -> pd.DataFrame:
        """
        Compute daily cross-sectional Information Coefficient (IC)
        between factorvalue_t and next-day return.
        """
        # 1️⃣ Compute forward returns for each ticker
        px = rawdata.sort_values(["Ticker", "Date"]).copy()
        px["Ret_fwd"] = px.groupby("Ticker")["Open"].pct_change(ret_horizon).shift(-ret_horizon)
        # Alternatively use 'Close' if you want close-to-close prediction

        # 2️⃣ Merge factor values with forward returns
        merged = pd.merge(
            factor_table,
            px[["Date", "Ticker", "Ret_fwd"]],
            on=["Date", "Ticker"],
            how="inner"
        )

        # 3️⃣ Compute IC per date
        def ic_func(df):
            if df["Ret_fwd"].notna().sum() < 5:
                return np.nan
            return df[factor_name].corr(df["Ret_fwd"], method=method)

        ic_series = merged.groupby("Date", group_keys=False).apply(ic_func).rename("IC")

        # 4️⃣ Summary stats
        ic_mean = ic_series.mean(skipna=True)
        ic_std = ic_series.std(ddof=0, skipna=True)
        ic_ir = ic_mean / ic_std if ic_std > 0 else np.nan

        print(f"Mean IC: {ic_mean:.4f},  Std: {ic_std:.4f},  ICIR: {ic_ir:.2f}")

        return pd.DataFrame({"Date": ic_series.index, "IC": ic_series.values})


    def get_return(self, factor_name: str) -> None:
        """Run monthly rebalance and plot basket performance for the factor."""
        ic_table=self.factor_table[['Date','Ticker',factor_name]]
        pivot_table = self._pivot_factor(factor_name)
        df = pivot_table.copy().reset_index()

        # # # identify last date of each month (use next row's month change)
        new_month = df["Date"].dt.to_period("M").ne(df["Date"].shift().dt.to_period("M"))
        out = df[new_month.shift(-1, fill_value=False)]

        w_opt_map, _ = self.implement_optimization(out)

        # # # Use raw Open; consider switching to Adjusted Open to avoid split jumps:
        # # # raw = self.rawdata[['Date','Ticker','Open','Close','Adj Close']].copy()
        # # # raw['AdjOpen'] = raw['Open'] * (raw['Adj Close'] / raw['Close'])
        # # # px = raw.pivot(index='Date', columns='Ticker', values='AdjOpen')
        open_price = self.rawdata[["Date", "Ticker", "Open"]]
        px = open_price.pivot(index="Date", columns="Ticker", values="Open")

        value_map, _ = Backtest.chain_mtd_rebalance(
            w_opt_map, px, num_baskets=5, start_value=1.0
        )
        Backtest.chart_the_graph(value_map)
        Backtest.compute_monthly_ic(ic_table,factor_name,open_price)

        return value_map,w_opt_map
      