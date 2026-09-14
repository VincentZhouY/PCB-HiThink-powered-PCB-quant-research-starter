"""
hithink-quant starter —— 端到端量化脚手架
============================================

链路: 数据 -> 因子 -> 信号 -> 组合权重 -> 风控 -> 回测 -> 绩效

这是给你(量化开发者)的一个**可运行起点**,不是成品策略。
每一层都是独立函数,你可以逐层替换掉,而不用动其他部分。

依赖: duckdb, pandas, numpy
数据: hithink-finance 本地 DuckDB —— 先在仓库根跑 `python python/bootstrap.py` 建库

用法:
    python quant_starter.py --db data/market.duckdb
    python quant_starter.py --db data/market.duckdb --start 2021-01-01 --top-n 80

对应官方文档:
    python/toolkit/marketdb/README.md   (本地行情 / SDK / SQL 视图)
    python/toolkit/fuyao/README.md      (远端行情、财报、估值)
"""

from __future__ import annotations


import argparse
import os
import sys

import duckdb
import numpy as np
import pandas as pd

# Windows 控制台按 UTF-8 输出;若终端仍乱码,先在终端执行 chcp 65001
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TRADING_DAYS = 243.0
RF = 0.02  # 无风险利率,用于夏普

# 复权口径 -> 官方 DuckDB 视图(见 python/toolkit/marketdb/README.md)
ADJUST_VIEWS = {
    "forward": "v_daily_qfq",
    "backward": "v_daily_hfq",
    "none": "v_daily",
}


# ══════════════════════════════════════════════════════════════════════
# 1. 数据访问层
# ══════════════════════════════════════════════════════════════════════
class DataHub:
    """本地 DuckDB 的只读封装。

    这里直接走 DuckDB 视图,行为最可控。官方也提供等价的 Python SDK:

        from marketdb import MarketDB
        with MarketDB.open("data/market.duckdb") as db:
            df = db.get_daily(["600519.SH"], start="2025-01-01", adjust="forward")

    两者读的是同一套视图,选哪个都行;想用 SDK 就把下面三个方法换掉。
    """

    def __init__(self, db_path: str):
        if not os.path.exists(db_path):
            raise FileNotFoundError(
                f"本地库不存在: {db_path}\n"
                f"请先在 monorepo 根执行:  python python/bootstrap.py"
            )
        self.con = duckdb.connect(db_path, read_only=True)
        self.db_path = db_path

    def close(self) -> None:
        self.con.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ---- 原始 SQL -------------------------------------------------
    def sql(self, query: str, params: list | None = None) -> pd.DataFrame:
        return self.con.execute(query, params or []).df()

    # ---- 日线(长表) ----------------------------------------------
    def daily(
        self,
        codes: list[str] | str,
        start: str | None = None,
        end: str | None = None,
        adjust: str = "forward",
    ) -> pd.DataFrame:
        if isinstance(codes, str):
            codes = [codes]
        view = ADJUST_VIEWS[adjust]
        where = [f"thscode IN ({','.join(['?'] * len(codes))})"]
        params: list = list(codes)
        if start:
            where.append("date >= ?")
            params.append(start)
        if end:
            where.append("date <= ?")
            params.append(end)
        q = (
            f"SELECT thscode, date, open, high, low, close, volume, turnover AS amount "
            f"FROM {view} WHERE {' AND '.join(where)} ORDER BY date, thscode"
        )
        return self.sql(q, params)

    # ---- 价格面板 -> 宽表 ----------------------------------------
    def price_panels(
        self, start: str, end: str, codes: list[str] | None = None, adjust: str = "forward"
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """返回 (close 宽表, amount 宽表),index=date, columns=thscode。

        宽表是因子计算的通用形态:每个因子都是 date x thscode 的矩阵。
        """
        view = ADJUST_VIEWS[adjust]
        where = ["date BETWEEN ? AND ?"]
        params: list = [start, end]
        if codes:
            where.append(f"thscode IN ({','.join(['?'] * len(codes))})")
            params.extend(codes)
        df = self.sql(
            f"SELECT thscode, date, close, turnover AS amount FROM {view} "
            f"WHERE {' AND '.join(where)}",
            params,
        )
        close = df.pivot(index="date", columns="thscode", values="close").sort_index()
        amount = df.pivot(index="date", columns="thscode", values="amount").sort_index()
        return close, amount

    # ---- 可交易标的池 --------------------------------------------
    def liquidity_universe(
        self, as_of: str, lookback_days: int = 243, top_n: int = 80
    ) -> list[str]:
        """按 as_of 之前 lookback_days 个交易日的平均成交额选流动性 top N。

        关键:`as_of` 之后的数据一律不碰,避免前视偏差。
        """
        df = self.sql(
            """
            WITH win AS (
                SELECT thscode, date, turnover AS amount,
                       ROW_NUMBER() OVER (PARTITION BY thscode ORDER BY date DESC) AS rn
                FROM v_daily_qfq
                WHERE date <= ?
            )
            SELECT thscode, AVG(amount) AS adv
            FROM win WHERE rn <= ?
            GROUP BY thscode
            HAVING COUNT(*) >= ?
            ORDER BY adv DESC
            LIMIT ?
            """,
            [as_of, lookback_days, int(lookback_days * 0.8), top_n],
        )
        return df["thscode"].tolist()


# ══════════════════════════════════════════════════════════════════════
# 2. 因子层 —— 每个因子 = 一个 date x thscode 的矩阵
# ══════════════════════════════════════════════════════════════════════
def factor_momentum(close: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """N 日动量(累计收益)。"""
    return close.pct_change(window, fill_method=None)


def factor_reversal(close: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """短周期反转(取负号后越大越强)。"""
    return -close.pct_change(window, fill_method=None)


def factor_volatility(close: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """N 日已实现波动率(年化)。"""
    return close.pct_change(fill_method=None).rolling(window).std() * np.sqrt(TRADING_DAYS)


def factor_ma_ratio(close: pd.DataFrame, fast: int = 5, slow: int = 20) -> pd.DataFrame:
    """快慢均线比 —— >1 表示短期强于长期。"""
    ma_f = close.rolling(fast).mean()
    ma_s = close.rolling(slow).mean()
    return ma_f / ma_s


def factor_ic(factor: pd.DataFrame, close: pd.DataFrame, forward: int = 5) -> pd.Series:
    """因子 IC(与未来 forward 日收益的横截面秩相关)。

    这是判断因子有没有用的第一道关:IC 均值长期 >0.03 且 IR 稳定,才算有信号。
    注意用的是 **未来** 收益,只用于评估(研究阶段),绝不能进回测的持仓逻辑。
    """
    fwd_ret = close.shift(-forward) / close - 1
    common = factor.index.intersection(fwd_ret.index)
    f, r = factor.reindex(common), fwd_ret.reindex(common)
    # 秩相关 = 先转秩再求 pearson(这样不依赖 scipy)
    return f.rank(axis=1).corrwith(r.rank(axis=1), axis=1)


# ══════════════════════════════════════════════════════════════════════
# 3. 信号层 —— 因子 -> 0/1 持仓矩阵
# ══════════════════════════════════════════════════════════════════════
def signal_threshold(factor: pd.DataFrame, threshold: float = 1.0) -> pd.DataFrame:
    """阈值法:因子超过阈值即持有。"""
    return (factor > threshold).astype(float)


def signal_cross_section_rank(
    factor: pd.DataFrame, top_pct: float = 0.2, min_names: int = 10
) -> pd.DataFrame:
    """截面排序法:每日取因子排名前 top_pct 的标的。这是多因子选股最常用的形态。"""
    rank = factor.rank(axis=1, ascending=False, pct=True)
    n_valid = factor.notna().sum(axis=1)
    raw = (rank <= top_pct).astype(float)
    # 有效标的太少时当天不做(避免在极少数票上押重注)
    raw[n_valid < min_names] = 0.0
    return raw


def signal_ma_cross(close: pd.DataFrame, fast: int = 5, slow: int = 20) -> pd.DataFrame:
    """经典双均线:金叉持有、死叉空仓。"""
    return (close.rolling(fast).mean() > close.rolling(slow).mean()).astype(float)


# ══════════════════════════════════════════════════════════════════════
# 4. 组合层 —— 信号 -> 目标权重(每行合计 <= 1,剩余即现金)
# ══════════════════════════════════════════════════════════════════════
def weight_equal(signal: pd.DataFrame) -> pd.DataFrame:
    """等权:信号为 1 的标的平均分配。"""
    n = signal.sum(axis=1).replace(0, np.nan)
    return signal.div(n, axis=0).fillna(0.0)


def weight_inverse_vol(
    signal: pd.DataFrame, vol: pd.DataFrame, floor: float = 1e-4
) -> pd.DataFrame:
    """波动率倒数加权 —— 低波动标的拿更多权重。"""
    inv = signal / vol.clip(lower=floor)
    inv = inv.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    total = inv.sum(axis=1).replace(0, np.nan)
    return inv.div(total, axis=0).fillna(0.0)


def weight_risk_parity(
    signal: pd.DataFrame, vol: pd.DataFrame, iters: int = 30
) -> pd.DataFrame:
    """简化风险平价:迭代调整使各持仓风险贡献趋同。

    严格版需要协方差矩阵;这里用对角近似(只看个体波动率),速度快、够用。
    """
    w = weight_inverse_vol(signal, vol)
    for _ in range(iters):
        rc = (w * vol.fillna(0.0)).where(signal > 0)        # 风险贡献近似
        active = signal > 0
        target = rc.sum(axis=1) / active.sum(axis=1).replace(0, np.nan)
        target = target.fillna(0.0).to_numpy()[:, None]
        adj = np.clip(target / rc.to_numpy(), 0.2, 5.0)
        adj = np.nan_to_num(adj, nan=1.0, posinf=5.0, neginf=0.2)
        w = w * pd.DataFrame(adj, index=w.index, columns=w.columns)
        w = w.where(active, 0.0)
        total = w.sum(axis=1).replace(0, np.nan)
        w = w.div(total, axis=0).fillna(0.0)
    return w


# ══════════════════════════════════════════════════════════════════════
# 5. 风控层 —— 作用在权重或净值上
# ══════════════════════════════════════════════════════════════════════
def risk_max_weight(weights: pd.DataFrame, cap: float = 0.10) -> pd.DataFrame:
    """单标的上限:超过 cap 的部分直接砍掉。

    注意:砍掉后总仓位会下降(不补到其他标的上)。想做「砍完再分配」,
    在下面接一层 weight_equal() / weight_inverse_vol() 重新归一即可。
    """
    return weights.clip(upper=cap)


def risk_vol_target(
    weights: pd.DataFrame,
    port_ret: pd.Series,
    target_vol: float = 0.15,
    window: int = 60,
) -> pd.DataFrame:
    """波动率目标:组合近期波动高于目标时整体降杠杆。"""
    realized = port_ret.rolling(window).std() * np.sqrt(TRADING_DAYS)
    lev = (target_vol / realized.replace(0, np.nan)).clip(upper=1.0)
    return weights.mul(lev.shift(1).fillna(1.0), axis=0)


def risk_turnover_buffer(
    weights: pd.DataFrame, threshold: float = 0.005
) -> pd.DataFrame:
    """调仓缓冲:与**上一期实际执行的权重**相比,变动小于 threshold 的就不动。

    必须在时间轴上递推,不能用 shift(1) 近似 —— 被挡住的调仓不会真的发生,
    所以后续的比较基准是「实际持仓」而不是「上期目标」。
    """
    arr = weights.to_numpy(dtype=float)
    out = np.zeros_like(arr)
    prev = np.zeros(arr.shape[1])
    for i in range(arr.shape[0]):
        cur = arr[i]
        delta = np.abs(cur - prev)
        out[i] = np.where(delta < threshold, prev, cur)
        prev = out[i]
    return pd.DataFrame(out, index=weights.index, columns=weights.columns)


# ══════════════════════════════════════════════════════════════════════
# 6. 回测引擎
# ══════════════════════════════════════════════════════════════════════
def backtest(
    weights: pd.DataFrame,
    close: pd.DataFrame,
    cost: float = 0.0003,
    lag: int = 1,
) -> dict:
    """按目标权重回测。

    lag=1 表示 T 日收盘算出的权重,T+1 日才生效 —— **这是防前视偏差的关键**,
    不要改成 0。
    """
    ret = close.pct_change(fill_method=None)
    idx = weights.index.intersection(ret.index)
    w = weights.reindex(idx).fillna(0.0)
    r = ret.reindex(idx).fillna(0.0)

    pos = w.shift(lag).fillna(0.0)                       # 权重生效日
    gross = (pos * r).sum(axis=1)
    turnover = (pos - pos.shift(1).fillna(0.0)).abs().sum(axis=1)
    net = gross - turnover * cost

    return {
        "nav": (1 + net).cumprod(),
        "returns": net,
        "gross_returns": gross,
        "turnover": turnover,
        "exposure": pos.sum(axis=1),
    }


def performance(returns: pd.Series, name: str = "strategy") -> dict:
    """绩效指标。"""
    years = len(returns) / TRADING_DAYS
    nav = (1 + returns).cumprod()
    total = nav.iloc[-1] - 1
    ann = (1 + total) ** (1 / years) - 1 if years > 0 else np.nan
    vol = returns.std() * np.sqrt(TRADING_DAYS)
    dd = (nav / nav.cummax() - 1).min()
    downside = returns[returns < 0].std() * np.sqrt(TRADING_DAYS)
    return {
        "策略": name,
        "累计收益": f"{total:+.2%}",
        "年化收益": f"{ann:+.2%}",
        "年化波动": f"{vol:.2%}",
        "夏普": f"{(ann - RF) / vol:.2f}" if vol > 0 else "-",
        "索提诺": f"{(ann - RF) / downside:.2f}" if downside > 0 else "-",
        "最大回撤": f"{dd:.2%}",
        "Calmar": f"{ann / abs(dd):.2f}" if dd < 0 else "-",
        "日胜率": f"{(returns > 0).mean():.1%}",
    }


# ══════════════════════════════════════════════════════════════════════
# 7. 主流程 —— 一个 baseline,拿去替换
# ══════════════════════════════════════════════════════════════════════
def main() -> int:
    ap = argparse.ArgumentParser(description="hithink-quant starter baseline")
    ap.add_argument(
        "--db",
        required=True,
        help=(
            "本地 DuckDB 数据库文件路径。"
            "例如：data/market.duckdb 或 C:\\path\\to\\market.duckdb"
        ),
    )


    ap.add_argument("--start", default="2021-01-01")
    ap.add_argument("--end", default=None, help="默认取库内最新日期")
    ap.add_argument("--top-n", type=int, default=80, help="流动性股票池规模")
    ap.add_argument("--cost", type=float, default=0.0003, help="单边成本")
    args = ap.parse_args()

    with DataHub(args.db) as hub:
        end = args.end or hub.sql(
            "SELECT CAST(MAX(date) AS VARCHAR) AS d FROM v_daily_qfq"
        )["d"].iloc[0]
        print(f"本地库      : {args.db}")
        print(f"回测区间    : {args.start} ~ {end}\n")

        # --- 股票池:只用 start 之前的数据选,不在回测期内挑票 ---
        pre = hub.sql(
            "SELECT CAST(MAX(date) AS VARCHAR) AS d FROM v_daily_qfq WHERE date < ?",
            [args.start],
        )["d"].iloc[0]
        universe = hub.liquidity_universe(
            as_of=pre, lookback_days=243, top_n=args.top_n
        )
        print(f"股票池      : {len(universe)} 只(按 {pre} 前 243 日成交额选取)")

        # --- 取价格(多取 90 个交易日做均线/波动率预热) ---
        lookback_start = hub.sql(
            "SELECT CAST(MIN(date) AS VARCHAR) AS d FROM ("
            "  SELECT DISTINCT date FROM v_daily_qfq "
            "  WHERE date < ? ORDER BY date DESC LIMIT 90)",
            [args.start],
        )["d"].iloc[0]
        close, _amount = hub.price_panels(lookback_start, end, codes=universe)
        close = close.ffill(limit=5)

    # --- 示例策略:20 日动量截面排序取前 20%,叠波动率目标 + 单票上限 + 调仓缓冲 ---
    mom = factor_momentum(close, window=20)
    sig = signal_cross_section_rank(mom, top_pct=0.2, min_names=10)
    vol = factor_volatility(close, window=20)
    w = weight_inverse_vol(sig, vol)

    bt_pre = backtest(w, close, cost=args.cost)          # 先跑一遍,给波动率目标提供组合收益
    w = risk_vol_target(w, bt_pre["returns"], target_vol=0.15)
    w = risk_max_weight(w, cap=0.10)
    w = risk_turnover_buffer(w, threshold=0.01)          # 小变动不动手,压换手

    bt = backtest(w, close, cost=args.cost)
    bench = close.pct_change(fill_method=None).mean(axis=1).fillna(0.0)

    mask = bt["returns"].index >= args.start
    res = pd.DataFrame([
        performance(bt["returns"][mask], "动量+波动率加权+风控"),
        performance(bench[mask], "等权买入持有(基准)"),
    ])

    r = bt["returns"][mask]
    print(f"实际统计区间: {r.index.min().date()} ~ {r.index.max().date()}")
    print(f"平均总仓位  : {bt['exposure'][mask].mean():.1%}   "
          f"年化换手: {bt['turnover'][mask].sum() / (mask.sum() / TRADING_DAYS):.1f}x\n")
    print(res.to_string(index=False))

    print("\n分年度收益(策略):")
    for yr, v in r.groupby(r.index.year).apply(lambda s: (1 + s).prod() - 1).items():
        print(f"  {yr}  {v:+7.2%}  {'#' * int(abs(v) * 50)}")

    # --- 因子体检:IC 均值与 IR(研究阶段用,不能进持仓逻辑) ---
    ic = factor_ic(mom, close, forward=5).loc[r.index.min():]
    if len(ic.dropna()) > 20:
        print(f"\n因子体检(20 日动量,5 日前瞻 IC):")
        print(f"  IC 均值 {ic.mean():+.4f}   IC 标准差 {ic.std():.4f}   "
              f"IR {ic.mean() / ic.std():+.3f}   胜率 {(ic > 0).mean():.1%}")
        print(f"  (经验:IC 均值长期 >0.03 且 IR 稳定才值得继续;否则换因子)")

    print("\n>>> 这是 baseline,不是结论。接下来该你替换的:")
    print("    - 因子层   factor_*()      : 加基本面因子、量价因子、另类数据")
    print("    - 信号层   signal_*()      : 换成 IC 加权、机器学习打分、事件驱动")
    print("    - 组合层   weight_*()      : 换成均值方差、Black-Litterman、约束优化")
    print("    - 风控层   risk_*()        : 加行业中性、回撤止损、流动性约束")
    print("    每次改动都跑一遍,看夏普 / 回撤 / 换手是变好还是变差。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
