"""
因子研究 + 滚动 IC 拟合示例

数据来源：同花顺 hithink-finance 本地 DuckDB
价格口径：v_daily_qfq（前复权日线）
注意：仅用于量化研究，不构成投资建议。

示例：
python factor_research.py
python factor_research.py --start 2021-01-01 --top-n 80

如需指定其他数据库：
DEFAULT_DB_PATH = "data/market.duckdb"
"""


from __future__ import annotations

import argparse
import os

import duckdb
import numpy as np
import pandas as pd


TRADING_DAYS = 243.0
DEFAULT_DB_PATH = "data/market.duckdb"
DEFAULT_OUTPUT_DIR = "research_output"



def select_universe(
    con: duckdb.DuckDBPyConnection,
    as_of: str,
    top_n: int,
    lookback_days: int = 243,
) -> list[str]:
    """只使用回测开始日前的信息，按历史平均成交额选择股票池。"""
    df = con.execute(
        """
        WITH win AS (
            SELECT
                thscode,
                turnover,
                ROW_NUMBER() OVER (
                    PARTITION BY thscode
                    ORDER BY date DESC
                ) AS rn
            FROM v_daily_qfq
            WHERE date < ?
        )
        SELECT thscode, AVG(turnover) AS avg_turnover
        FROM win
        WHERE rn <= ?
        GROUP BY thscode
        HAVING COUNT(*) >= ?
        ORDER BY avg_turnover DESC
        LIMIT ?
        """,
        [as_of, lookback_days, int(lookback_days * 0.8), top_n],
    ).df()
    return df["thscode"].tolist()


def load_panels(
    con: duckdb.DuckDBPyConnection,
    codes: list[str],
    end: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """加载选定股票池的历史前复权收盘价和成交额。"""
    placeholders = ",".join(["?"] * len(codes))
    where_end = "AND date <= ?" if end else ""
    params: list = list(codes) + ([end] if end else [])

    df = con.execute(
        f"""
        SELECT thscode, date, close, turnover
        FROM v_daily_qfq
        WHERE thscode IN ({placeholders})
        {where_end}
        ORDER BY date, thscode
        """,
        params,
    ).df()

    df["date"] = pd.to_datetime(df["date"])
    close = df.pivot(index="date", columns="thscode", values="close").sort_index()
    turnover = df.pivot(index="date", columns="thscode", values="turnover").sort_index()

    # 最多向前填充 5 日，避免长期停牌或缺失价格被伪造。
    return close.ffill(limit=5), turnover


def make_factors(
    close: pd.DataFrame,
    turnover: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """构建因子。约定：因子值越大，预期未来收益越高。"""
    daily_ret = close.pct_change(fill_method=None)
    vol_20 = daily_ret.rolling(20).std() * np.sqrt(TRADING_DAYS)

    factors = {
        # 短周期、20 日反转
        "reversal_5": -close.pct_change(5, fill_method=None),
        "reversal_20": -close.pct_change(20, fill_method=None),

        # 中期动量
        "momentum_60": close.pct_change(60, fill_method=None),

        # 低波动效应：低波动的因子值更高
        "low_vol_20": -vol_20,

        # 相对放量：相对过去 20 日均量的放量程度
        "volume_ratio_20": turnover / turnover.rolling(20).mean() - 1.0,

        # Amihud 流动性：数值越小越流动，因此取负号
        "liquidity_amihud_20": -(
            daily_ret.abs() / turnover.replace(0, np.nan)
        ).rolling(20).mean(),
    }
    return factors


def factor_ic(
    factor: pd.DataFrame,
    forward_returns: pd.DataFrame,
) -> pd.Series:
    """每日 Spearman 秩 IC。"""
    f_rank = factor.rank(axis=1, pct=True)
    r_rank = forward_returns.rank(axis=1, pct=True)
    return f_rank.corrwith(r_rank, axis=1)


def factor_summary(
    name: str,
    factor: pd.DataFrame,
    forward_returns: pd.DataFrame,
    start: str,
) -> tuple[dict, pd.Series]:
    """输出单因子 IC、IR、胜率和多空分层收益。"""
    ic = factor_ic(factor, forward_returns)
    ic = ic.loc[ic.index >= pd.Timestamp(start)].dropna()

    rank = factor.rank(axis=1, pct=True)
    top_ret = forward_returns.where(rank >= 0.8).mean(axis=1)
    bottom_ret = forward_returns.where(rank <= 0.2).mean(axis=1)
    long_short = (top_ret - bottom_ret).loc[ic.index].dropna()

    return {
        "因子": name,
        "IC均值": ic.mean(),
        "IC标准差": ic.std(),
        "IR": ic.mean() / ic.std() if ic.std() > 0 else np.nan,
        "IC胜率": (ic > 0).mean(),
        "Top-Bottom均值": long_short.mean(),
        "Top-Bottom胜率": (long_short > 0).mean(),
        "有效IC天数": len(ic),
    }, ic


def rolling_ic_weighted_score(
    factors: dict[str, pd.DataFrame],
    forward_returns: pd.DataFrame,
    lookback: int = 252,
    ic_delay: int = 6,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    使用过去已实现的 IC 来给因子加权。

    ic_delay=6：
    因为使用 5 日前瞻收益评估 IC，当日并不知道刚产生的 IC，
    必须延迟至少 5 个交易日再用于拟合，避免前视偏差。
    """
    ranked_factors = {
        name: value.rank(axis=1, pct=True)
        for name, value in factors.items()
    }

    ic_map = pd.DataFrame({
        name: factor_ic(value, forward_returns)
        for name, value in factors.items()
    })

    rolling_mean = ic_map.rolling(lookback, min_periods=80).mean().shift(ic_delay)
    rolling_std = ic_map.rolling(lookback, min_periods=80).std().shift(ic_delay)
    raw_weight = rolling_mean / rolling_std.replace(0, np.nan)

    # 每日按绝对值归一化。IC 为负的因子自动反向使用。
    weights = raw_weight.div(raw_weight.abs().sum(axis=1), axis=0).fillna(0.0)

    score = pd.DataFrame(0.0, index=forward_returns.index, columns=forward_returns.columns)
    for name, ranked in ranked_factors.items():
        score = score.add(ranked.mul(weights[name], axis=0), fill_value=0.0)

    return score, weights


def backtest_top_quantile(
    score: pd.DataFrame,
    close: pd.DataFrame,
    start: str,
    top_pct: float = 0.2,
    cost: float = 0.0003,
) -> dict:
    """每日选得分前 top_pct，等权；T 日信号、T+1 日生效。"""
    daily_ret = close.pct_change(fill_method=None).fillna(0.0)
    rank = score.rank(axis=1, ascending=False, pct=True)

    signal = (rank <= top_pct).astype(float)
    n = signal.sum(axis=1).replace(0, np.nan)
    weights = signal.div(n, axis=0).fillna(0.0)

    # 关键：使用下一日生效的仓位，禁止 lag=0。
    position = weights.shift(1).fillna(0.0)
    gross_ret = (position * daily_ret).sum(axis=1)
    turnover = position.sub(position.shift(1).fillna(0.0)).abs().sum(axis=1)
    net_ret = gross_ret - turnover * cost

    net_ret = net_ret.loc[net_ret.index >= pd.Timestamp(start)]
    turnover = turnover.loc[net_ret.index]
    nav = (1.0 + net_ret).cumprod()

    years = len(net_ret) / TRADING_DAYS
    annual_return = nav.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan
    annual_vol = net_ret.std() * np.sqrt(TRADING_DAYS)
    drawdown = (nav / nav.cummax() - 1.0).min()

    return {
        "returns": net_ret,
        "nav": nav,
        "年化收益": annual_return,
        "年化波动": annual_vol,
        "夏普": (annual_return - 0.02) / annual_vol if annual_vol > 0 else np.nan,
        "最大回撤": drawdown,
        "年化换手": turnover.sum() / years if years > 0 else np.nan,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="量价因子研究与滚动 IC 拟合")
    parser.add_argument(
        "--db",
        default=DEFAULT_DB_PATH,
        help=(

        ),
    )

    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=(

        ),
    )

    parser.add_argument(
        "--db",
        default=DEFAULT_DB_PATH,
        help=(
            "DuckDB 数据库路径。默认：data/market.duckdb。"
            "数据库不随 GitHub 仓库提供，请自行构建或指定本地路径。"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="研究结果输出目录，默认：research_output。",
    )

    parser.add_argument(
        "--start",
        default="2021-01-01",
        help="回测/统计起始日。",
    )

    parser.add_argument(
        "--end",
        default=None,
        help="结束日，默认数据库最新日期。",
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=80,
        help="股票池规模。",
    )

    parser.add_argument(
        "--forward",
        type=int,
        default=5,
        help="IC 的前瞻交易日数。",
    )

    parser.add_argument(
        "--cost",
        type=float,
        default=0.0003,
        help="单边交易成本。",
    )

    args = parser.parse_args()


    if not os.path.exists(args.db):
        raise FileNotFoundError(f"找不到数据库：{args.db}")

    with duckdb.connect(args.db, read_only=True) as con:
        if args.end is None:
            args.end = con.execute(
                "SELECT CAST(MAX(date) AS VARCHAR) FROM v_daily_qfq"
            ).fetchone()[0]

        universe = select_universe(con, args.start, args.top_n)
        close, turnover = load_panels(con, universe, args.end)

    forward_returns = close.shift(-args.forward) / close - 1.0
    factors = make_factors(close, turnover)

    print(f"\n数据口径：v_daily_qfq（前复权）")
    print(f"股票池：{len(universe)} 只；按 {args.start} 前 243 日平均成交额固定选取")
    print(f"研究区间：{args.start} ~ {args.end}")
    print(f"IC 前瞻期：{args.forward} 个交易日\n")

    rows = []
    for name, factor in factors.items():
        summary, _ = factor_summary(name, factor, forward_returns, args.start)
        rows.append(summary)

    report = pd.DataFrame(rows).sort_values("IR", ascending=False)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print("=== 单因子体检（按 IR 排序）===")
    print(report.to_string(index=False))

    score, factor_weights = rolling_ic_weighted_score(
        factors,
        forward_returns,
        lookback=252,
        ic_delay=args.forward + 1,
    )
    bt = backtest_top_quantile(
        score,
        close,
        start=args.start,
        top_pct=0.2,
        cost=args.cost,
    )

    print("\n=== 滚动 IC 加权组合（样本内历史滚动拟合）===")
    print(f"年化收益：{bt['年化收益']:+.2%}")
    print(f"年化波动：{bt['年化波动']:.2%}")
    print(f"夏普：{bt['夏普']:.2f}")
    print(f"最大回撤：{bt['最大回撤']:.2%}")
    print(f"年化换手：{bt['年化换手']:.1f}x")

    output_dir = os.path.abspath(
        os.path.expanduser(args.output_dir)
    )
    os.makedirs(output_dir, exist_ok=True)

    report.to_csv(f"{output_dir}/factor_report.csv", index=False, encoding="utf-8-sig")
    factor_weights.to_csv(f"{output_dir}/rolling_factor_weights.csv", encoding="utf-8-sig")
    bt["nav"].rename("nav").to_csv(f"{output_dir}/combo_nav.csv", encoding="utf-8-sig")

    print(f"\n结果文件已保存到：{os.path.abspath(output_dir)}")
    print("  - factor_report.csv：单因子 IC / IR / 分层收益")
    print("  - rolling_factor_weights.csv：每个交易日的历史滚动拟合权重")
    print("  - combo_nav.csv：组合净值")
    print("\n提示：这是研究工具。不要根据全样本结果直接实盘，应继续做样本外验证。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())