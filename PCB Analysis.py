"""
PCB / 电子产业链：最强个股识别 + 双标签横截面量化模型
============================================================

核心目标：
1. 不再强制每周选 Top N；
2. 不再因为“同行排名第一”就机械选择股票；
3. 首先识别个股自身是否真正处于强势趋势；
4. 同时判断该股未来是否更可能上涨（绝对收益模型）；
5. 同时判断该股是否更可能跑赢 PCB 同业（相对收益模型）；
6. 每周仅输出唯一一只最高置信度股票；
7. 如果没有股票满足全部强势条件，则输出“本周无高置信度标的”。

数据：
- 本地 DuckDB：v_daily_qfq
- 字段：thscode, date, open, high, low, close, volume, turnover

信号与回测：
- 信号在每周最后一个交易日收盘后生成；
- 下一交易日开盘买入；
- 下一周信号日后的下一交易日开盘卖出；
- 无信号时空仓，不强制交易；
- 所有结果仅用于量化研究，不构成投资建议。
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


EPS = 1e-12

PCB_UNIVERSE = [
    "600183",  # 生益科技
    "002384",  # 东山精密
    "002916",  # 深南电路
    "002463",  # 沪电股份
    "300476",  # 胜宏科技
    "002938",  # 鹏鼎控股
    "600176",
    "301377",
    "000657",
    "603256",
    "301526",
    "603228",  # 景旺电子
    "688183",
    "000988",
    "301217",
    "002080",
    "002636",  # 金安国纪
]


# ============================================================
# 通用工具函数
# ============================================================

def normalize_code(value) -> str | None:
    """从证券代码字符串中提取六码代码。"""
    if pd.isna(value):
        return None

    matches = re.findall(r"\d{6}", str(value))
    return matches[-1] if matches else None


def board_type(code: str) -> str:
    """按代码识别主板、创业板、科创板。"""
    code = str(code)

    if code.startswith(("300", "301")):
        return "CHINEXT"

    if code.startswith("688"):
        return "STAR"

    return "MAIN"


def safe_pct_change(
    series: pd.Series,
    periods: int = 1,
) -> pd.Series:
    """不使用 pandas 默认缺失值填充的涨跌幅计算。"""
    return series.pct_change(
        periods=periods,
        fill_method=None,
    )


def rolling_zscore(
    series: pd.Series,
    window: int,
    min_periods: int,
) -> pd.Series:
    """时间序列滚动 Z 分数。"""
    mean_value = series.rolling(
        window=window,
        min_periods=min_periods,
    ).mean()

    std_value = series.rolling(
        window=window,
        min_periods=min_periods,
    ).std()

    return (series - mean_value) / (std_value + EPS)


def weekly_last_trading_dates(
    dates: pd.Series,
) -> pd.DatetimeIndex:
    """返回每周最后一个实际交易日。"""
    calendar = pd.DataFrame(
        {
            "date": pd.Series(
                pd.to_datetime(dates).unique()
            )
        }
    ).sort_values("date")

    calendar["week"] = calendar["date"].dt.to_period("W-FRI")

    return pd.DatetimeIndex(
        calendar.groupby("week")["date"].max().tolist()
    )


def format_pct(value) -> str:
    """安全格式化百分比。"""
    if pd.isna(value):
        return "N/A"

    return f"{value:.2%}"


def format_num(value, digits: int = 4) -> str:
    """安全格式化数值。"""
    if pd.isna(value):
        return "N/A"

    return f"{value:.{digits}f}"


# ============================================================
# 数据读取
# ============================================================

def check_database_view(
    con: duckdb.DuckDBPyConnection,
) -> None:
    """检查 v_daily_qfq 是否存在。"""
    result = con.execute(
        """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_name = 'v_daily_qfq'
        """
    ).fetchall()

    if not result:
        raise RuntimeError(
            "数据库中未找到 v_daily_qfq 视图。"
        )


def load_universe_daily(
    con: duckdb.DuckDBPyConnection,
    codes: list[str],
    start_date: str,
    end_date: str | None,
) -> pd.DataFrame:
    """
    读取股票池日线数据。

    正式起点之前额外读取约 500 个自然日，
    以便计算均线、动量、成交额与波动率等因子。
    """
    warmup_start = (
        pd.Timestamp(start_date)
        - pd.Timedelta(days=500)
    ).strftime("%Y-%m-%d")

    placeholders = ", ".join(["?"] * len(codes))

    conditions = [
        "date >= ?",
        (
            "regexp_extract("
            "CAST(thscode AS VARCHAR), "
            "'[0-9]{6}'"
            f") IN ({placeholders})"
        ),
    ]

    params = [warmup_start] + list(codes)

    if end_date is not None:
        conditions.append("date <= ?")
        params.append(end_date)

    sql = f"""
        SELECT
            thscode,
            date,
            open,
            high,
            low,
            close,
            volume,
            turnover
        FROM v_daily_qfq
        WHERE {" AND ".join(conditions)}
        ORDER BY date, thscode
    """

    print("\n正在读取 PCB / 电子产业链股票池日线数据，请稍候...")

    df = con.execute(sql, params).fetchdf()

    if df.empty:
        raise RuntimeError(
            "读取结果为空，请检查数据库路径、股票代码或 thscode 格式。"
        )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

    df["code"] = df["thscode"].map(normalize_code)

    numeric_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "turnover",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    df = df.dropna(
        subset=[
            "code",
            "date",
            "open",
            "high",
            "low",
            "close",
        ]
    ).copy()

    df = df[
        (df["open"] > 0)
        & (df["high"] > 0)
        & (df["low"] > 0)
        & (df["close"] > 0)
    ].copy()

    df["volume"] = df["volume"].fillna(0.0)
    df["turnover"] = df["turnover"].fillna(0.0)
    df["board"] = df["code"].map(board_type)

    df = (
        df.sort_values(["code", "date"])
        .drop_duplicates(["code", "date"], keep="last")
        .reset_index(drop=True)
    )

    found_codes = sorted(df["code"].unique())
    missing_codes = sorted(set(codes) - set(found_codes))

    print(
        f"读取完成：{len(df):,} 行；"
        f"股票数：{df['code'].nunique()}；"
        f"日期：{df['date'].min().date()} 至 "
        f"{df['date'].max().date()}。"
    )

    if missing_codes:
        print("\n警告：以下代码未找到：")
        print(", ".join(missing_codes))

    coverage = (
        df.groupby("code", as_index=False)
        .agg(
            thscode=("thscode", "first"),
            board=("board", "first"),
            trading_days=("date", "count"),
            first_date=("date", "min"),
            last_date=("date", "max"),
        )
        .sort_values("code")
    )

    print("\n股票池覆盖情况：")
    print(coverage.to_string(index=False))

    return df


# ============================================================
# 单股票因子
# ============================================================

def build_single_stock_factors(
    stock_df: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    """
    构建单股票技术、趋势、量价、流动性因子。

    标签交易定义：

    fwd_return =
    open[t + horizon + 1] / open[t + 1] - 1

    即信号日收盘后，下一交易日开盘买入，
    horizon 个交易日后开盘卖出。
    """
    x = (
        stock_df.sort_values("date")
        .drop_duplicates("date")
        .copy()
    )

    # 基础收益与标签
    x["ret_1d"] = safe_pct_change(x["close"], 1)
    x["ret_5d"] = safe_pct_change(x["close"], 5)

    x["entry_open_next"] = x["open"].shift(-1)
    x["exit_open_future"] = x["open"].shift(
        -(horizon + 1)
    )

    x["fwd_return"] = (
        x["exit_open_future"]
        / (x["entry_open_next"] + EPS)
        - 1
    )

    # 趋势、均线与动量
    x["momentum_5"] = safe_pct_change(x["close"], 5)
    x["momentum_20"] = safe_pct_change(x["close"], 20)
    x["momentum_60"] = safe_pct_change(x["close"], 60)

    x["ma20"] = x["close"].rolling(
        window=20,
        min_periods=20,
    ).mean()

    x["ma60"] = x["close"].rolling(
        window=60,
        min_periods=40,
    ).mean()

    x["max_close_60"] = x["close"].rolling(
        window=60,
        min_periods=40,
    ).max()

    x["min_close_20"] = x["close"].rolling(
        window=20,
        min_periods=15,
    ).min()

    x["close_ma20_gap"] = (
        x["close"] / (x["ma20"] + EPS) - 1
    )

    x["close_ma60_gap"] = (
        x["close"] / (x["ma60"] + EPS) - 1
    )

    x["breakout_60"] = (
        x["close"] / (x["max_close_60"] + EPS) - 1
    )

    x["position_20"] = (
        (x["close"] - x["min_close_20"])
        / (
            x["max_close_60"]
            - x["min_close_20"]
            + EPS
        )
    )

    # 成交额与量价
    log_turnover = np.log1p(x["turnover"])
    log_volume = np.log1p(x["volume"])

    x["turnover_z20"] = rolling_zscore(
        log_turnover,
        window=20,
        min_periods=10,
    )

    x["volume_z20"] = rolling_zscore(
        log_volume,
        window=20,
        min_periods=10,
    )

    x["turnover_ma5"] = x["turnover"].rolling(
        window=5,
        min_periods=5,
    ).mean()

    x["turnover_ma20"] = x["turnover"].rolling(
        window=20,
        min_periods=10,
    ).mean()

    x["turnover_ratio_5_20"] = (
        x["turnover_ma5"]
        / (x["turnover_ma20"] + EPS)
    )

    x["volume_price_confirmation"] = (
        x["volume_z20"] * x["ret_1d"]
    )

    x["volume_price_trend_5"] = (
        (x["ret_1d"] * x["turnover_z20"])
        .rolling(window=5, min_periods=5)
        .mean()
    )

    # 波动、回撤与流动性
    x["volatility_20"] = (
        x["ret_1d"]
        .rolling(window=20, min_periods=15)
        .std()
    )

    x["rolling_peak_20"] = x["close"].rolling(
        window=20,
        min_periods=15,
    ).max()

    x["drawdown_20"] = (
        x["close"] / (x["rolling_peak_20"] + EPS) - 1
    )

    x["amihud_20"] = (
        (
            x["ret_1d"].abs()
            / (x["turnover"] + EPS)
        )
        .rolling(window=20, min_periods=15)
        .mean()
    )

    # MACD
    ema12 = x["close"].ewm(
        span=12,
        adjust=False,
    ).mean()

    ema26 = x["close"].ewm(
        span=26,
        adjust=False,
    ).mean()

    x["macd_dif"] = ema12 - ema26
    x["macd_dea"] = x["macd_dif"].ewm(
        span=9,
        adjust=False,
    ).mean()

    x["macd_hist"] = 2 * (
        x["macd_dif"] - x["macd_dea"]
    )

    # 可用历史数据长度
    x["available_history_days"] = np.arange(
        1,
        len(x) + 1,
    )

    x["avg_turnover_20"] = x["turnover_ma20"]

    return x


def build_factor_panel(
    daily_data: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    """逐股票构建因子，拼接为股票面板。"""
    parts = []

    for _, group in daily_data.groupby("code", sort=True):
        parts.append(
            build_single_stock_factors(
                stock_df=group,
                horizon=horizon,
            )
        )

    return (
        pd.concat(parts, ignore_index=True)
        .sort_values(["date", "code"])
        .reset_index(drop=True)
    )


# ============================================================
# 横截面排名、标签与资格
# ============================================================

def add_cross_section_data(
    factor_df: pd.DataFrame,
    raw_factor_cols: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """
    构建因子横截面排名与相对同行收益标签。

    absolute target：
        fwd_return

    relative target：
        fwd_excess_peer =
        fwd_return - 同日同行平均 fwd_return
    """
    x = factor_df.copy()

    x["peer_fwd_return"] = x.groupby("date")[
        "fwd_return"
    ].transform("mean")

    x["fwd_excess_peer"] = (
        x["fwd_return"] - x["peer_fwd_return"]
    )

    rank_factor_cols = []

    for factor in raw_factor_cols:
        rank_col = f"{factor}_rank"

        x[rank_col] = x.groupby("date")[factor].rank(
            pct=True,
            method="average",
        )

        rank_factor_cols.append(rank_col)

    x["cross_section_count"] = x.groupby("date")[
        "code"
    ].transform("count")

    return x, rank_factor_cols


def add_eligibility_flags(
    factor_df: pd.DataFrame,
    min_history_days: int,
    liquidity_percentile: float,
    rank_factor_cols: list[str],
) -> pd.DataFrame:
    """
    基础资格：
    - 有足够的可用历史；
    - 流动性不处于股票池最低分位；
    - 核心因子有效。
    """
    x = factor_df.copy()

    x["liquidity_rank"] = x.groupby("date")[
        "avg_turnover_20"
    ].rank(
        pct=True,
        method="average",
    )

    x["enough_history"] = (
        x["available_history_days"] >= min_history_days
    )

    x["enough_liquidity"] = (
        x["liquidity_rank"] >= liquidity_percentile
    )

    x["factor_complete"] = x[
        rank_factor_cols
    ].notna().all(axis=1)

    x["eligible"] = (
        x["enough_history"]
        & x["enough_liquidity"]
        & x["factor_complete"]
    )

    return x


# ============================================================
# 双标签 Rank IC 与动态权重
# ============================================================

def daily_rank_ic(
    factor_df: pd.DataFrame,
    rank_factor_cols: list[str],
    target: str,
    min_cross_section: int = 8,
) -> pd.DataFrame:
    """计算每日横截面 Spearman Rank IC。"""
    records = []

    for date, group in factor_df.groupby("date", sort=True):
        base = group[[target] + rank_factor_cols].dropna(
            subset=[target]
        )

        if len(base) < min_cross_section:
            continue

        for factor in rank_factor_cols:
            temp = base[[factor, target]].dropna()

            if len(temp) < min_cross_section:
                continue

            ic = temp[factor].corr(
                temp[target],
                method="spearman",
            )

            if pd.notna(ic):
                records.append(
                    {
                        "date": date,
                        "factor": factor,
                        "rank_ic": ic,
                        "n_stocks": len(temp),
                    }
                )

    result = pd.DataFrame(records)

    if result.empty:
        return result

    return result.sort_values(
        ["date", "factor"]
    ).reset_index(drop=True)


def build_rolling_factor_weights(
    daily_ic_df: pd.DataFrame,
    trading_dates: pd.Series,
    rank_factor_cols: list[str],
    horizon: int,
    lookback: int,
    min_ic_days: int,
    score_name: str,
) -> pd.DataFrame:
    """
    用历史已实现 IC 构建滚动 ICIR 权重。

    当日权重使用滞后 horizon + 1 日的历史 IC，
    避免使用尚未实现未来收益标签。
    """
    if daily_ic_df.empty:
        return pd.DataFrame()

    dates = pd.DatetimeIndex(
        pd.Series(
            pd.to_datetime(trading_dates).unique()
        ).sort_values()
    )

    ic_matrix = (
        daily_ic_df.pivot(
            index="date",
            columns="factor",
            values="rank_ic",
        )
        .reindex(
            index=dates,
            columns=rank_factor_cols,
        )
        .sort_index()
    )

    realized_ic = ic_matrix.shift(horizon + 1)

    rolling_mean = realized_ic.rolling(
        window=lookback,
        min_periods=min_ic_days,
    ).mean()

    rolling_std = realized_ic.rolling(
        window=lookback,
        min_periods=min_ic_days,
    ).std()

    rolling_icir = rolling_mean / (rolling_std + EPS)

    rolling_icir = rolling_icir.clip(
        lower=-3.0,
        upper=3.0,
    )

    abs_sum = rolling_icir.abs().sum(axis=1)

    weights = rolling_icir.div(
        abs_sum.replace(0, np.nan),
        axis=0,
    ).fillna(0.0)

    result = (
        weights
        .reset_index(names="date")
        .melt(
            id_vars="date",
            var_name="factor",
            value_name="weight",
        )
    )

    result["model"] = score_name

    icir_long = (
        rolling_icir
        .reset_index(names="date")
        .melt(
            id_vars="date",
            var_name="factor",
            value_name="rolling_icir",
        )
    )

    result = result.merge(
        icir_long,
        on=["date", "factor"],
        how="left",
    )

    return result.sort_values(
        ["date", "factor"]
    ).reset_index(drop=True)


def add_dynamic_score(
    factor_df: pd.DataFrame,
    weight_df: pd.DataFrame,
    rank_factor_cols: list[str],
    score_prefix: str,
) -> pd.DataFrame:
    """
    按动态 ICIR 权重计算综合得分和同行百分位。

    score_prefix：
    - absolute：未来绝对收益模型
    - relative：未来相对同行收益模型
    """
    x = factor_df.copy()

    score_col = f"{score_prefix}_score"
    rank_col = f"{score_prefix}_score_rank"

    if weight_df.empty:
        x[score_col] = np.nan
        x[rank_col] = np.nan
        return x

    weight_wide = (
        weight_df.pivot(
            index="date",
            columns="factor",
            values="weight",
        )
        .reindex(columns=rank_factor_cols)
        .fillna(0.0)
        .add_suffix("_weight")
        .reset_index()
    )

    x = x.merge(
        weight_wide,
        on="date",
        how="left",
    )

    score = pd.Series(0.0, index=x.index)
    active_weight = pd.Series(0.0, index=x.index)

    for factor in rank_factor_cols:
        weight_col = f"{factor}_weight"

        factor_value = x[factor]
        current_weight = x[weight_col].fillna(0.0)
        valid = factor_value.notna()

        score = score + (
            factor_value.fillna(0.0) * current_weight
        )

        active_weight = active_weight + (
            current_weight.abs() * valid.astype(float)
        )

    x[score_col] = score / (active_weight + EPS)

    x.loc[
        active_weight <= EPS,
        score_col,
    ] = np.nan

    x[rank_col] = np.nan

    valid_mask = (
        x["eligible"]
        & x[score_col].notna()
    )

    x.loc[valid_mask, rank_col] = (
        x.loc[valid_mask]
        .groupby("date")[score_col]
        .rank(
            pct=True,
            ascending=True,
            method="first",
        )
    )

    drop_weight_cols = [
        f"{factor}_weight"
        for factor in rank_factor_cols
        if f"{factor}_weight" in x.columns
    ]

    return x.drop(columns=drop_weight_cols)


# ============================================================
# 个股强势状态与最终候选信号
# ============================================================

def add_strong_stock_signal(
    scored_df: pd.DataFrame,
    breakout_ratio: float,
    trend_min_score: int,
    min_absolute_rank: float,
    min_relative_rank: float,
    min_confidence_score: float,
    require_volume_confirmation: bool,
) -> pd.DataFrame:
    """
    个股绝对强势条件，共 7 项：

    1. 收盘价站上 MA20；
    2. MA20 站上 MA60；
    3. 20 日动量为正；
    4. MACD 柱为正；
    5. 收盘价接近 60 日高点；
    6. 5 日平均成交额高于 20 日平均成交额；
    7. 近 5 日量价趋势为正。

    最终高置信度条件：

    - 基础资格有效；
    - 个股趋势强；
    - 绝对收益模型评分高；
    - 相对同行模型评分不弱；
    - 综合置信度达到阈值；
    - 可选：量价确认。
    """
    x = scored_df.copy()

    x["above_ma20"] = x["close"] > x["ma20"]
    x["ma20_above_ma60"] = x["ma20"] > x["ma60"]
    x["momentum_positive"] = x["momentum_20"] > 0
    x["macd_positive"] = x["macd_hist"] > 0

    x["near_60d_high"] = (
        x["close"]
        >= breakout_ratio * x["max_close_60"]
    )

    x["turnover_expanding"] = (
        x["turnover_ratio_5_20"] >= 1.0
    )

    x["volume_price_positive"] = (
        x["volume_price_trend_5"] > 0
    )

    trend_conditions = [
        "above_ma20",
        "ma20_above_ma60",
        "momentum_positive",
        "macd_positive",
        "near_60d_high",
        "turnover_expanding",
        "volume_price_positive",
    ]

    x["trend_strength_count"] = x[
        trend_conditions
    ].fillna(False).astype(int).sum(axis=1)

    x["trend_strength_score"] = (
        x["trend_strength_count"] / 7.0
    )

    # 4 条不可缺少的趋势底线。
    x["core_uptrend"] = (
        x["above_ma20"]
        & x["ma20_above_ma60"]
        & x["momentum_positive"]
        & x["macd_positive"]
    )

    x["absolute_trend_strong"] = (
        x["core_uptrend"]
        & (
            x["trend_strength_count"]
            >= trend_min_score
        )
    )

    x["volume_confirmation"] = (
        x["turnover_expanding"]
        | x["volume_price_positive"]
    )

    # 最终置信度：
    # 绝对上涨预测 45%
    # 相对同行表现 25%
    # 个股趋势质量 30%
    x["confidence_score"] = (
        0.45 * x["absolute_score_rank"]
        + 0.25 * x["relative_score_rank"]
        + 0.30 * x["trend_strength_score"]
    )

    x["absolute_model_strong"] = (
        x["absolute_score_rank"] >= min_absolute_rank
    )

    x["relative_model_not_weak"] = (
        x["relative_score_rank"] >= min_relative_rank
    )

    x["high_confidence_candidate"] = (
        x["eligible"]
        & x["absolute_trend_strong"]
        & x["absolute_model_strong"]
        & x["relative_model_not_weak"]
        & (
            x["confidence_score"]
            >= min_confidence_score
        )
    )

    if require_volume_confirmation:
        x["high_confidence_candidate"] = (
            x["high_confidence_candidate"]
            & x["volume_confirmation"]
        )

    return x


def select_one_best_stock(
    signal_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    在高置信度候选中选择唯一一只最高分股票。

    无候选时返回空表，即该周没有交易信号。
    """
    candidates = signal_df[
        signal_df["high_confidence_candidate"]
    ].copy()

    if candidates.empty:
        return candidates

    candidates = candidates.sort_values(
        [
            "confidence_score",
            "absolute_score_rank",
            "relative_score_rank",
            "trend_strength_score",
            "absolute_score",
            "code",
        ],
        ascending=[
            False,
            False,
            False,
            False,
            False,
            True,
        ],
    )

    return candidates.head(1).copy()

# ============================================================
# 长期单股票筛选与买入持有回测
# ============================================================

def calculate_buy_hold_metrics(
    price_df: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    对股票池中每只股票计算买入持有指标。

    交易口径：
    - 在 start_date 对应的第一个可交易日开盘买入；
    - 在 end_date 对应的最后一个可交易日开盘卖出；
    - 使用开盘价计算，以减少收盘后信号无法成交的问题。

    输出：
    - total_return：区间累计收益；
    - annualized_return：年化收益；
    - max_drawdown：以开盘价近似计算的最大回撤；
    - calmar_ratio：年化收益 / 最大回撤绝对值；
    - weekly_win_rate：周度收益为正的比例；
    - annualized_excess_return：相对同行等权基准年化超额收益。
    """
    x = price_df.copy()

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)

    x = x[
        (x["date"] >= start_ts)
        & (x["date"] <= end_ts)
    ].copy()

    if x.empty:
        return pd.DataFrame()

    all_dates = pd.DatetimeIndex(
        x["date"].drop_duplicates().sort_values()
    )

    if len(all_dates) < 20:
        return pd.DataFrame()

    actual_start = all_dates[0]
    actual_end = all_dates[-1]

    # --------------------------------------------------------
    # 先构建每只股票的每日开盘价净值
    # --------------------------------------------------------
    price_wide = (
        x.pivot(
            index="date",
            columns="code",
            values="open",
        )
        .sort_index()
    )

    # 只保留在起点和终点均有有效价格的股票。
    valid_codes = [
        code
        for code in price_wide.columns
        if pd.notna(price_wide[code].iloc[0])
        and pd.notna(price_wide[code].iloc[-1])
    ]

    price_wide = price_wide[valid_codes].copy()

    # 统一基准：每只股票起始价格归一化为 1。
    nav_wide = price_wide.div(
        price_wide.iloc[0],
        axis=1,
    )

    # 同行业等权基准。
    peer_nav = nav_wide.mean(axis=1)

    # --------------------------------------------------------
    # 计算周度收益，用于周胜率
    # --------------------------------------------------------
    weekly_nav = nav_wide.resample("W-FRI").last().dropna(how="all")
    weekly_return = weekly_nav.pct_change(
    fill_method=None
)


    # 交易日年化：252。
    trading_days = len(nav_wide)
    years = trading_days / 252.0

    records = []

    for code in valid_codes:
        stock_nav = nav_wide[code].dropna()

        if len(stock_nav) < 20:
            continue

        total_return = stock_nav.iloc[-1] - 1

        annualized_return = (
            stock_nav.iloc[-1] ** (1.0 / years) - 1
            if years > 0
            else np.nan
        )

        drawdown = (
            stock_nav / stock_nav.cummax() - 1
        )

        max_drawdown = drawdown.min()

        calmar_ratio = (
            annualized_return / abs(max_drawdown)
            if pd.notna(max_drawdown)
            and abs(max_drawdown) > EPS
            else np.nan
        )

        stock_weekly_return = weekly_return[code].dropna()

        weekly_win_rate = (
            (stock_weekly_return > 0).mean()
            if not stock_weekly_return.empty
            else np.nan
        )

        # 股票和同行基准的相对净值。
        excess_nav = stock_nav / peer_nav.loc[stock_nav.index]

        annualized_excess_return = (
            excess_nav.iloc[-1] ** (1.0 / years) - 1
            if years > 0
            else np.nan
        )

        records.append(
            {
                "code": code,
                "actual_start_date": actual_start,
                "actual_end_date": actual_end,
                "trading_days": trading_days,
                "years": years,
                "total_return": total_return,
                "annualized_return": annualized_return,
                "max_drawdown": max_drawdown,
                "calmar_ratio": calmar_ratio,
                "weekly_win_rate": weekly_win_rate,
                "annualized_excess_return": (
                    annualized_excess_return
                ),
            }
        )

    return pd.DataFrame(records)


def select_one_long_term_stock(
    daily_data: pd.DataFrame,
    train_start: str,
    train_end: str,
) -> pd.DataFrame:
    """
    使用训练集，选出一只最适合长期持有的股票。

    注意：
    该函数只允许使用训练期数据。
    不能使用样本外期间数据选股，否则会形成未来函数。
    """
    metrics = calculate_buy_hold_metrics(
        price_df=daily_data,
        start_date=train_start,
        end_date=train_end,
    )

    if metrics.empty:
        return metrics

    rank_cols = [
        "annualized_return",
        "annualized_excess_return",
        "calmar_ratio",
        "weekly_win_rate",
    ]

    for col in rank_cols:
        metrics[f"{col}_rank"] = metrics[col].rank(
            pct=True,
            ascending=True,
            method="average",
        )

    # 长期持有综合评分。
    metrics["long_term_score"] = (
        0.35 * metrics["annualized_return_rank"]
        + 0.30 * metrics["annualized_excess_return_rank"]
        + 0.20 * metrics["calmar_ratio_rank"]
        + 0.15 * metrics["weekly_win_rate_rank"]
    )

    metrics = metrics.sort_values(
        [
            "long_term_score",
            "annualized_excess_return",
            "calmar_ratio",
            "annualized_return",
        ],
        ascending=[
            False,
            False,
            False,
            False,
        ],
    ).reset_index(drop=True)

    metrics["selection_rank"] = np.arange(
        1,
        len(metrics) + 1,
    )

    return metrics


def run_fixed_stock_buy_hold_backtest(
    daily_data: pd.DataFrame,
    selected_code: str,
    test_start: str,
    test_end: str,
    cost_bps_one_way: float,
) -> pd.DataFrame:
    """
    对训练期选出的唯一股票进行样本外买入持有回测。

    规则：
    1. 样本外第一天开盘买入；
    2. 最后一天开盘卖出；
    3. 中途不调仓；
    4. 使用双边交易成本；
    5. 同时输出 PCB 股票池等权基准表现。
    """
    x = daily_data.copy()

    start_ts = pd.Timestamp(test_start)
    end_ts = pd.Timestamp(test_end)

    x = x[
        (x["date"] >= start_ts)
        & (x["date"] <= end_ts)
    ].copy()

    if x.empty:
        return pd.DataFrame()

    all_dates = pd.DatetimeIndex(
        x["date"].drop_duplicates().sort_values()
    )

    if len(all_dates) < 2:
        return pd.DataFrame()

    actual_start = all_dates[0]
    actual_end = all_dates[-1]

    price_wide = (
        x.pivot(
            index="date",
            columns="code",
            values="open",
        )
        .sort_index()
    )

    if selected_code not in price_wide.columns:
        raise RuntimeError(
            f"样本外数据中未找到选中股票：{selected_code}"
        )

    stock_price = price_wide[selected_code].dropna()

    if len(stock_price) < 2:
        raise RuntimeError(
            f"{selected_code} 样本外有效开盘价不足。"
        )

    stock_nav_gross = stock_price / stock_price.iloc[0]

    # 等权同行基准：只计算有价格股票的等权净值。
    peer_nav = price_wide.div(
        price_wide.iloc[0],
        axis=1,
    ).mean(axis=1)

    # 双边交易成本。
    total_cost = 2.0 * cost_bps_one_way / 10000.0

    # 成本在建仓与平仓后一次性近似扣减。
    stock_nav_net = stock_nav_gross * (1.0 - total_cost)

    stock_drawdown = (
        stock_nav_net
        / stock_nav_net.cummax()
        - 1
    )

    peer_drawdown = (
        peer_nav / peer_nav.cummax() - 1
    )

    result = pd.DataFrame(
        {
            "date": stock_nav_net.index,
            "selected_code": selected_code,
            "stock_nav_gross": stock_nav_gross.values,
            "stock_nav_net": stock_nav_net.values,
            "peer_nav": peer_nav.loc[
                stock_nav_net.index
            ].values,
        }
    )

    result["excess_nav"] = (
        result["stock_nav_net"]
        / (result["peer_nav"] + EPS)
    )

    result["stock_drawdown"] = stock_drawdown.values
    result["peer_drawdown"] = peer_drawdown.loc[
        stock_nav_net.index
    ].values

    result["actual_start_date"] = actual_start
    result["actual_end_date"] = actual_end
    result["total_cost"] = total_cost

    return result

# ============================================================
# 信号回测
# ============================================================

def run_weekly_best_signal_backtest(
    signal_df: pd.DataFrame,
    start_date: str,
    cost_bps_one_way: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    回测每周唯一最高置信度信号。

    无信号周空仓，策略收益为 0。
    有信号周下一交易日开盘买入，
    下一周调仓对应开盘卖出。
    """
    x = signal_df[
        signal_df["date"] >= pd.Timestamp(start_date)
    ].copy()

    all_dates = pd.DatetimeIndex(
        x["date"].drop_duplicates().sort_values()
    )

    signal_dates = weekly_last_trading_dates(all_dates)

    if len(signal_dates) < 2:
        return pd.DataFrame(), pd.DataFrame()

    next_date_map = {
        all_dates[i]: all_dates[i + 1]
        for i in range(len(all_dates) - 1)
    }

    selection_records = []
    period_records = []

    for i in range(len(signal_dates) - 1):
        signal_date = signal_dates[i]
        next_signal_date = signal_dates[i + 1]

        entry_date = next_date_map.get(signal_date)
        exit_date = next_date_map.get(next_signal_date)

        if entry_date is None or exit_date is None:
            continue

        snapshot = x[x["date"] == signal_date].copy()
        selected = select_one_best_stock(snapshot)

        peer_entry = x.loc[
            x["date"] == entry_date,
            ["code", "open"],
        ].rename(columns={"open": "peer_entry_open"})

        peer_exit = x.loc[
            x["date"] == exit_date,
            ["code", "open"],
        ].rename(columns={"open": "peer_exit_open"})

        peer = peer_entry.merge(
            peer_exit,
            on="code",
            how="inner",
        )

        peer["peer_return"] = (
            peer["peer_exit_open"]
            / (peer["peer_entry_open"] + EPS)
            - 1
        )

        peer_return = peer["peer_return"].mean()

        if selected.empty:
            period_records.append(
                {
                    "signal_date": signal_date,
                    "entry_date": entry_date,
                    "exit_date": exit_date,
                    "has_signal": False,
                    "selected_code": None,
                    "gross_return": np.nan,
                    "net_return": np.nan,
                    "peer_return": peer_return,
                    "excess_peer_return": np.nan,
                    "trading_cost": 0.0,
                }
            )
            continue

        code = selected.iloc[0]["code"]

        entry = x.loc[
            (x["date"] == entry_date)
            & (x["code"] == code),
            "open",
        ]

        exit_ = x.loc[
            (x["date"] == exit_date)
            & (x["code"] == code),
            "open",
        ]

        if entry.empty or exit_.empty:
            period_records.append(
                {
                    "signal_date": signal_date,
                    "entry_date": entry_date,
                    "exit_date": exit_date,
                    "has_signal": False,
                    "selected_code": code,
                    "gross_return": np.nan,
                    "net_return": np.nan,
                    "peer_return": peer_return,
                    "excess_peer_return": np.nan,
                    "trading_cost": 0.0,
                }
            )
            continue

        gross_return = (
            exit_.iloc[0] / (entry.iloc[0] + EPS) - 1
        )

        trading_cost = 2.0 * cost_bps_one_way / 10000.0
        net_return = gross_return - trading_cost
        excess_return = net_return - peer_return

        selected["signal_date"] = signal_date
        selected["entry_date"] = entry_date
        selected["exit_date"] = exit_date
        selected["entry_open"] = entry.iloc[0]
        selected["exit_open"] = exit_.iloc[0]
        selected["gross_return"] = gross_return
        selected["net_return"] = net_return
        selected["peer_return"] = peer_return
        selected["excess_peer_return"] = excess_return
        selected["trading_cost"] = trading_cost

        selection_records.append(selected)

        period_records.append(
            {
                "signal_date": signal_date,
                "entry_date": entry_date,
                "exit_date": exit_date,
                "has_signal": True,
                "selected_code": code,
                "gross_return": gross_return,
                "net_return": net_return,
                "peer_return": peer_return,
                "excess_peer_return": excess_return,
                "trading_cost": trading_cost,
            }
        )

    selections = (
        pd.concat(selection_records, ignore_index=True)
        if selection_records
        else pd.DataFrame()
    )

    periods = pd.DataFrame(period_records)

    if periods.empty:
        return selections, periods

    periods = periods.sort_values(
        "entry_date"
    ).reset_index(drop=True)

    periods["strategy_return_for_nav"] = periods[
        "net_return"
    ].fillna(0.0)

    periods["strategy_nav"] = (
        1 + periods["strategy_return_for_nav"]
    ).cumprod()

    periods["peer_nav"] = (
        1 + periods["peer_return"].fillna(0.0)
    ).cumprod()

    periods["excess_nav"] = (
        periods["strategy_nav"]
        / (periods["peer_nav"] + EPS)
    )

    return selections, periods


def build_signal_backtest_summary(
    backtest_df: pd.DataFrame,
    split_date: str,
) -> pd.DataFrame:
    """
    汇总周度动态最强股策略表现。

    同时计算：
    - 策略收益：无信号周空仓，收益记为 0；
    - PCB 等权基准收益：每周持续持有行业等权组合；
    - 策略相对 PCB 等权基准的累计超额收益。
    """
    if backtest_df.empty:
        return pd.DataFrame()

    split_ts = pd.Timestamp(split_date)

    def summarize(
        frame: pd.DataFrame,
        period_name: str,
    ) -> dict:
        total_weeks = len(frame)

        if frame.empty:
            return {
                "period": period_name,
                "weeks": 0,
                "signal_count": 0,
                "signal_coverage": np.nan,
                "strategy_total_return": np.nan,
                "peer_total_return": np.nan,
                "excess_total_return": np.nan,
                "signal_mean_return": np.nan,
                "signal_median_return": np.nan,
                "signal_win_rate": np.nan,
                "signal_mean_excess_return": np.nan,
                "signal_excess_win_rate": np.nan,
                "max_single_loss": np.nan,
                "strategy_max_drawdown": np.nan,
                "peer_max_drawdown": np.nan,
            }

        # ----------------------------------------------------
        # 策略净值：没有信号的周，空仓，周收益为 0。
        # ----------------------------------------------------
        strategy_weekly_return = frame["net_return"].fillna(0.0)
        strategy_nav = (1.0 + strategy_weekly_return).cumprod()

        # ----------------------------------------------------
        # PCB 等权基准净值：每周始终保持行业等权持仓。
        # ----------------------------------------------------
        peer_weekly_return = frame["peer_return"].fillna(0.0)
        peer_nav = (1.0 + peer_weekly_return).cumprod()

        # 策略相对 PCB 等权基准的净值。
        excess_nav = strategy_nav / (peer_nav + EPS)

        strategy_total_return = strategy_nav.iloc[-1] - 1.0
        peer_total_return = peer_nav.iloc[-1] - 1.0
        excess_total_return = excess_nav.iloc[-1] - 1.0

        strategy_max_drawdown = (
            strategy_nav / strategy_nav.cummax() - 1.0
        ).min()

        peer_max_drawdown = (
            peer_nav / peer_nav.cummax() - 1.0
        ).min()

        # 只有实际产生股票交易的周，才纳入信号胜率统计。
        trades = frame[
            frame["has_signal"]
            & frame["net_return"].notna()
        ].copy()

        if trades.empty:
            return {
                "period": period_name,
                "weeks": total_weeks,
                "signal_count": 0,
                "signal_coverage": 0.0,
                "strategy_total_return": strategy_total_return,
                "peer_total_return": peer_total_return,
                "excess_total_return": excess_total_return,
                "signal_mean_return": np.nan,
                "signal_median_return": np.nan,
                "signal_win_rate": np.nan,
                "signal_mean_excess_return": np.nan,
                "signal_excess_win_rate": np.nan,
                "max_single_loss": np.nan,
                "strategy_max_drawdown": strategy_max_drawdown,
                "peer_max_drawdown": peer_max_drawdown,
            }

        return {
            "period": period_name,
            "weeks": total_weeks,
            "signal_count": len(trades),
            "signal_coverage": len(trades) / total_weeks,
            "strategy_total_return": strategy_total_return,
            "peer_total_return": peer_total_return,
            "excess_total_return": excess_total_return,
            "signal_mean_return": trades["net_return"].mean(),
            "signal_median_return": trades["net_return"].median(),
            "signal_win_rate": (
                trades["net_return"] > 0
            ).mean(),
            "signal_mean_excess_return": trades[
                "excess_peer_return"
            ].mean(),
            "signal_excess_win_rate": (
                trades["excess_peer_return"] > 0
            ).mean(),
            "max_single_loss": trades["net_return"].min(),
            "strategy_max_drawdown": strategy_max_drawdown,
            "peer_max_drawdown": peer_max_drawdown,
        }

    train_frame = backtest_df[
        backtest_df["signal_date"] < split_ts
    ].copy()

    test_frame = backtest_df[
        backtest_df["signal_date"] >= split_ts
    ].copy()

    return pd.DataFrame(
        [
            summarize(backtest_df, "全样本"),
            summarize(train_frame, "训练集"),
            summarize(test_frame, "样本外测试集"),
        ]
    )



# ============================================================
# IC 报告
# ============================================================

def factor_ic_summary(
    daily_ic_df: pd.DataFrame,
    split_date: str,
) -> pd.DataFrame:
    """汇总全样本、训练集、样本外的 Rank IC。"""
    if daily_ic_df.empty:
        return pd.DataFrame()

    split_ts = pd.Timestamp(split_date)

    def summarize(
        frame: pd.DataFrame,
        prefix: str,
    ) -> pd.DataFrame:
        if frame.empty:
            return pd.DataFrame(columns=["factor"])

        result = (
            frame.groupby("factor", as_index=False)["rank_ic"]
            .agg(
                ic_days="count",
                mean_ic="mean",
                ic_std="std",
                positive_ratio=lambda s: (s > 0).mean(),
            )
        )

        result["icir"] = (
            result["mean_ic"]
            / (result["ic_std"] + EPS)
        )

        return result.rename(
            columns={
                "ic_days": f"{prefix}_ic_days",
                "mean_ic": f"{prefix}_mean_ic",
                "ic_std": f"{prefix}_ic_std",
                "positive_ratio": f"{prefix}_positive_ratio",
                "icir": f"{prefix}_icir",
            }
        )

    all_summary = summarize(daily_ic_df, "all")

    train_summary = summarize(
        daily_ic_df[daily_ic_df["date"] < split_ts],
        "train",
    )

    test_summary = summarize(
        daily_ic_df[daily_ic_df["date"] >= split_ts],
        "test",
    )

    return (
        all_summary
        .merge(train_summary, on="factor", how="left")
        .merge(test_summary, on="factor", how="left")
        .sort_values(
            "test_icir",
            ascending=False,
            na_position="last",
        )
        .reset_index(drop=True)
    )


# ============================================================
# 最新推荐输出
# ============================================================

def print_latest_recommendation(
    signal_df: pd.DataFrame,
) -> pd.DataFrame:
    """打印最新周度信号日的唯一高置信度股票。"""
    all_dates = pd.DatetimeIndex(
        signal_df["date"].drop_duplicates().sort_values()
    )

    signal_dates = weekly_last_trading_dates(all_dates)

    latest_signal_date = signal_dates[-1]

    snapshot = signal_df[
        signal_df["date"] == latest_signal_date
    ].copy()

    selected = select_one_best_stock(snapshot)

    print("\n" + "=" * 92)
    print("【本周唯一最高置信度标的】")
    print("=" * 92)
    print(f"信号日期：{latest_signal_date.date()}")

    if selected.empty:
        print("\n结论：本周无高置信度标的。")
        print(
            "原因：股票池内没有标的同时通过绝对趋势、"
            "绝对收益模型、相对同行模型与置信度阈值。"
        )
        return selected

    row = selected.iloc[0]

    print("\n结论：当前模型最看好后市的唯一标的是：")
    print(f"股票代码：{row['code']}")
    print(f"同花顺代码：{row['thscode']}")
    print(f"板块：{row['board']}")
    print(f"信号日收盘价：{row['close']:.2f}")

    print("\n【个股绝对强势】")
    print(f"收盘价站上 MA20：{bool(row['above_ma20'])}")
    print(f"MA20 位于 MA60 上方：{bool(row['ma20_above_ma60'])}")
    print(f"20 日动量为正：{bool(row['momentum_positive'])}")
    print(f"MACD 柱体为正：{bool(row['macd_positive'])}")
    print(f"接近 60 日高点：{bool(row['near_60d_high'])}")
    print(
        f"趋势强度：{int(row['trend_strength_count'])}/7 "
        f"（{format_pct(row['trend_strength_score'])}）"
    )

    print("\n【绝对收益模型：未来上涨预测】")
    print(
        f"绝对模型得分："
        f"{format_num(row['absolute_score'], 6)}"
    )
    print(
        f"绝对模型同行分位："
        f"{format_pct(row['absolute_score_rank'])}"
    )

    print("\n【相对同行模型：跑赢同业预测】")
    print(
        f"相对模型得分："
        f"{format_num(row['relative_score'], 6)}"
    )
    print(
        f"相对模型同行分位："
        f"{format_pct(row['relative_score_rank'])}"
    )

    print("\n【量价与流动性】")
    print(
        f"5 日 / 20 日成交额比："
        f"{format_num(row['turnover_ratio_5_20'], 3)}"
    )
    print(
        f"成交额放大："
        f"{bool(row['turnover_expanding'])}"
    )
    print(
        f"近 5 日量价趋势为正："
        f"{bool(row['volume_price_positive'])}"
    )
    print(
        f"20 日平均成交额同行分位："
        f"{format_pct(row['liquidity_rank'])}"
    )

    print("\n【最终结论】")
    print(
        f"综合置信度评分："
        f"{format_pct(row['confidence_score'])}"
    )
    print(
        "该结论为历史统计框架下的量化筛选结果，"
        "不构成任何买卖建议。"
    )

    return selected


# ============================================================
# 主程序
# ============================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "PCB / 电子产业链："
            "最强个股识别 + 双标签横截面量化模型"
        )
    )

    parser.add_argument(
        "--db",
        required=True,
        help=(
            "本地 DuckDB 数据库路径。例如："
            "C:\\path\\to\\market.duckdb"
        ),
    )


    parser.add_argument(
        "--start",
        default="2021-01-01",
        help="正式研究开始日期，默认 2021-01-01。",
    )

    parser.add_argument(
        "--end",
        default=None,
        help="研究结束日期，默认使用数据库最新日期。",
    )

    parser.add_argument(
        "--horizon",
        type=int,
        default=5,
        help="未来收益标签持有期，单位为交易日，默认 5。",
    )

    parser.add_argument(
        "--lookback",
        type=int,
        default=500,
        help="滚动 ICIR 回看窗口，默认 500 个交易日。",
    )

    parser.add_argument(
        "--min-ic-days",
        type=int,
        default=100,
        help="计算滚动 ICIR 所需最少已实现 IC 天数，默认 100。",
    )

    parser.add_argument(
        "--min-history-days",
        type=int,
        default=120,
        help="进入候选池需要的最少可用历史天数，默认 120。",
    )

    parser.add_argument(
        "--liquidity-percentile",
        type=float,
        default=0.20,
        help="最低流动性横截面分位，默认 0.20。",
    )

    parser.add_argument(
        "--breakout-ratio",
        type=float,
        default=0.95,
        help="接近 60 日高点阈值，默认 0.95。",
    )

    parser.add_argument(
        "--trend-min-score",
        type=int,
        default=5,
        help="7 项趋势条件中至少满足几项，默认 5。",
    )

    parser.add_argument(
        "--min-absolute-rank",
        type=float,
        default=0.65,
        help="绝对收益模型最低同行分位，默认 0.65。",
    )

    parser.add_argument(
        "--min-relative-rank",
        type=float,
        default=0.50,
        help="相对同行模型最低同行分位，默认 0.50。",
    )

    parser.add_argument(
        "--min-confidence-score",
        type=float,
        default=0.72,
        help="最终综合置信度最低阈值，默认 0.72。",
    )

    parser.add_argument(
        "--require-volume-confirmation",
        action="store_true",
        help=(
            "启用后，候选股还必须满足成交额放大或"
            "近 5 日量价趋势为正。"
        ),
    )

    parser.add_argument(
        "--cost-bps-one-way",
        type=float,
        default=8.0,
        help="单边交易成本，单位 bps，默认 8。",
    )

    parser.add_argument(
        "--split-date",
        default="2025-01-01",
        help="样本外测试起点，默认 2025-01-01。",
    )

    parser.add_argument(
        "--out-dir",
        default=None,
        help=(
            "输出目录；默认使用数据库目录下的 "
            "pcb_strongest_stock_output。"
        ),
    )

    args = parser.parse_args()

    # 参数校验
    if args.horizon <= 0:
        raise ValueError("--horizon 必须是正整数。")

    if args.lookback <= 0:
        raise ValueError("--lookback 必须是正整数。")

    if args.min_ic_days <= 0:
        raise ValueError("--min-ic-days 必须是正整数。")

    if args.min_ic_days > args.lookback:
        raise ValueError(
            "--min-ic-days 不能大于 --lookback。"
        )

    if args.min_history_days <= 0:
        raise ValueError(
            "--min-history-days 必须是正整数。"
        )

    if not 0 <= args.liquidity_percentile < 1:
        raise ValueError(
            "--liquidity-percentile 必须在 [0, 1) 内。"
        )

    if not 0 < args.breakout_ratio <= 1:
        raise ValueError(
            "--breakout-ratio 必须在 (0, 1] 内。"
        )

    if not 1 <= args.trend_min_score <= 7:
        raise ValueError(
            "--trend-min-score 必须在 1 到 7 内。"
        )

    for name, value in [
        ("--min-absolute-rank", args.min_absolute_rank),
        ("--min-relative-rank", args.min_relative_rank),
        ("--min-confidence-score", args.min_confidence_score),
    ]:
        if not 0 <= value <= 1:
            raise ValueError(
                f"{name} 必须在 [0, 1] 内。"
            )

    if args.cost_bps_one_way < 0:
        raise ValueError(
            "--cost-bps-one-way 不能小于 0。"
        )

    if args.end is not None:
        if pd.Timestamp(args.end) < pd.Timestamp(args.start):
            raise ValueError(
                "--end 不能早于 --start。"
            )

    db_path = Path(args.db).expanduser().resolve()

    if not db_path.exists():
        raise FileNotFoundError(
            f"找不到 DuckDB 数据库：{db_path}"
        )

    if args.out_dir is None:
        project_root = Path(__file__).resolve().parent
        output_dir = (
            project_root
            / "out"
            / "pcb_strongest_stock_output"
        )
    else:
        output_dir = Path(args.out_dir).expanduser().resolve()


    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 92)
    print("PCB / 电子产业链：最强个股识别 + 双标签横截面量化模型")
    print("=" * 92)
    print(f"数据库：{db_path}")
    print(f"研究股票池数量：{len(PCB_UNIVERSE)}")
    print(f"正式研究区间：{args.start} 至 {args.end or '数据库最新日期'}")
    print(f"标签持有期：{args.horizon} 个交易日")
    print(f"滚动 ICIR 回看窗口：{args.lookback} 个交易日")
    print(f"最少历史 IC 天数：{args.min_ic_days}")
    print(f"流动性最低分位：{args.liquidity_percentile:.0%}")
    print(f"接近 60 日高点阈值：{args.breakout_ratio:.0%}")
    print(f"趋势至少满足条件数：{args.trend_min_score}/7")
    print(f"绝对模型最低分位：{args.min_absolute_rank:.0%}")
    print(f"相对模型最低分位：{args.min_relative_rank:.0%}")
    print(f"最终置信度最低阈值：{args.min_confidence_score:.0%}")
    print(
        "是否强制量价确认："
        f"{'是' if args.require_volume_confirmation else '否'}"
    )
    print(f"单边交易成本：{args.cost_bps_one_way:.2f} bps")
    print(f"样本外测试起点：{args.split_date}")
    print(f"输出目录：{output_dir}")

    con = duckdb.connect(str(db_path), read_only=True)

    try:
        check_database_view(con)

        if args.end is None:
            args.end = con.execute(
                """
                SELECT CAST(MAX(date) AS VARCHAR)
                FROM v_daily_qfq
                """
            ).fetchone()[0]

        daily_data = load_universe_daily(
            con=con,
            codes=PCB_UNIVERSE,
            start_date=args.start,
            end_date=args.end,
        )
    finally:
        con.close()

    if daily_data["code"].nunique() < 8:
        raise RuntimeError(
            "可用股票数量少于 8，只能停止横截面研究。"
        )

    print("\n正在逐股票计算原始因子...")

    factor_df = build_factor_panel(
        daily_data=daily_data,
        horizon=args.horizon,
    )

    # 保留低重复、兼顾趋势、位置、量价、风险和流动性的因子。
    raw_factor_cols = [
        "momentum_20",
        "breakout_60",
        "volume_price_trend_5",
        "turnover_ratio_5_20",
        "volatility_20",
        "amihud_20",
    ]

    raw_factor_cols = [
        col for col in raw_factor_cols
        if col in factor_df.columns
    ]

    print("正在构建横截面排名、绝对收益与相对收益标签...")

    factor_df, rank_factor_cols = add_cross_section_data(
        factor_df=factor_df,
        raw_factor_cols=raw_factor_cols,
    )

    # 去掉预热期，只保留正式研究区间。
    factor_df = factor_df[
        factor_df["date"] >= pd.Timestamp(args.start)
    ].copy()

    factor_df = add_eligibility_flags(
        factor_df=factor_df,
        min_history_days=args.min_history_days,
        liquidity_percentile=args.liquidity_percentile,
        rank_factor_cols=rank_factor_cols,
    )

    print("正在计算绝对收益模型 Rank IC...")
    absolute_ic_df = daily_rank_ic(
        factor_df=factor_df,
        rank_factor_cols=rank_factor_cols,
        target="fwd_return",
    )

    print("正在计算相对同行模型 Rank IC...")
    relative_ic_df = daily_rank_ic(
        factor_df=factor_df,
        rank_factor_cols=rank_factor_cols,
        target="fwd_excess_peer",
    )

    if absolute_ic_df.empty or relative_ic_df.empty:
        raise RuntimeError(
            "无法计算有效 Rank IC，请检查数据历史和因子有效性。"
        )

    print("正在构建绝对收益模型动态权重...")
    absolute_weight_df = build_rolling_factor_weights(
        daily_ic_df=absolute_ic_df,
        trading_dates=factor_df["date"],
        rank_factor_cols=rank_factor_cols,
        horizon=args.horizon,
        lookback=args.lookback,
        min_ic_days=args.min_ic_days,
        score_name="absolute",
    )

    print("正在构建相对同行模型动态权重...")
    relative_weight_df = build_rolling_factor_weights(
        daily_ic_df=relative_ic_df,
        trading_dates=factor_df["date"],
        rank_factor_cols=rank_factor_cols,
        horizon=args.horizon,
        lookback=args.lookback,
        min_ic_days=args.min_ic_days,
        score_name="relative",
    )

    print("正在计算双模型得分与强势信号...")

    scored_df = add_dynamic_score(
        factor_df=factor_df,
        weight_df=absolute_weight_df,
        rank_factor_cols=rank_factor_cols,
        score_prefix="absolute",
    )

    scored_df = add_dynamic_score(
        factor_df=scored_df,
        weight_df=relative_weight_df,
        rank_factor_cols=rank_factor_cols,
        score_prefix="relative",
    )

    signal_df = add_strong_stock_signal(
        scored_df=scored_df,
        breakout_ratio=args.breakout_ratio,
        trend_min_score=args.trend_min_score,
        min_absolute_rank=args.min_absolute_rank,
        min_relative_rank=args.min_relative_rank,
        min_confidence_score=args.min_confidence_score,
        require_volume_confirmation=args.require_volume_confirmation,
    )

    absolute_ic_summary = factor_ic_summary(
        absolute_ic_df,
        args.split_date,
    )

    relative_ic_summary = factor_ic_summary(
        relative_ic_df,
        args.split_date,
    )

    print("\n" + "=" * 92)
    print("【绝对收益模型 Rank IC 汇总：预测未来是否上涨】")
    print("=" * 92)

    print(
        absolute_ic_summary[
            [
                "factor",
                "all_mean_ic",
                "all_icir",
                "train_mean_ic",
                "test_mean_ic",
                "test_icir",
            ]
        ].to_string(
            index=False,
            float_format=lambda v: f"{v:.4f}",
        )
    )

    print("\n" + "=" * 92)
    print("【相对同行模型 Rank IC 汇总：预测是否跑赢同业】")
    print("=" * 92)

    print(
        relative_ic_summary[
            [
                "factor",
                "all_mean_ic",
                "all_icir",
                "train_mean_ic",
                "test_mean_ic",
                "test_icir",
            ]
        ].to_string(
            index=False,
            float_format=lambda v: f"{v:.4f}",
        )
    )

    # --------------------------------------------------------
    # 固定单股票长期持有策略
    # 训练期选出一只，样本外阶段买入后一直持有
    # --------------------------------------------------------
    train_start = args.start

    train_end = (
        pd.Timestamp(args.split_date)
        - pd.Timedelta(days=1)
    ).strftime("%Y-%m-%d")

    test_start = args.split_date
    test_end = args.end

    print("\n正在使用训练集筛选唯一长期持有股票...")

    long_term_ranking_df = select_one_long_term_stock(
        daily_data=daily_data,
        train_start=train_start,
        train_end=train_end,
    )

    if long_term_ranking_df.empty:
        raise RuntimeError(
            "长期持有股票筛选失败。"
            "请检查训练期是否有足够数据。"
        )

    selected_long_term_stock = long_term_ranking_df.iloc[0]
    selected_code = selected_long_term_stock["code"]

    print("\n" + "=" * 92)
    print("【训练期长期持有股票排名】")
    print("=" * 92)

    display_cols = [
        "selection_rank",
        "code",
        "long_term_score",
        "annualized_return",
        "annualized_excess_return",
        "calmar_ratio",
        "weekly_win_rate",
        "max_drawdown",
    ]

    print(
        long_term_ranking_df[display_cols]
        .head(17)
        .to_string(
            index=False,
            float_format=lambda v: (
                f"{v:.4%}" if abs(v) < 10 else f"{v:.4f}"
            ),
        )
    )

    print("\n" + "=" * 92)
    print("【训练期筛选出的唯一长期持有股票】")
    print("=" * 92)
    print(f"股票代码：{selected_code}")
    print(
        f"长期持有综合评分："
        f"{selected_long_term_stock['long_term_score']:.4f}"
    )
    print(
        f"训练期累计收益："
        f"{selected_long_term_stock['total_return']:.2%}"
    )
    print(
        f"训练期年化收益："
        f"{selected_long_term_stock['annualized_return']:.2%}"
    )
    print(
        f"训练期年化相对同行超额："
        f"{selected_long_term_stock['annualized_excess_return']:.2%}"
    )
    print(
        f"训练期最大回撤："
        f"{selected_long_term_stock['max_drawdown']:.2%}"
    )
    print(
        f"训练期 Calmar 比率："
        f"{selected_long_term_stock['calmar_ratio']:.4f}"
    )
    print(
        f"训练期周度胜率："
        f"{selected_long_term_stock['weekly_win_rate']:.2%}"
    )

    print("\n正在进行样本外“买入并长期持有”回测...")

    fixed_stock_backtest_df = run_fixed_stock_buy_hold_backtest(
        daily_data=daily_data,
        selected_code=selected_code,
        test_start=test_start,
        test_end=test_end,
        cost_bps_one_way=args.cost_bps_one_way,
    )

    if fixed_stock_backtest_df.empty:
        print("长期持有样本外回测为空。")
    else:
        final_row = fixed_stock_backtest_df.iloc[-1]

        stock_total_return = final_row["stock_nav_net"] - 1
        peer_total_return = final_row["peer_nav"] - 1
        excess_total_return = final_row["excess_nav"] - 1

        stock_max_drawdown = fixed_stock_backtest_df[
            "stock_drawdown"
        ].min()

        peer_max_drawdown = fixed_stock_backtest_df[
            "peer_drawdown"
        ].min()

        print("\n" + "=" * 92)
        print("【样本外唯一股票长期持有回测】")
        print("=" * 92)
        print(f"训练期选中股票：{selected_code}")
        print(
            f"样本外持有区间："
            f"{final_row['actual_start_date'].date()} 至 "
            f"{final_row['actual_end_date'].date()}"
        )
        print(f"股票累计净收益：{stock_total_return:.2%}")
        print(f"PCB 同业等权累计收益：{peer_total_return:.2%}")
        print(f"相对同行累计超额：{excess_total_return:.2%}")
        print(f"股票持有期最大回撤：{stock_max_drawdown:.2%}")
        print(f"同行等权最大回撤：{peer_max_drawdown:.2%}")
        print(
            f"双边交易成本假设："
            f"{final_row['total_cost']:.2%}"
        )
    # --------------------------------------------------------
    # 每周动态选出当前最强股票
    # - 每周重新比较股票池；
    # - 最强股改变则换股；
    # - 无高置信度标的则空仓。
    # --------------------------------------------------------
    print("\n正在回测“每周动态选择唯一最强股票”策略...")

    selections_df, weekly_backtest_df = (
        run_weekly_best_signal_backtest(
            signal_df=signal_df,
            start_date=args.start,
            cost_bps_one_way=args.cost_bps_one_way,
        )
    )

    weekly_backtest_summary = build_signal_backtest_summary(
        backtest_df=weekly_backtest_df,
        split_date=args.split_date,
    )

    print("\n" + "=" * 92)
    print("【每周动态选择唯一最强股票：回测结果】")
    print("=" * 92)
        # --------------------------------------------------------
    # 月度归因报告
    # 用于检查策略在每个月是否跑赢 PCB 等权基准，
    # 特别关注科技股回调较明显的月份。
    # --------------------------------------------------------
    if not weekly_backtest_df.empty:
        monthly_report = weekly_backtest_df.copy()

        monthly_report["month"] = (
            pd.to_datetime(monthly_report["signal_date"])
            .dt.to_period("M")
            .astype(str)
        )

        # 无信号时策略空仓，因此该周策略收益为 0。
        monthly_report["strategy_weekly_return"] = (
            monthly_report["net_return"].fillna(0.0)
        )

        # PCB 等权基准每周始终有收益。
        monthly_report["peer_weekly_return"] = (
            monthly_report["peer_return"].fillna(0.0)
        )

        monthly_summary = (
            monthly_report.groupby("month", as_index=False)
            .agg(
                weeks=("month", "size"),
                signal_count=("has_signal", "sum"),
                strategy_total_return=(
                    "strategy_weekly_return",
                    lambda x: (1.0 + x).prod() - 1.0,
                ),
                peer_total_return=(
                    "peer_weekly_return",
                    lambda x: (1.0 + x).prod() - 1.0,
                ),
            )
        )

        monthly_summary["signal_coverage"] = (
            monthly_summary["signal_count"]
            / monthly_summary["weeks"]
        )

        monthly_summary["excess_total_return"] = (
            (1.0 + monthly_summary["strategy_total_return"])
            / (1.0 + monthly_summary["peer_total_return"])
            - 1.0
        )

        print("\n" + "=" * 92)
        print("【每周动态最强股策略：月度收益归因】")
        print("=" * 92)
        print(
            monthly_summary.to_string(
                index=False,
                float_format=lambda v: f"{v:.2%}",
            )
        )

        monthly_report_file = (
            output_dir / "pcb_weekly_strategy_monthly_summary.csv"
        )
        monthly_summary.to_csv(
            monthly_report_file,
            index=False,
            encoding="utf-8-sig",
        )

        weekly_selection_file = (
            output_dir / "pcb_weekly_dynamic_selection_backtest.csv"
        )
        weekly_backtest_df.to_csv(
            weekly_selection_file,
            index=False,
            encoding="utf-8-sig",
        )

    if weekly_backtest_summary.empty:
        print("没有可用的周度动态选股回测结果。")
    else:
        print(
            weekly_backtest_summary.to_string(
                index=False,
                float_format=lambda v: (
                    f"{v:.2%}" if abs(v) < 10 else f"{v:.4f}"
                ),
            )
        )

    # 输出截至最新一个周度信号日的当前最强股票。
    latest_selected = print_latest_recommendation(signal_df)

    # --------------------------------------------------------
    # 输出 CSV
    # --------------------------------------------------------
    factor_file = output_dir / "pcb_factor_panel.csv"
    signal_file = output_dir / "pcb_signal_panel.csv"

    absolute_ic_file = output_dir / "pcb_absolute_daily_rank_ic.csv"
    relative_ic_file = output_dir / "pcb_relative_daily_rank_ic.csv"

    absolute_ic_summary_file = (
        output_dir / "pcb_absolute_ic_summary.csv"
    )

    relative_ic_summary_file = (
        output_dir / "pcb_relative_ic_summary.csv"
    )

    absolute_weight_file = (
        output_dir / "pcb_absolute_factor_weights.csv"
    )

    relative_weight_file = (
        output_dir / "pcb_relative_factor_weights.csv"
    )

    long_term_ranking_file = (
        output_dir / "pcb_long_term_stock_ranking_train.csv"
    )

    fixed_stock_backtest_file = (
        output_dir / "pcb_fixed_stock_buy_hold_oos.csv"
    )

    factor_df.to_csv(
        factor_file,
        index=False,
        encoding="utf-8-sig",
    )

    signal_df.to_csv(
        signal_file,
        index=False,
        encoding="utf-8-sig",
    )

    absolute_ic_df.to_csv(
        absolute_ic_file,
        index=False,
        encoding="utf-8-sig",
    )

    relative_ic_df.to_csv(
        relative_ic_file,
        index=False,
        encoding="utf-8-sig",
    )

    absolute_ic_summary.to_csv(
        absolute_ic_summary_file,
        index=False,
        encoding="utf-8-sig",
    )

    relative_ic_summary.to_csv(
        relative_ic_summary_file,
        index=False,
        encoding="utf-8-sig",
    )

    absolute_weight_df.to_csv(
        absolute_weight_file,
        index=False,
        encoding="utf-8-sig",
    )

    relative_weight_df.to_csv(
        relative_weight_file,
        index=False,
        encoding="utf-8-sig",
    )

    long_term_ranking_df.to_csv(
        long_term_ranking_file,
        index=False,
        encoding="utf-8-sig",
    )

    fixed_stock_backtest_df.to_csv(
        fixed_stock_backtest_file,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 92)
    print("【程序运行完成】")
    print("=" * 92)
    print(f"原始因子面板：{factor_file}")
    print(f"双模型得分与最终信号面板：{signal_file}")
    print(f"绝对收益模型日度 IC：{absolute_ic_file}")
    print(f"相对同行模型日度 IC：{relative_ic_file}")
    print(f"绝对收益模型 IC 汇总：{absolute_ic_summary_file}")
    print(f"相对同行模型 IC 汇总：{relative_ic_summary_file}")
    print(f"绝对收益模型动态权重：{absolute_weight_file}")
    print(f"相对同行模型动态权重：{relative_weight_file}")
    print(f"训练期长期持有股票排名：{long_term_ranking_file}")
    print(f"样本外固定股票买入持有回测：{fixed_stock_backtest_file}")

    print("\n说明：")
    print("1. 训练期仅用于在股票池中选择唯一长期持有股票。")
    print("2. 样本外期间不再换股，第一天开盘买入、最后一天开盘卖出。")
    print("3. 样本外数据未参与股票选择，避免未来函数。")
    print("4. 长期持有评分综合考虑收益、超额、回撤和周度胜率。")
    print("5. 所有结果只用于量化研究，不构成买卖建议。")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
