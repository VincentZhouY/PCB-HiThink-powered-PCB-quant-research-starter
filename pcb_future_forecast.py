"""
PCB / 电子产业链：未来 5 日收益概率预测模型
=============================================

功能：
1. 使用历史量价、趋势、波动和流动性特征；
2. 预测每只股票未来 horizon 个交易日的预期收益；
3. 预测未来 horizon 个交易日上涨的概率；
4. 依据历史 Walk-forward 预测误差，生成预测收益区间；
5. 使用逐周扩展窗口训练，严格避免将未实现标签用于训练；
6. 导出历史预测、评估报告与最新预测结果。

交易标签定义：
    信号日 t 收盘后生成信号；
    t+1 开盘买入；
    t+horizon+1 开盘卖出；

    fwd_return =
    open[t + horizon + 1] / open[t + 1] - 1

重要声明：
- 本程序仅用于量化研究；
- 股票预测只能提供统计概率，不可能准确预知真实未来价格；
- 回测有效不代表未来一定有效，不构成任何投资建议。
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle


from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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


# ==========================================================
# 通用工具
# ==========================================================

def normalize_code(value) -> str | None:
    """从证券代码字段中提取六码股票代码。"""
    if pd.isna(value):
        return None

    matches = re.findall(r"\d{6}", str(value))
    return matches[-1] if matches else None


def safe_pct_change(series: pd.Series, periods: int) -> pd.Series:
    """不使用 pandas 默认前向填充计算收益率。"""
    return series.pct_change(periods=periods, fill_method=None)


def rolling_zscore(
    series: pd.Series,
    window: int,
    min_periods: int,
) -> pd.Series:
    """时间序列滚动 Z 分数。"""
    rolling_mean = series.rolling(
        window=window,
        min_periods=min_periods,
    ).mean()

    rolling_std = series.rolling(
        window=window,
        min_periods=min_periods,
    ).std()

    return (series - rolling_mean) / (rolling_std + EPS)


def weekly_last_trading_dates(dates: pd.Series) -> pd.DatetimeIndex:
    """取得每周最后一个实际交易日。"""
    calendar = pd.DataFrame(
        {
            "date": pd.Series(pd.to_datetime(dates).unique())
        }
    ).sort_values("date")

    calendar["week"] = calendar["date"].dt.to_period("W-FRI")

    return pd.DatetimeIndex(
        calendar.groupby("week")["date"].max().tolist()
    )


def format_pct(value: float | int | None) -> str:
    """格式化百分比。"""
    if value is None or pd.isna(value):
        return "N/A"

    return f"{value:.2%}"


def format_num(value: float | int | None, digits: int = 4) -> str:
    """格式化普通数值。"""
    if value is None or pd.isna(value):
        return "N/A"

    return f"{value:.{digits}f}"


# ==========================================================
# 数据读取
# ==========================================================

def check_database_view(con: duckdb.DuckDBPyConnection) -> None:
    """检查日线视图是否存在。"""
    result = con.execute(
        """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_name = 'v_daily_qfq'
        """
    ).fetchall()

    if not result:
        raise RuntimeError("数据库中未找到 v_daily_qfq 视图。")


def load_universe_daily(
    con: duckdb.DuckDBPyConnection,
    codes: list[str],
    start_date: str,
    end_date: str | None,
) -> pd.DataFrame:
    """
    读取股票池日线数据。

    为计算 60 日因子，正式起点前额外读取约 500 个自然日。
    """
    warmup_start = (
        pd.Timestamp(start_date) - pd.Timedelta(days=500)
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

    params: list[str] = [warmup_start] + list(codes)

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

    print("\n正在读取 PCB 股票池日线数据，请稍候...")

    df = con.execute(sql, params).fetchdf()

    if df.empty:
        raise RuntimeError(
            "读取结果为空，请检查数据库路径、股票代码和 thscode 格式。"
        )

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
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
        df[col] = pd.to_numeric(df[col], errors="coerce")

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
        print("\n警告：以下股票代码未在数据库中找到：")
        print(", ".join(missing_codes))

    return df


# ==========================================================
# 特征工程与未来标签
# ==========================================================

def build_single_stock_features(
    stock_df: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    """
    对单只股票构建特征与未来标签。

    所有特征仅基于信号日及此前数据。
    未来收益标签只用于历史训练、评估，最新交易日的标签自然为空。
    """
    x = (
        stock_df.sort_values("date")
        .drop_duplicates("date")
        .copy()
    )

    # ------------------------------------------------------
    # 收益标签：下一日开盘买入，持有 horizon 日后开盘卖出
    # ------------------------------------------------------
    x["entry_open_next"] = x["open"].shift(-1)
    x["exit_open_future"] = x["open"].shift(
        -(horizon + 1)
    )

    x["fwd_return"] = (
        x["exit_open_future"]
        / (x["entry_open_next"] + EPS)
        - 1.0
    )

    x["target_up"] = np.where(
        x["fwd_return"].notna(),
        (x["fwd_return"] > 0).astype(int),
        np.nan,
    )

    # ------------------------------------------------------
    # 基础收益与动量
    # ------------------------------------------------------
    x["ret_1d"] = safe_pct_change(x["close"], 1)
    x["ret_5d"] = safe_pct_change(x["close"], 5)

    x["momentum_5"] = safe_pct_change(x["close"], 5)
    x["momentum_10"] = safe_pct_change(x["close"], 10)
    x["momentum_20"] = safe_pct_change(x["close"], 20)
    x["momentum_60"] = safe_pct_change(x["close"], 60)

    # ------------------------------------------------------
    # 均线、趋势、位置
    # ------------------------------------------------------
    x["ma5"] = x["close"].rolling(
        window=5,
        min_periods=5,
    ).mean()

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

    x["close_ma5_gap"] = (
        x["close"] / (x["ma5"] + EPS) - 1.0
    )

    x["close_ma20_gap"] = (
        x["close"] / (x["ma20"] + EPS) - 1.0
    )

    x["close_ma60_gap"] = (
        x["close"] / (x["ma60"] + EPS) - 1.0
    )

    x["ma20_ma60_gap"] = (
        x["ma20"] / (x["ma60"] + EPS) - 1.0
    )

    x["breakout_60"] = (
        x["close"] / (x["max_close_60"] + EPS) - 1.0
    )

    x["position_20_60"] = (
        (x["close"] - x["min_close_20"])
        / (x["max_close_60"] - x["min_close_20"] + EPS)
    )

    # ------------------------------------------------------
    # 成交额、成交量与量价关系
    # ------------------------------------------------------
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
        x["turnover_ma5"] / (x["turnover_ma20"] + EPS)
    )

    x["volume_price_confirmation"] = (
        x["volume_z20"] * x["ret_1d"]
    )

    x["volume_price_trend_5"] = (
        x["volume_price_confirmation"]
        .rolling(window=5, min_periods=5)
        .mean()
    )

    # ------------------------------------------------------
    # 波动、回撤、流动性
    # ------------------------------------------------------
    x["volatility_5"] = (
        x["ret_1d"]
        .rolling(window=5, min_periods=5)
        .std()
    )

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
        x["close"] / (x["rolling_peak_20"] + EPS) - 1.0
    )

    x["amihud_20"] = (
        (
            x["ret_1d"].abs()
            / (x["turnover"] + EPS)
        )
        .rolling(window=20, min_periods=15)
        .mean()
    )

    # ------------------------------------------------------
    # MACD
    # ------------------------------------------------------
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

    x["macd_hist"] = 2.0 * (
        x["macd_dif"] - x["macd_dea"]
    )

    return x


def build_feature_panel(
    daily_data: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    """逐股票构建特征，合并成全股票池面板。"""
    parts = []

    for _, group in daily_data.groupby("code", sort=True):
        parts.append(
            build_single_stock_features(
                stock_df=group,
                horizon=horizon,
            )
        )

    panel = (
        pd.concat(parts, ignore_index=True)
        .sort_values(["date", "code"])
        .reset_index(drop=True)
    )

    return panel


def add_cross_section_features(panel: pd.DataFrame) -> pd.DataFrame:
    """
    加入在当天可见的同行横截面特征。

    这些不是未来信息：
    - 同行平均 20 日动量；
    - 股票相对同行的 20 日动量；
    - 股票当日同行动量分位；
    - 股票当日同行波动率分位；
    - 股票当日同行成交额分位。
    """
    x = panel.copy()

    x["peer_momentum_20"] = x.groupby("date")[
        "momentum_20"
    ].transform("mean")

    x["relative_momentum_20"] = (
        x["momentum_20"] - x["peer_momentum_20"]
    )

    x["momentum_20_cs_rank"] = x.groupby("date")[
        "momentum_20"
    ].rank(
        pct=True,
        method="average",
    )

    x["breakout_60_cs_rank"] = x.groupby("date")[
        "breakout_60"
    ].rank(
        pct=True,
        method="average",
    )

    x["volatility_20_cs_rank"] = x.groupby("date")[
        "volatility_20"
    ].rank(
        pct=True,
        method="average",
    )

    x["turnover_cs_rank"] = x.groupby("date")[
        "turnover_ma20"
    ].rank(
        pct=True,
        method="average",
    )

    return x


# ==========================================================
# Walk-forward 逐周预测
# ==========================================================

def make_regression_model(alpha: float) -> Pipeline:
    """建立收益率预测模型。"""
    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            ("scaler", StandardScaler()),
            (
                "model",
                Ridge(alpha=alpha),
            ),
        ]
    )


def make_classification_model(c_value: float) -> Pipeline:
    """建立上涨概率预测模型。"""
    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=c_value,
                    max_iter=3000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def run_walk_forward_forecast(
    panel: pd.DataFrame,
    feature_cols: list[str],
    start_date: str,
    horizon: int,
    min_train_samples: int,
    ridge_alpha: float,
    logistic_c: float,
    interval_min_residuals: int,
) -> pd.DataFrame:
    """
    使用逐周扩展窗口进行历史预测。

    对信号日 t：
    1. 仅使用日期 <= t - (horizon + 1) 的已实现标签训练；
    2. 对 t 当天全部股票预测；
    3. 输出预期收益、上涨概率、预测区间；
    4. 最新信号日即使没有真实未来标签，也会正常输出预测。

    这避免了将未来尚未实现的收益标签泄露给模型。
    """
    x = panel[
        panel["date"] >= pd.Timestamp(start_date)
    ].copy()

    x = x.sort_values(["date", "code"]).reset_index(drop=True)

    all_dates = pd.DatetimeIndex(
        x["date"].drop_duplicates().sort_values()
    )

    signal_dates = weekly_last_trading_dates(all_dates)

    if len(signal_dates) == 0:
        return pd.DataFrame()

    date_to_position = {
        date: i for i, date in enumerate(all_dates)
    }

    records: list[pd.DataFrame] = []

    print(
        "\n正在执行 Walk-forward 逐周训练与预测，"
        "首次运行可能需要一些时间..."
    )

    for i, signal_date in enumerate(signal_dates):
        signal_pos = date_to_position[signal_date]

        # 信号日 t 的标签在 t+horizon+1 才完全实现。
        latest_train_pos = signal_pos - (horizon + 1)

        if latest_train_pos < 0:
            continue

        latest_train_date = all_dates[latest_train_pos]

        train = x[
            (x["date"] <= latest_train_date)
            & x["fwd_return"].notna()
        ].copy()

        train = train.dropna(
            subset=["target_up"]
        ).copy()

        if len(train) < min_train_samples:
            continue

        if train["target_up"].nunique() < 2:
            continue

        test_snapshot = x[
            x["date"] == signal_date
        ].copy()

        if test_snapshot.empty:
            continue

        x_train = train[feature_cols]
        y_return = train["fwd_return"].astype(float)
        y_up = train["target_up"].astype(int)

        regression_model = make_regression_model(
            alpha=ridge_alpha
        )
        classification_model = make_classification_model(
            c_value=logistic_c
        )

        regression_model.fit(x_train, y_return)
        classification_model.fit(x_train, y_up)

        test_snapshot["predicted_return"] = (
            regression_model.predict(
                test_snapshot[feature_cols]
            )
        )

        test_snapshot["up_probability"] = (
            classification_model.predict_proba(
                test_snapshot[feature_cols]
            )[:, 1]
        )

        # 用此前已实现的 Walk-forward 预测误差生成区间。
        # 不能使用当前或未来日期的误差，否则会形成未来函数。
        if records:
            past_forecasts = pd.concat(
                records,
                ignore_index=True,
            )

            resolved_cutoff = (
                signal_date
                - pd.Timedelta(days=horizon + 1)
            )

            past_residuals = past_forecasts[
                (past_forecasts["signal_date"] <= resolved_cutoff)
                & past_forecasts["fwd_return"].notna()
            ].copy()

            residuals = (
                past_residuals["fwd_return"]
                - past_residuals["predicted_return"]
            ).dropna()

        else:
            residuals = pd.Series(dtype=float)

        if len(residuals) >= interval_min_residuals:
            residual_low = residuals.quantile(0.10)
            residual_high = residuals.quantile(0.90)

            test_snapshot["return_interval_low"] = (
                test_snapshot["predicted_return"]
                + residual_low
            )

            test_snapshot["return_interval_high"] = (
                test_snapshot["predicted_return"]
                + residual_high
            )
        else:
            test_snapshot["return_interval_low"] = np.nan
            test_snapshot["return_interval_high"] = np.nan

        test_snapshot["signal_date"] = signal_date
        test_snapshot["train_end_date"] = latest_train_date
        test_snapshot["train_samples"] = len(train)

        records.append(test_snapshot)

        if (i + 1) % 25 == 0 or i == len(signal_dates) - 1:
            print(
                f"已处理 {i + 1}/{len(signal_dates)} 个周度信号日："
                f"{signal_date.date()}，"
                f"训练样本 {len(train):,}。"
            )

    if not records:
        return pd.DataFrame()

    result = pd.concat(records, ignore_index=True)

    result = result.sort_values(
        ["signal_date", "up_probability", "predicted_return"],
        ascending=[True, False, False],
    ).reset_index(drop=True)

    return result


# ==========================================================
# 模型评估与输出
# ==========================================================

def evaluate_forecasts(
    forecast_df: pd.DataFrame,
    split_date: str,
) -> pd.DataFrame:
    """
    评估历史已实现预测。

    指标：
    - MAE：预测收益和实际收益的平均绝对误差；
    - RMSE：大误差敏感的预测误差；
    - 收益相关性：预测收益与真实收益的 Pearson 相关；
    - 方向准确率：预测涨跌方向是否正确；
    - AUC：上涨概率排序能力；
    - Brier：上涨概率校准误差，越低越好。
    """
    if forecast_df.empty:
        return pd.DataFrame()

    split_ts = pd.Timestamp(split_date)

    resolved = forecast_df[
        forecast_df["fwd_return"].notna()
    ].copy()

    if resolved.empty:
        return pd.DataFrame()

    def summarize(
        frame: pd.DataFrame,
        name: str,
    ) -> dict:
        if frame.empty:
            return {
                "period": name,
                "samples": 0,
                "actual_mean_return": np.nan,
                "predicted_mean_return": np.nan,
                "mae": np.nan,
                "rmse": np.nan,
                "return_correlation": np.nan,
                "direction_accuracy": np.nan,
                "up_probability_auc": np.nan,
                "brier_score": np.nan,
                "actual_up_rate": np.nan,
                "mean_up_probability": np.nan,
            }

        actual = frame["fwd_return"].astype(float)
        predicted = frame["predicted_return"].astype(float)
        actual_up = frame["target_up"].astype(int)
        up_prob = frame["up_probability"].astype(float)

        if actual.nunique() >= 2 and predicted.nunique() >= 2:
            return_correlation = actual.corr(predicted)
        else:
            return_correlation = np.nan

        direction_accuracy = (
            ((predicted > 0).astype(int) == actual_up).mean()
        )

        if actual_up.nunique() >= 2:
            auc = roc_auc_score(actual_up, up_prob)
            brier = brier_score_loss(actual_up, up_prob)
        else:
            auc = np.nan
            brier = np.nan

        return {
            "period": name,
            "samples": len(frame),
            "actual_mean_return": actual.mean(),
            "predicted_mean_return": predicted.mean(),
            "mae": mean_absolute_error(actual, predicted),
            "rmse": np.sqrt(
                mean_squared_error(
                actual,
                predicted,
                )
            ),

            "return_correlation": return_correlation,
            "direction_accuracy": direction_accuracy,
            "up_probability_auc": auc,
            "brier_score": brier,
            "actual_up_rate": actual_up.mean(),
            "mean_up_probability": up_prob.mean(),
        }

    train_frame = resolved[
        resolved["signal_date"] < split_ts
    ].copy()

    test_frame = resolved[
        resolved["signal_date"] >= split_ts
    ].copy()

    return pd.DataFrame(
        [
            summarize(resolved, "全样本已实现预测"),
            summarize(train_frame, "训练期已实现预测"),
            summarize(test_frame, "样本外已实现预测"),
        ]
    )


def add_forecast_status(forecast_df: pd.DataFrame) -> pd.DataFrame:
    """
    根据预期收益、上涨概率和区间给出研究用途的状态标签。

    规则不是交易建议，仅用于使输出更容易阅读。
    """
    x = forecast_df.copy()

    conditions = [
        (x["up_probability"] >= 0.60)
        & (x["predicted_return"] > 0),
        (x["up_probability"] >= 0.52)
        & (x["predicted_return"] > 0),
        (x["up_probability"] < 0.45)
        & (x["predicted_return"] < 0),
    ]

    labels = [
        "偏多",
        "中性偏多",
        "偏空",
    ]

    x["forecast_status"] = np.select(
        conditions,
        labels,
        default="中性",
    )

    return x


def print_latest_forecast(
    forecast_df: pd.DataFrame,
    top_n: int,
) -> pd.DataFrame:
    """打印最新信号日的预测结果。"""
    if forecast_df.empty:
        print("\n没有可输出的预测结果。")
        return pd.DataFrame()

    latest_date = forecast_df["signal_date"].max()

    latest = forecast_df[
        forecast_df["signal_date"] == latest_date
    ].copy()

    latest = add_forecast_status(latest)

    latest = latest.sort_values(
        [
            "up_probability",
            "predicted_return",
            "return_interval_low",
        ],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    display = latest[
        [
            "code",
            "thscode",
            "close",
            "predicted_return",
            "up_probability",
            "return_interval_low",
            "return_interval_high",
            "forecast_status",
        ]
    ].head(top_n).copy()

    print("\n" + "=" * 100)
    print("【最新 PCB 股票池：未来收益概率预测】")
    print("=" * 100)
    print(f"信号日期：{latest_date.date()}")
    print(
        "预测口径：下一交易日开盘买入，"
        "持有 horizon 个交易日后开盘卖出。"
    )
    print(
        "\n说明：收益区间来自历史 Walk-forward 预测误差的 "
        "10% 至 90% 残差分位数；"
        "若显示 N/A，说明可用历史误差仍不足。"
    )

    print(
        "\n"
        + display.to_string(
            index=False,
            formatters={
                "close": lambda v: f"{v:.2f}",
                "predicted_return": format_pct,
                "up_probability": format_pct,
                "return_interval_low": format_pct,
                "return_interval_high": format_pct,
            },
        )
    )

    print("\n状态解释：")
    print("- 偏多：上涨概率至少 60%，且预测收益为正。")
    print("- 中性偏多：上涨概率至少 52%，且预测收益为正。")
    print("- 中性：模型没有形成明确方向。")
    print("- 偏空：上涨概率低于 45%，且预测收益为负。")
    print("\n以上输出仅为量化研究预测，不构成买卖建议。")

    return latest


# ==========================================================
# 主程序
# ==========================================================

# ==========================================================
# V1 可视化：K 线、预测路径、实际路径与统计检验
# ==========================================================

def load_single_stock_kline(
    db_path: Path,
    code: str,
) -> pd.DataFrame:
    """读取指定股票的前复权日 K 数据。"""
    con = duckdb.connect(str(db_path), read_only=True)

    try:
        sql = """
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
            WHERE regexp_extract(
                CAST(thscode AS VARCHAR),
                '[0-9]{6}'
            ) = ?
            ORDER BY date
        """

        df = con.execute(sql, [code]).fetchdf()
    finally:
        con.close()

    if df.empty:
        raise RuntimeError(f"数据库中找不到股票 {code} 的 K 线数据。")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(
        subset=["date", "open", "high", "low", "close"]
    ).copy()

    df = df[
        (df["open"] > 0)
        & (df["high"] > 0)
        & (df["low"] > 0)
        & (df["close"] > 0)
    ].copy()

    return df.sort_values("date").reset_index(drop=True)


def draw_candles(
    ax,
    kline: pd.DataFrame,
) -> None:
    """
    手工绘制 K 线：
    A 股习惯：红色上涨，绿色下跌。
    """
    for i, row in kline.reset_index(drop=True).iterrows():
        is_up = row["close"] >= row["open"]
        color = "#d62728" if is_up else "#2ca02c"

        # 上下影线
        ax.vlines(
            x=i,
            ymin=row["low"],
            ymax=row["high"],
            color=color,
            linewidth=0.8,
            alpha=0.9,
        )

        # K 线实体
        body_bottom = min(row["open"], row["close"])
        body_height = abs(row["close"] - row["open"])

        # 十字星也显示极小实体
        if body_height <= 0:
            body_height = max(row["close"] * 0.0005, 0.01)

        candle = Rectangle(
            xy=(i - 0.32, body_bottom),
            width=0.64,
            height=body_height,
            facecolor=color,
            edgecolor=color,
            linewidth=0.6,
            alpha=0.88,
        )

        ax.add_patch(candle)


def plot_kline_forecast_comparison(
    db_path: Path,
    forecast_df: pd.DataFrame,
    code: str,
    horizon: int,
    last_n_signals: int,
    output_file: Path,
) -> None:
    """
    绘制指定股票的 K 线、V1 预测路径与实际路径。

    图例：
    - 蓝色上三角：模型预测未来收益为正；
    - 紫色下三角：模型预测未来收益为负；
    - 蓝/紫虚线：模型预测的理论价格路径；
    - 绿色实线：实际持有期盈利；
    - 红色实线：实际持有期亏损。

    V1 交易定义：
    信号日 t 收盘后生成信号；
    t+1 开盘买入；
    t+horizon+1 开盘卖出。
    """
    forecasts = forecast_df.copy()
    forecasts["code"] = (
        forecasts["code"]
        .astype(str)
        .str.extract(r"(\d{6})", expand=False)
        .str.zfill(6)
    )

    forecasts["signal_date"] = pd.to_datetime(
        forecasts["signal_date"],
        errors="coerce",
    )

    stock_forecasts = forecasts[
        forecasts["code"] == code
    ].copy()

    stock_forecasts = stock_forecasts.sort_values(
        "signal_date"
    )

    # K 线对照必须优先使用真实收益已实现的预测记录。
    resolved = stock_forecasts[
        stock_forecasts["fwd_return"].notna()
    ].copy()

    if resolved.empty:
        raise RuntimeError(
            f"股票 {code} 没有已实现预测记录，无法绘制对比图。"
        )

    selected = resolved.tail(last_n_signals).copy()

    kline = load_single_stock_kline(
        db_path=db_path,
        code=code,
    )

    full_date_to_idx = {
        date: i
        for i, date in enumerate(kline["date"])
    }

    selected = selected[
        selected["signal_date"].isin(full_date_to_idx)
    ].copy()

    if selected.empty:
        raise RuntimeError(
            "预测信号日期与 K 线日期无法匹配。"
        )

    first_signal_idx = min(
        full_date_to_idx[d]
        for d in selected["signal_date"]
    )

    last_signal_idx = max(
        full_date_to_idx[d]
        for d in selected["signal_date"]
    )

    # 左侧保留约 100 个交易日作为趋势背景；
    # 右侧保留预测持有期后的若干交易日。
    chart_start_idx = max(0, first_signal_idx - 100)
    chart_end_idx = min(
        len(kline) - 1,
        last_signal_idx + horizon + 15,
    )

    chart = kline.iloc[
        chart_start_idx: chart_end_idx + 1
    ].copy().reset_index(drop=True)

    full_to_chart_idx = {
        full_idx: full_idx - chart_start_idx
        for full_idx in range(
            chart_start_idx,
            chart_end_idx + 1,
        )
    }

    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(19, 10))
    draw_candles(ax, chart)

    labels_drawn = {
        "positive_signal": False,
        "negative_signal": False,
        "actual_profit": False,
        "actual_loss": False,
    }

    latest_annotation_dates = set(
        selected["signal_date"].tail(5).tolist()
    )

    for _, row in selected.iterrows():
        signal_date = pd.Timestamp(row["signal_date"])
        signal_full_idx = full_date_to_idx[signal_date]

        entry_full_idx = signal_full_idx + 1
        exit_full_idx = signal_full_idx + horizon + 1

        if entry_full_idx >= len(kline):
            continue

        if exit_full_idx >= len(kline):
            continue

        if entry_full_idx not in full_to_chart_idx:
            continue

        if exit_full_idx not in full_to_chart_idx:
            continue

        entry_chart_idx = full_to_chart_idx[entry_full_idx]
        exit_chart_idx = full_to_chart_idx[exit_full_idx]

        entry_price = float(
            kline.iloc[entry_full_idx]["open"]
        )

        actual_exit_price = float(
            kline.iloc[exit_full_idx]["open"]
        )

        pred_return = float(row["predicted_return"])
        actual_return = float(row["fwd_return"])
        up_probability = float(row["up_probability"])

        predicted_exit_price = entry_price * (
            1.0 + pred_return
        )

        # 预测方向标记
        if pred_return >= 0:
            marker = "^"
            marker_color = "#1f77b4"
            label_key = "positive_signal"
            label_name = "预测收益为正"
        else:
            marker = "v"
            marker_color = "#9467bd"
            label_key = "negative_signal"
            label_name = "预测收益为负"

        ax.scatter(
            entry_chart_idx,
            entry_price,
            marker=marker,
            s=92,
            color=marker_color,
            edgecolor="white",
            linewidth=0.8,
            zorder=7,
            label=(
                label_name
                if not labels_drawn[label_key]
                else None
            ),
        )

        labels_drawn[label_key] = True

        # 虚线：模型预测路径
        ax.plot(
            [entry_chart_idx, exit_chart_idx],
            [entry_price, predicted_exit_price],
            linestyle="--",
            linewidth=1.5,
            color=marker_color,
            alpha=0.75,
            zorder=4,
        )

        # 实线：真实交易期路径
        if actual_return >= 0:
            actual_color = "#00a65a"
            actual_label_key = "actual_profit"
            actual_label = "实际持有期盈利"
        else:
            actual_color = "#e74c3c"
            actual_label_key = "actual_loss"
            actual_label = "实际持有期亏损"

        ax.plot(
            [entry_chart_idx, exit_chart_idx],
            [entry_price, actual_exit_price],
            linestyle="-",
            linewidth=2.5,
            color=actual_color,
            alpha=0.9,
            zorder=5,
            label=(
                actual_label
                if not labels_drawn[actual_label_key]
                else None
            ),
        )

        labels_drawn[actual_label_key] = True

        # 仅为最近 5 个信号标注具体预测值，避免文字过度拥挤。
        if signal_date in latest_annotation_dates:
            text = (
                f"预测 {pred_return:.1%}\n"
                f"概率 {up_probability:.0%}\n"
                f"实际 {actual_return:.1%}"
            )

            ax.annotate(
                text,
                xy=(entry_chart_idx, entry_price),
                xytext=(0, 20),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                color=marker_color,
                arrowprops={
                    "arrowstyle": "-",
                    "color": marker_color,
                    "alpha": 0.55,
                },
            )

    tick_step = max(len(chart) // 12, 1)
    tick_positions = list(range(0, len(chart), tick_step))
    tick_labels = [
        chart.iloc[i]["date"].strftime("%Y-%m-%d")
        for i in tick_positions
    ]

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(
        tick_labels,
        rotation=30,
        ha="right",
    )

    ax.set_xlim(-1, len(chart))
    ax.set_ylabel("前复权价格")
    ax.set_title(
        f"{code}：V1 Walk-forward 预测与历史 K 线对比\n"
        "虚线＝模型预测持有期终点；实线＝真实持有期终点",
        fontsize=15,
        fontweight="bold",
    )

    ax.grid(axis="y", linestyle=":", alpha=0.35)
    ax.legend(loc="best")
    ax.autoscale_view()

    fig.tight_layout()
    fig.savefig(
        output_file,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_prediction_scatter(
    forecast_df: pd.DataFrame,
    split_date: str,
    output_file: Path,
) -> None:
    """
    绘制预测收益与真实收益的散点图。

    左图：全部已实现 Walk-forward 预测；
    右图：样本外已实现预测。

    好模型的散点整体应呈左下到右上分布，
    且拟合线应向右上倾斜。
    """
    data = forecast_df[
        forecast_df["fwd_return"].notna()
    ].copy()

    if data.empty:
        return

    data["signal_date"] = pd.to_datetime(
        data["signal_date"],
        errors="coerce",
    )

    split_ts = pd.Timestamp(split_date)

    periods = [
        ("全样本已实现预测", data),
        (
            "样本外已实现预测",
            data[data["signal_date"] >= split_ts].copy(),
        ),
    ]

    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(18, 8),
    )

    for ax, (title, frame) in zip(axes, periods):
        if frame.empty:
            ax.set_title(f"{title}\n无数据")
            continue

        x = frame["predicted_return"].astype(float) * 100
        y = frame["fwd_return"].astype(float) * 100

        correlation = x.corr(y)

        colors = np.where(
            frame["target_up"].astype(int) == 1,
            "#d62728",
            "#2ca02c",
        )

        ax.scatter(
            x,
            y,
            c=colors,
            s=20,
            alpha=0.38,
            edgecolors="none",
        )

        lower = min(x.min(), y.min())
        upper = max(x.max(), y.max())
        padding = max((upper - lower) * 0.08, 1.0)

        lower -= padding
        upper += padding

        ax.plot(
            [lower, upper],
            [lower, upper],
            color="gray",
            linestyle="--",
            linewidth=1.2,
            label="理想线：预测 = 实际",
        )

        if x.nunique() > 1:
            slope, intercept = np.polyfit(x, y, 1)
            line_x = np.linspace(lower, upper, 100)

            ax.plot(
                line_x,
                slope * line_x + intercept,
                color="#1f77b4",
                linewidth=2.2,
                label=f"实际拟合线：斜率 {slope:.3f}",
            )

        ax.axhline(0, color="black", linewidth=0.8)
        ax.axvline(0, color="black", linewidth=0.8)

        ax.set_xlim(lower, upper)
        ax.set_ylim(lower, upper)
        ax.set_xlabel("模型预测未来 5 日收益（%）")
        ax.set_ylabel("真实未来 5 日收益（%）")
        ax.set_title(
            f"{title}\n"
            f"预测收益与实际收益相关系数：{correlation:.4f}",
            fontsize=13,
            fontweight="bold",
        )
        ax.grid(linestyle=":", alpha=0.4)
        ax.legend(loc="best")

    fig.suptitle(
        "V1：预测未来收益与真实未来收益的拟合检验",
        fontsize=16,
        fontweight="bold",
    )

    fig.tight_layout()
    fig.savefig(
        output_file,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_probability_group_test(
    forecast_df: pd.DataFrame,
    split_date: str,
    output_file: Path,
) -> None:
    """
    对样本外预测按上涨概率分 5 组。

    有效模型应表现为：
    从第 1 组到第 5 组，
    真实上涨率与真实平均收益总体上升。
    """
    data = forecast_df[
        forecast_df["fwd_return"].notna()
    ].copy()

    data["signal_date"] = pd.to_datetime(
        data["signal_date"],
        errors="coerce",
    )

    data = data[
        data["signal_date"] >= pd.Timestamp(split_date)
    ].copy()

    if data.empty or data["up_probability"].nunique() < 2:
        return

    try:
        data["probability_group"] = pd.qcut(
            data["up_probability"],
            q=5,
            duplicates="drop",
        )
    except ValueError:
        return

    group = data.groupby(
        "probability_group",
        observed=True,
    ).agg(
        samples=("fwd_return", "size"),
        avg_predicted_probability=(
            "up_probability",
            "mean",
        ),
        actual_up_rate=("target_up", "mean"),
        actual_mean_return=("fwd_return", "mean"),
    ).reset_index()

    if group.empty:
        return

    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax1 = plt.subplots(figsize=(13, 8))

    positions = np.arange(len(group))
    width = 0.34

    pred_bars = ax1.bar(
        positions - width / 2,
        group["avg_predicted_probability"] * 100,
        width=width,
        color="#1f77b4",
        alpha=0.85,
        label="平均预测上涨概率",
    )

    actual_bars = ax1.bar(
        positions + width / 2,
        group["actual_up_rate"] * 100,
        width=width,
        color="#ff7f0e",
        alpha=0.85,
        label="真实上涨率",
    )

    ax1.set_ylabel("上涨概率 / 真实上涨率（%）")
    ax1.set_ylim(0, 100)
    ax1.set_xticks(positions)
    ax1.set_xticklabels(
        [f"第 {i + 1} 组" for i in range(len(group))]
    )
    ax1.grid(axis="y", linestyle=":", alpha=0.4)

    ax2 = ax1.twinx()

    actual_return_line = ax2.plot(
        positions,
        group["actual_mean_return"] * 100,
        color="#2ca02c",
        marker="o",
        linewidth=2.2,
        label="真实平均未来收益",
    )

    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_ylabel("真实平均未来 5 日收益（%）")

    for bar in list(pred_bars) + list(actual_bars):
        height = bar.get_height()

        ax1.annotate(
            f"{height:.1f}%",
            xy=(
                bar.get_x() + bar.get_width() / 2,
                height,
            ),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    for i, value in enumerate(
        group["actual_mean_return"] * 100
    ):
        ax2.annotate(
            f"{value:.2f}%",
            xy=(i, value),
            xytext=(0, 10 if value >= 0 else -16),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color="#2ca02c",
        )

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()

    ax1.legend(
        h1 + h2,
        l1 + l2,
        loc="best",
    )

    ax1.set_title(
        "V1 样本外：上涨概率分组检验\n"
        "有效模型应使高概率组的真实上涨率、真实收益整体更高",
        fontsize=14,
        fontweight="bold",
    )

    fig.tight_layout()
    fig.savefig(
        output_file,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def generate_v1_visualizations(
    db_path: Path,
    forecast_df: pd.DataFrame,
    output_dir: Path,
    code: str,
    horizon: int,
    split_date: str,
    last_n_signals: int,
) -> None:
    """统一生成 V1 的三张可视化图片。"""
    if last_n_signals <= 0:
        raise ValueError("--plot-last-n 必须大于 0。")

    code = str(code).zfill(6)

    visualization_dir = output_dir / "visualizations"
    visualization_dir.mkdir(parents=True, exist_ok=True)

    kline_file = (
        visualization_dir
        / f"{code}_v1_kline_prediction_vs_actual.png"
    )

    scatter_file = (
        visualization_dir
        / "v1_predicted_vs_actual_scatter.png"
    )

    probability_file = (
        visualization_dir
        / "v1_oos_probability_group_test.png"
    )

    print("\n正在生成 V1 可视化图片...")

    plot_kline_forecast_comparison(
        db_path=db_path,
        forecast_df=forecast_df,
        code=code,
        horizon=horizon,
        last_n_signals=last_n_signals,
        output_file=kline_file,
    )

    plot_prediction_scatter(
        forecast_df=forecast_df,
        split_date=split_date,
        output_file=scatter_file,
    )

    plot_probability_group_test(
        forecast_df=forecast_df,
        split_date=split_date,
        output_file=probability_file,
    )

    print("可视化图片已生成：")
    print(f"1. K 线 + 预测 + 实际路径：{kline_file}")
    print(f"2. 预测收益与实际收益散点图：{scatter_file}")
    print(f"3. 样本外概率分组检验图：{probability_file}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "PCB / 电子产业链："
            "未来收益、上涨概率与收益区间预测模型"
        )
    )

    parser.add_argument(
        "--db",
        required=True,
        help=(
            "本地 DuckDB 数据库文件路径。"
            "例如：C:\\path\\to\\market.duckdb"
        ),
    )


    parser.add_argument(
        "--start",
        default="2021-01-01",
        help="正式研究起点，默认 2021-01-01。",
    )

    parser.add_argument(
        "--end",
        default=None,
        help="研究终点；默认读取数据库最新日期。",
    )

    parser.add_argument(
        "--horizon",
        type=int,
        default=5,
        help="预测持有期，单位为交易日，默认 5。",
    )

    parser.add_argument(
        "--min-train-samples",
        type=int,
        default=500,
        help="每次训练所需最少历史样本数，默认 500。",
    )

    parser.add_argument(
        "--ridge-alpha",
        type=float,
        default=10.0,
        help="Ridge 正则化系数，默认 10.0。",
    )

    parser.add_argument(
        "--logistic-c",
        type=float,
        default=0.20,
        help=(
            "逻辑回归 C 参数，越小正则化越强，"
            "默认 0.20。"
        ),
    )

    parser.add_argument(
        "--interval-min-residuals",
        type=int,
        default=150,
        help=(
            "计算预测区间所需最少历史预测残差数，"
            "默认 150。"
        ),
    )

    parser.add_argument(
        "--split-date",
        default="2025-01-01",
        help="样本外评估起点，默认 2025-01-01。",
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=17,
        help="最新预测表显示前几只股票，默认全部 17 只。",
    )

    parser.add_argument(
        "--out-dir",
        default=None,
        help=(
            "输出目录；默认使用数据库目录下的 "
            "pcb_future_forecast_output。"
        ),
    )

    parser.add_argument(
        "--plot-code",
        default=None,
        help=(
            "可选：生成指定股票的 K 线与预测对比图。"
            "例如 --plot-code 300476。"
        ),
    )

    parser.add_argument(
        "--plot-last-n",
        type=int,
        default=12,
        help="K 线图显示最近多少个已实现周度预测信号，默认 12。",
    )

    parser.add_argument(
        "--plot-all",
        action="store_true",
        help=(
            "生成股票池内全部股票的 K 线、历史预测和真实结果对比图。"
        ),
    )




    args = parser.parse_args()

    if args.horizon <= 0:
        raise ValueError("--horizon 必须是正整数。")

    if args.min_train_samples < 100:
        raise ValueError(
            "--min-train-samples 不应低于 100。"
        )

    if args.ridge_alpha <= 0:
        raise ValueError("--ridge-alpha 必须大于 0。")

    if args.logistic_c <= 0:
        raise ValueError("--logistic-c 必须大于 0。")

    if args.interval_min_residuals <= 0:
        raise ValueError(
            "--interval-min-residuals 必须是正整数。"
        )

    if args.top_n <= 0:
        raise ValueError("--top-n 必须是正整数。")

    if args.end is not None:
        if pd.Timestamp(args.end) < pd.Timestamp(args.start):
            raise ValueError("--end 不能早于 --start。")

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
            / "pcb_future_forecast_output"
        )
    else:
        output_dir = Path(args.out_dir).expanduser().resolve()


    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("PCB / 电子产业链：未来收益概率预测模型")
    print("=" * 100)
    print(f"数据库：{db_path}")
    print(f"股票池数量：{len(PCB_UNIVERSE)}")
    print(
        f"正式研究区间：{args.start} 至 "
        f"{args.end or '数据库最新日期'}"
    )
    print(f"预测持有期：{args.horizon} 个交易日")
    print(f"最低训练样本数：{args.min_train_samples}")
    print(f"Ridge alpha：{args.ridge_alpha}")
    print(f"LogisticRegression C：{args.logistic_c}")
    print(f"样本外评估起点：{args.split_date}")
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
            "可用股票数量少于 8，无法进行稳定的股票池研究。"
        )

    print("\n正在构建量价、趋势、风险和流动性特征...")

    panel = build_feature_panel(
        daily_data=daily_data,
        horizon=args.horizon,
    )

    panel = add_cross_section_features(panel)

    # 去除用于因子预热的正式起点以前记录。
    panel = panel[
        panel["date"] >= pd.Timestamp(args.start)
    ].copy()

    # 使用截至信号日可获得的信息预测未来收益。
    feature_cols = [
        "momentum_5",
        "momentum_10",
        "momentum_20",
        "momentum_60",
        "close_ma5_gap",
        "close_ma20_gap",
        "close_ma60_gap",
        "ma20_ma60_gap",
        "breakout_60",
        "position_20_60",
        "turnover_z20",
        "volume_z20",
        "turnover_ratio_5_20",
        "volume_price_confirmation",
        "volume_price_trend_5",
        "volatility_5",
        "volatility_20",
        "drawdown_20",
        "amihud_20",
        "macd_dif",
        "macd_hist",
        "relative_momentum_20",
        "momentum_20_cs_rank",
        "breakout_60_cs_rank",
        "volatility_20_cs_rank",
        "turnover_cs_rank",
    ]

    feature_cols = [
        col for col in feature_cols
        if col in panel.columns
    ]

    if not feature_cols:
        raise RuntimeError("没有可用特征，无法训练模型。")

    print(f"可用特征数量：{len(feature_cols)}")
    print("特征列表：")
    print(", ".join(feature_cols))

    forecast_df = run_walk_forward_forecast(
        panel=panel,
        feature_cols=feature_cols,
        start_date=args.start,
        horizon=args.horizon,
        min_train_samples=args.min_train_samples,
        ridge_alpha=args.ridge_alpha,
        logistic_c=args.logistic_c,
        interval_min_residuals=args.interval_min_residuals,
    )

    if forecast_df.empty:
        raise RuntimeError(
            "Walk-forward 预测结果为空。"
            "可尝试将 --min-train-samples 从 500 降至 300，"
            "或检查数据覆盖情况。"
        )

    forecast_df = add_forecast_status(forecast_df)

    evaluation_df = evaluate_forecasts(
        forecast_df=forecast_df,
        split_date=args.split_date,
    )

    print("\n" + "=" * 100)
    print("【历史 Walk-forward 预测评估】")
    print("=" * 100)

    if evaluation_df.empty:
        print("尚无可评估的已实现预测。")
    else:
        print(
            evaluation_df.to_string(
                index=False,
                formatters={
                    "actual_mean_return": format_pct,
                    "predicted_mean_return": format_pct,
                    "mae": format_pct,
                    "rmse": format_pct,
                    "return_correlation": lambda v: (
                        f"{v:.4f}" if pd.notna(v) else "N/A"
                    ),
                    "direction_accuracy": format_pct,
                    "up_probability_auc": lambda v: (
                        f"{v:.4f}" if pd.notna(v) else "N/A"
                    ),
                    "brier_score": lambda v: (
                        f"{v:.4f}" if pd.notna(v) else "N/A"
                    ),
                    "actual_up_rate": format_pct,
                    "mean_up_probability": format_pct,
                },
            )
        )

        print("\n指标解释：")
        print("- MAE / RMSE：预测收益与真实收益的误差，越低越好。")
        print("- 收益相关性：预测收益与真实收益的同向关系，越高越好。")
        print("- 方向准确率：预测涨跌是否与真实涨跌一致。")
        print("- AUC：上涨概率的区分能力，0.50 接近随机，越高越好。")
        print("- Brier Score：上涨概率误差，越低越好。")

    latest_df = print_latest_forecast(
        forecast_df=forecast_df,
        top_n=args.top_n,
    )

    # ------------------------------------------------------
    # 导出文件
    # ------------------------------------------------------
    feature_panel_file = (
        output_dir / "pcb_forecast_feature_panel.csv"
    )

    forecast_file = (
        output_dir / "pcb_walk_forward_forecasts.csv"
    )

    evaluation_file = (
        output_dir / "pcb_forecast_evaluation.csv"
    )

    latest_file = (
        output_dir / "pcb_latest_forecast.csv"
    )

    feature_list_file = (
        output_dir / "pcb_forecast_feature_list.csv"
    )

    panel.to_csv(
        feature_panel_file,
        index=False,
        encoding="utf-8-sig",
    )

    forecast_df.to_csv(
        forecast_file,
        index=False,
        encoding="utf-8-sig",
    )

    evaluation_df.to_csv(
        evaluation_file,
        index=False,
        encoding="utf-8-sig",
    )

    latest_df.to_csv(
        latest_file,
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        {
            "feature": feature_cols
        }
    ).to_csv(
        feature_list_file,
        index=False,
        encoding="utf-8-sig",
    )


    # ------------------------------------------------------
    # 可选：生成 K 线与 V1 历史预测对比图
    #
    # --plot-code 300476：
    #     只生成 300476 一只股票的 K 线图，
    #     同时生成全局散点图、概率分组图。
    #
    # --plot-all：
    #     为股票池全部股票生成 K 线对比图，
    #     全局散点图、概率分组图只生成一次。
    # ------------------------------------------------------
    visualization_dir = output_dir / "visualizations"

    if args.plot_all:
        visualization_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 100)
        print("【正在生成全部股票的 K 线预测对比图】")
        print("=" * 100)

        available_codes = sorted(
            forecast_df["code"]
            .astype(str)
            .str.extract(r"(\d{6})", expand=False)
            .dropna()
            .unique()
        )

        for i, code in enumerate(available_codes, start=1):
            output_file = (
                visualization_dir
                / f"{code}_v1_kline_prediction_vs_actual.png"
            )

            try:
                print(
                    f"正在生成 {i}/{len(available_codes)}："
                    f"{code}"
                )

                plot_kline_forecast_comparison(
                    db_path=db_path,
                    forecast_df=forecast_df,
                    code=code,
                    horizon=args.horizon,
                    last_n_signals=args.plot_last_n,
                    output_file=output_file,
                )

            except Exception as exc:
                print(
                    f"警告：股票 {code} 的 K 线图生成失败：{exc}"
                )

        print("\n正在生成全股票池统计图...")

        plot_prediction_scatter(
            forecast_df=forecast_df,
            split_date=args.split_date,
            output_file=(
                visualization_dir
                / "v1_predicted_vs_actual_scatter.png"
            ),
        )

        plot_probability_group_test(
            forecast_df=forecast_df,
            split_date=args.split_date,
            output_file=(
                visualization_dir
                / "v1_oos_probability_group_test.png"
            ),
        )

        print("\n全部可视化图片已保存至：")
        print(visualization_dir)

    elif args.plot_code is not None:
        generate_v1_visualizations(
            db_path=db_path,
            forecast_df=forecast_df,
            output_dir=output_dir,
            code=args.plot_code,
            horizon=args.horizon,
            split_date=args.split_date,
            last_n_signals=args.plot_last_n,
        )



    print("\n" + "=" * 100)
    print("【程序运行完成】")
    print("=" * 100)
    print(f"特征面板：{feature_panel_file}")
    print(f"历史 Walk-forward 预测明细：{forecast_file}")
    print(f"模型预测评估：{evaluation_file}")
    print(f"最新预测结果：{latest_file}")
    print(f"特征列表：{feature_list_file}")

    print("\n重要说明：")
    print(
        "1. 最新一个信号日没有真实未来收益标签，"
        "这是正常现象；它就是模型对未来的当前预测。"
    )
    print(
        "2. 训练时只使用在信号日之前已经完全实现的未来收益标签，"
        "避免未来函数。"
    )
    print(
        "3. 预测区间不是价格保证，而是根据历史预测误差估计的"
        "不确定性范围。"
    )
    print(
        "4. 若样本外 AUC 长期接近 0.50、收益相关性接近 0，"
        "说明模型接近随机预测，不应据此交易。"
    )
    print("5. 所有输出仅用于量化研究，不构成投资建议。")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())