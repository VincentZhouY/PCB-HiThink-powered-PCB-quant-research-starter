# 同花顺金融数据 · 量化开发接入包

> 交给量化开发者的一份完整起步材料:把同花顺官方金融数据接进你的机器,
> 外加一个能立刻跑起来的端到端回测脚手架。
>
> 数据来源:同花顺官方 `hithink-finance`(GitHub 仓库 `HiThink-Tech/Financial-API`)。
> **API Key 需要你自己申请**(见第 3 节),本包内不含任何凭据。

---

## 0. 快速开始

四条命令跑通全流程(细节见后文):

```bash
git clone https://github.com/HiThink-Tech/Financial-API.git && cd Financial-API
python -m pip install -e ./python          # 装 SDK
python python/bootstrap.py                 # 建本地历史库(约 800 MB,耗时较长)
python <本包>/quant_starter.py --db data/market.duckdb
```

---

## 1. 环境要求

| 项 | 要求 | 检查 |
| --- | --- | --- |
| Python | **>= 3.11** | `python --version` |
| git | 拉仓库 | `git --version` |
| 磁盘 | 建议预留 **3 GB** | 本地库约 800 MB |
| 网络 | 首次装包需访问 `github.com` / `pypi.org` | |

依赖包(`duckdb` / `pandas` / `numpy`)会随 `pip install -e ./python` 一起装上。
**不需要 scipy** —— 脚手架里的秩相关用 pandas 原生实现。

---

## 2. 安装

```bash
git clone https://github.com/HiThink-Tech/Financial-API.git
cd Financial-API
python -m pip install -e ./python
```

国内网络可加镜像:

```bash
python -m pip install -e ./python -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> 这套 Python SDK **没有独立 PyPI 包**,必须从 monorepo 源码安装。

---

## 3. 配置 API Key(自己申请)

**Key 获取地址:<https://fuyao.aicubes.cn/admin>**

同一个 Key 通用 —— REST API / MCP / CLI / Python SDK 共用,不需要重复申请。

拿到后**用环境变量配置**:

**Windows(PowerShell)** —— 设完必须重开终端:

```powershell
[Environment]::SetEnvironmentVariable('HITHINK_FINANCE_API_KEY', '<你的Key>', 'User')
```

**macOS / Linux:**

```bash
echo 'export HITHINK_FINANCE_API_KEY=<你的Key>' >> ~/.bashrc && source ~/.bashrc
```

或者放进项目根的 `.env`(本包已附 `.env.example`,**务必确认 `.gitignore` 已忽略 `.env`**)。

> Python 侧也会读取用户级凭据文件 `hithink-finance/credentials.env`;
> `FUYAO_TOKEN` / `API_KEY` 是旧版兼容来源,新项目统一用 `HITHINK_FINANCE_API_KEY`。

---

## 4. 验证是否接通

```bash
# 先确认环境变量生效
python -c "import os; print('SET' if os.environ.get('HITHINK_FINANCE_API_KEY') else 'MISSING')"

# 再打一个真实远端请求
python python/toolkit/fuyao/scripts/fuyao.py prices-snapshot --thscodes 600519.SH
```

**返回贵州茅台的行情记录 = 通了。** 退出码 0、返回里有真实记录,才算成功;
只看 `--help` 通过不能证明 Key 有数据权限。

---

## 5. 数据能力

### 本地(历史行情,做回测用这个)

`bootstrap.py` 建好的 DuckDB 提供这些视图:

| 视图 | 含义 |
| --- | --- |
| `v_daily_qfq` | **前复权**日线 —— 回测默认用这个 |
| `v_daily_hfq` | 后复权日线 |
| `v_daily` | 不复权原始日线 |
| `v_symbol` | 标的目录 |

参考规模(实测):**约 1027 万行日线 × 5560 只标的**,覆盖 2016-09 ~ 2026-09。

### 远端(实时 / 基本面)

财报、估值、实时行情、指数成分、基金、涨跌停/龙虎榜等特色数据,走 `fuyao` 远端接口:

```bash
FUYAO=python python/toolkit/fuyao/scripts/fuyao.py

$FUYAO tickers-search      --q "宁德时代"              # 代码检索
$FUYAO prices-snapshot     --thscodes 600519.SH        # 实时行情
$FUYAO financials-income   --thscode 600519.SH --limit 4   # 利润表
$FUYAO valuations-snapshot --thscodes 600519.SH        # 估值指标
$FUYAO --help                                          # 全部能力
```

> 远端接口有**额度/频率限制**。批量、多年、全市场的取数**一律走本地库**,
> 不要写循环去打远端。

---

## 6. 三种数据访问方式

### A. `marketdb` Python SDK(官方推荐)

```python
from marketdb import MarketDB

with MarketDB.open("data/market.duckdb") as db:
    df    = db.get_daily("600519.SH", start="2025-01-01", adjust="forward")   # 单股
    batch = db.get_daily(["600519.SH", "300750.SZ"], adjust="forward")        # 批量
    panel = db.get_panel(start="2026-01-01", end="2026-01-31")                # 全市场截面
    out   = db.query_sql("SELECT thscode, AVG(amount) FROM v_daily_qfq GROUP BY 1")  # 任意 SQL
```

### B. 直接查 DuckDB 视图(脚手架用的就是这种)

```python
import duckdb
con = duckdb.connect("data/market.duckdb", read_only=True)
df = con.execute("SELECT * FROM v_daily_qfq WHERE thscode='600519.SH'").df()
```

### C. `marketdb` CLI

```bash
marketdb status   --json --db data/market.duckdb   # 状态
marketdb describe --db data/market.duckdb          # 自动探查 schema
marketdb validate --json --db data/market.duckdb   # 数据质量
marketdb auto-sync --db data/market.duckdb         # 增量更新(自动判 FULL/INCREMENTAL)
marketdb query --json --sql "SELECT ... FROM v_daily_qfq"
```

**不知道有什么表/字段时,先跑 `marketdb describe`。**

---

## 7. 脚手架 `quant_starter.py`

端到端链路,每一层都是独立函数,可以逐层替换:

| 层 | 函数 | 说明 |
| --- | --- | --- |
| 数据 | `DataHub` | DuckDB 只读封装(可直接换成 `MarketDB`) |
| 因子 | `factor_momentum` / `factor_reversal` / `factor_volatility` / `factor_ma_ratio` / `factor_ic` | 每个因子 = 一个 `date × thscode` 矩阵 |
| 信号 | `signal_threshold` / `signal_cross_section_rank` / `signal_ma_cross` | 因子 → 0/1 持仓 |
| 组合 | `weight_equal` / `weight_inverse_vol` / `weight_risk_parity` | 持仓 → 权重 |
| 风控 | `risk_max_weight` / `risk_vol_target` / `risk_turnover_buffer` | 单票上限 / 波动率目标 / 调仓缓冲 |
| 回测 | `backtest` / `performance` | 含成本、防前视偏差、9 项绩效指标 |

```bash
python quant_starter.py --db data/market.duckdb
python quant_starter.py --db data/market.duckdb --start 2022-01-01 --top-n 120 --cost 0.0005
```

### 基线实测结果(供你对照)

```
区间 2021-01-04 ~ 2026-09-10   股票池 80 只   平均仓位 62.5%   年化换手 56.9x

策略(20日动量+波动率加权+风控)   累计 +14.94%  年化 +2.48%  夏普 0.03  最大回撤 -37.74%
等权买入持有(基准)              累计  +6.59%  年化 +1.13%  夏普 -0.04 最大回撤 -40.07%

因子体检(20 日动量,5 日前瞻 IC): IC 均值 -0.0223   IR -0.090   胜率 48.2%
```

**注意 IC 是负的** —— 说明这个池子上 20 日动量其实不如反转,基线本身就没跑赢多少。
这是刻意留的起点:它跑得通、指标齐、但**不够好**,改进空间明确。

---

## 8. 你的任务范围

### ① 策略信号逻辑

- 起点:`signal_*()`
- 方向:IC 加权合成多因子、机器学习打分(树模型 / 线性)、事件驱动(财报超预期、涨停打开)、日内与隔夜拆分
- 验收:信号层改动后,IC / IR 与回测夏普同时改善

### ② 因子挖掘

- 起点:`factor_*()` + `factor_ic()`
- 方向:量价(动量/反转/换手/波动/量价背离)、基本面(PB/ROE/增速,**走远端接口**)、行业与风格中性化、因子正交
- 验收:新因子单独 IC 均值长期 > 0.03 且 IR 稳定;与已有因子相关性不过高

### ③ 回测系统

- 起点:`backtest()` / `performance()`
- 方向:逐笔撮合、涨跌停/停牌不可成交、冲击成本模型、分行业归因、参数敏感性扫描
- 验收:**无前视偏差**(见第 9 节自检),结果可复现,成本假设可调

### ④ 风控 / 组合优化

- 起点:`risk_*()` / `weight_*()`
- 方向:行业与风格中性约束、回撤止损、流动性约束(单票持仓不超过日均成交额比例)、均值方差 / Black-Litterman / 带约束二次规划
- 验收:在收益不显著恶化的前提下,**最大回撤与换手**下降

---

## 9. 口径与陷阱(最容易翻车的地方)

### 复权

回测**必须用前复权**(`v_daily_qfq` / `adjust="forward"`)。
用不复权价格算收益,遇到除权除息会产生假跳空,年化收益直接算错。

### 前视偏差 —— 最容易凭空造出收益的坑

两个必查点:

1. **选股池不能用回测期内的数据。** 脚手架里 `liquidity_universe(as_of=...)` 只用 `as_of` 之前的数据。
   实测对比:同一策略、同一数据,只把选池时点从"全样本"改成"每年用上一年",累计收益从 **+266% 掉到 +67%**。
2. **信号不能当日生效。** `backtest(lag=1)` 表示 T 日收盘出信号、T+1 日才持有。
   冒烟测试实测:`lag=1` 净值 0.83,`lag=0` 净值 4.37 —— **改一个参数,收益差 5 倍,全是假的**。

**不要动 `lag` 的默认值。**

### 幸存者偏差

本地库只含**当前仍在交易**的标的,已退市个股缺失。回测收益会偏乐观,这个无法在库内修复,
请在结论里如实标注。

### 成本与换手

基线年化换手高达 56.9x(平均持仓约 4 天)。成本已按单边万分之三计入,但真实还要考虑
冲击成本、涨跌停无法成交。**优化时把换手当作一等指标看**,别只看收益。

### 其它

- 未计入:分红税、停牌期间资金占用
- 所有输出请标注**来源、时间、复权口径**,并声明**非投资建议**(官方安全要求)

---

## 10. 安全红线

官方 `AGENTS.md` 明确要求:

1. ❌ 不要把 Key 写进 `.py` / `.ipynb` / README / 日志 / Git
2. ❌ 不要用 `--api-key <value>` 明文参数(会进 shell 历史和进程列表)
3. ✅ 只用:交互式隐藏输入、`HITHINK_FINANCE_API_KEY` 环境变量、`--api-key-stdin`、系统凭据库
4. ✅ 发 Issue / 贴日志前,先确认里面没有 Key

本包附了 `.gitignore`(已忽略 `.env`)和 `.env.example`(占位符)。**提交代码前扫一眼。**

---

## 11. 参考文档

| 需要了解 | 位置 |
| --- | --- |
| 本地行情 / SDK / SQL 视图 | `python/toolkit/marketdb/README.md` |
| 远端行情、财报、估值 | `python/toolkit/fuyao/README.md` |
| 可运行示例(单股 / 截面 / 财报 join) | `python/examples/` |
| 统一 Skill(给 AI agent 用) | `skills/hithink-finance/SKILL.md` |
| 完整 API 参数与字段 | `docs/api/` |
| Key 管理 | <https://fuyao.aicubes.cn/admin> |
