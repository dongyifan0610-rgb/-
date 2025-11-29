"""
量化因子挖掘流程
==============================

本模块整合了读取股票数据、生成基础因子、调用大模型生成变异因子与交叉因子、
计算信息系数（IC）以及保存结果的完整流程，并添加了中文注释以方便理解。

模块主要包含以下部分：

1. **配置类（Config）**：用来指定数据文件路径、模型参数、评估参数等。
2. **数据读取函数（read_stock_data_from_excel）**：读取一个或多个 Excel/CSV 文件，
   清洗并转换为宽表格式（行是日期、列是股票代码、值是指标）。
3. **基础因子库（BASE_FACTORS）和 Alpha101 因子库（ALPHA101_FACTORS）**：
   提供一组简单因子和完整的 Alpha101 因子表达式供后续评估。
4. **大模型客户端（QwenClient）**：封装阿里云千问模型的调用，用于生成变异因子和交叉因子。
5. **因子生成函数（generate_new_factors）**：利用大模型生成基于基础因子的变异因子和交叉因子，并进行去重。
6. **因子评估函数（calculate_ic）**：计算横截面或时间序列的信息系数，用于衡量因子的预测能力。
7. **辅助函数**：如因子表达式解析、去重、时间序列滚动运算等。
8. **主函数（main）**：串联整个流程：读取数据、初始化模型、生成因子、评估因子并保存结果。

如何运行
--------

1. 准备好包含股票日线数据的 Excel 或 CSV 文件，确保有列：日期、代码、开盘价、最高价、最低价、收盘价、成交量。
2. 修改 ``Config`` 中的 ``excel_path`` 或 ``excel_paths`` 为实际文件路径。
3. 若需要调用大模型生成因子，请在 ``create_qwen_client`` 中设置你的 API key。
4. 在命令行运行 ``python factor_pipeline_cn.py`` 即可执行完整流程。程序会在控制台打印进度，并将评估结果保存为 Excel 文件。

本代码示例仅供学习参考，实际使用时可根据需要调整参数和功能。
"""

from __future__ import annotations

import math
import os
import random
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests


@dataclass
class Config:
    """配置参数类

    主要控制数据路径、模型参数、评估标准等。可以根据需要修改默认值。
    """

    # 单个文件路径（可选）
    excel_path: str = ""
    # 多文件路径列表，如果设置则优先生效；支持一次读取多个股票的数据文件
    excel_paths: List[str] = field(default_factory=list)

    # 因子生成参数：控制变异因子、交叉因子的数量
    n_mutations: int = 3  # 每个基础因子生成多少个变异因子
    n_crossovers: int = 5  # 总共生成多少个交叉因子

    # 评估参数：控制 IC 的有效性判定和最小样本数
    ic_threshold: float = 0.02  # 判断因子有效的 IC 阈值
    min_samples_for_ic: int = 3  # 每日计算 IC 的最小股票数量

    # 模型参数（调用千问时使用）
    model_name: str = "qwen-turbo"
    temperature: float = 0.6
    max_tokens: int = 2000

    # 因子去重阈值（控制相似因子合并）
    factor_similarity_threshold: float = 0.9

    def get_excel_paths(self) -> List[str]:
        """返回要读取的文件路径列表。若 ``excel_paths`` 不为空则使用该列表，否则使用 ``excel_path``。"""
        # 优先使用 excel_paths；如果为空则退化到单个 excel_path；两者都没配置则返回空列表
        if self.excel_paths:
            return self.excel_paths
        if self.excel_path:
            return [self.excel_path]
        return []


def read_stock_data_from_excel(cfg: Config) -> Dict[str, pd.DataFrame]:
    """读取并清洗股票数据，返回宽表格式的多字段数据。

    参数
    ----
    cfg: Config
        配置对象，包含数据文件路径。

    返回
    ----
    Dict[str, pd.DataFrame]
        键为字段名（open, high, low, close, volume），值为以日期为索引、股票代码为列的 DataFrame。
    """
    # 取得所有需要读取的文件路径
    paths = cfg.get_excel_paths()
    if not paths:
        raise ValueError("未指定任何 Excel/CSV 文件路径，请检查配置")

    # 定义文件中各列对应的中文列名，方便后续统一改名
    excel_columns = {
        "date": "日期",
        "stock": "代码",
        "open": "开盘价(元)",
        "high": "最高价(元)",
        "low": "最低价(元)",
        "close": "收盘价(元)",
        "volume": "成交量(股)",
    }

    # 逐个文件读取并放入 frames
    frames: List[pd.DataFrame] = []
    for path in paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"找不到文件：{path}")
        ext = os.path.splitext(path)[1].lower()
        usecols = list(excel_columns.values())
        if ext == ".csv":
            # CSV 读取：显式指定编码、日期列、股票代码类型
            df = pd.read_csv(
                path,
                usecols=usecols,
                parse_dates=[excel_columns["date"]],
                dtype={excel_columns["stock"]: str},
                encoding="utf-8-sig",
            )
        else:
            # Excel 读取：使用 openpyxl 引擎保持兼容性
            df = pd.read_excel(
                path,
                usecols=usecols,
                parse_dates=[excel_columns["date"]],
                dtype={excel_columns["stock"]: str},
                engine="openpyxl",
            )
        frames.append(df)

    # 合并所有文件；忽略原行索引并统一列名为英文 key
    raw = pd.concat(frames, ignore_index=True)
    raw = raw.rename(columns={v: k for k, v in excel_columns.items()})

    # 统一股票代码格式：沪市补 .SH，深市补 .SZ，其余保持原样
    def fmt(code: str) -> str:
        code = str(code).strip()
        if code.startswith(("60", "68")):
            return f"{code}.SH"
        if code.startswith(("00", "30")):
            return f"{code}.SZ"
        return code

    # 应用代码格式化，并过滤掉价格或成交量为非正数的异常行
    raw["stock"] = raw["stock"].apply(fmt)
    valid = (raw[["open", "high", "low", "close", "volume"]] > 0).all(axis=1)
    raw = raw.loc[valid].dropna(subset=["date", "stock"]).copy()

    # 宽表输出：逐字段透视，行=日期，列=股票代码，值=对应字段
    wide: Dict[str, pd.DataFrame] = {}
    for field in ["open", "high", "low", "close", "volume"]:
        pivot = raw.pivot(index="date", columns="stock", values=field)
        wide[field] = pivot.sort_index()
    return wide


BASE_FACTORS: List[str] = [
    "rank(zscore(momentum(close, 20)))",
    "rank(zscore(volume / rolling_mean(volume, 20)))",
    "rank(zscore((close - open) / open))",
    "rank(zscore((high - low) / open))",
    "rank(zscore(close / high))",
    "rank(zscore(open / low))",
]

ALPHA101_FACTORS: Dict[str, str] = {
    "alpha001": "rank(ts_argmax(signedpower(((returns < 0) ? stddev(returns, 20) : close), 2), 5)) - 0.5",
    "alpha002": "-1 * correlation(rank(delta(log(volume), 2)), rank(((close - open) / open)), 6)",
    "alpha003": "-1 * correlation(rank(open), rank(volume), 10)",
    "alpha004": "-1 * ts_rank(rank(low), 9)",
    "alpha005": "rank((open - (sum(vwap, 10) / 10))) * (-1 * abs(rank((close - vwap))))",
    "alpha006": "-1 * correlation(open, volume, 10)",
    "alpha007": "(adv20 < volume) ? ((-1 * ts_rank(abs(delta(close, 7)), 60)) * sign(delta(close, 7))) : (-1 * 1)",
    "alpha008": "-1 * rank(((sum(open, 5) * sum(returns, 5)) - delay((sum(open, 5) * sum(returns, 5)), 10)))",
    "alpha009": "(0 < ts_min(delta(close, 1), 5)) ? delta(close, 1) : ((ts_max(delta(close, 1), 5) < 0) ? delta(close, 1) : (-1 * delta(close, 1)))",
    "alpha010": "rank((0 < ts_min(delta(close, 1), 4)) ? delta(close, 1) : ((ts_max(delta(close, 1), 4) < 0) ? delta(close, 1) : (-1 * delta(close, 1))))",
    "alpha011": "(rank(ts_max((vwap - close), 3)) + rank(ts_min((vwap - close), 3))) * rank(delta(volume, 3))",
    "alpha012": "sign(delta(volume, 1)) * (-1 * delta(close, 1))",
    "alpha013": "-1 * rank(covariance(rank(close), rank(volume), 5))",
    "alpha014": "(-1 * rank(delta(returns, 3))) * correlation(open, volume, 10)",
    "alpha015": "-1 * sum(rank(correlation(rank(high), rank(volume), 3)), 3)",
    "alpha016": "-1 * rank(covariance(rank(high), rank(volume), 5))",
    "alpha017": "((-1 * rank(ts_rank(close, 10))) * rank(delta(delta(close, 1), 1))) * rank(ts_rank((volume / adv20), 5))",
    "alpha018": "-1 * rank(((stddev(abs((close - open)), 5) + (close - open)) + correlation(close, open, 10)))",
    "alpha019": "(-1 * sign(((close - delay(close, 7)) + delta(close, 7)))) * (1 + rank((1 + sum(returns, 250))))",
    "alpha020": "((-1 * rank((open - delay(high, 1)))) * rank((open - delay(close, 1)))) * rank((open - delay(low, 1)))",
    "alpha021": "(((sum(close, 8) / 8) + stddev(close, 8)) < (sum(close, 2) / 2)) ? (-1 * 1) : (((sum(close, 2) / 2) < ((sum(close, 8) / 8) - stddev(close, 8))) ? 1 : (((1 < (volume / adv20)) || ((volume / adv20) == 1)) ? 1 : (-1 * 1)))",
    "alpha022": "-1 * (delta(correlation(high, volume, 5), 5) * rank(stddev(close, 20)))",
    "alpha023": "((sum(high, 20) / 20) < high) ? (-1 * delta(high, 2)) : 0",
    "alpha024": "(((delta((sum(close, 100) / 100), 100) / delay(close, 100)) < 0.05) || ((delta((sum(close, 100) / 100), 100) / delay(close, 100)) == 0.05)) ? (-1 * (close - ts_min(close, 100))) : (-1 * delta(close, 3))",
    "alpha025": "rank((((-1 * returns) * adv20) * vwap) * (high - close))",
    "alpha026": "-1 * ts_max(correlation(ts_rank(volume, 5), ts_rank(high, 5), 5), 3)",
    "alpha027": "(0.5 < rank((sum(correlation(rank(volume), rank(vwap), 6), 2) / 2.0))) ? (-1 * 1) : 1",
    "alpha028": "scale(((correlation(adv20, low, 5) + ((high + low) / 2)) - close))",
    "alpha029": "min(product(rank(rank(scale(log(sum(ts_min(rank(rank((-1 * rank(delta((close - 1), 5))))), 2), 1))))), 1), 5) + ts_rank(delay((-1 * returns), 6), 5)",
    "alpha030": "((1.0 - rank((sign((close - delay(close, 1))) + sign((delay(close, 1) - delay(close, 2))) + sign((delay(close, 2) - delay(close, 3)))))) * sum(volume, 5)) / sum(volume, 20)",
    "alpha031": "(rank(rank(rank(decay_linear((-1 * rank(rank(delta(close, 10)))), 10)))) + rank((-1 * delta(close, 3)))) + sign(scale(correlation(adv20, low, 12)))",
    "alpha032": "scale(((sum(close, 7) / 7) - close)) + (20 * scale(correlation(vwap, delay(close, 5), 230)))",
    "alpha033": "rank(-1 * ((1 - (open / close)) ^ 1))",
    "alpha034": "rank((1 - rank((stddev(returns, 2) / stddev(returns, 5)))) + (1 - rank(delta(close, 1))))",
    "alpha035": "(ts_rank(volume, 32) * (1 - ts_rank(((close + high) - low), 16))) * (1 - ts_rank(returns, 32))",
    "alpha036": "(((2.21 * rank(correlation((close - open), delay(volume, 1), 15))) + (0.7 * rank((open - close))) + (0.73 * rank(ts_rank(delay((-1 * returns), 6), 5))) + rank(abs(correlation(vwap, adv20, 6)))) + (0.6 * rank(((sum(close, 200) / 200) - open) * (close - open))))",
    "alpha037": "rank(correlation(delay((open - close), 1), close, 200)) + rank((open - close))",
    "alpha038": "-1 * rank(ts_rank(close, 10)) * rank((close / open))",
    "alpha039": "-1 * rank((delta(close, 7) * (1 - rank(decay_linear((volume / adv20), 9))))) * (1 + rank(sum(returns, 250)))",
    "alpha040": "-1 * rank(stddev(high, 10)) * correlation(high, volume, 10)",
    "alpha041": "(high * low) ^ 0.5 - vwap",
    "alpha042": "rank((vwap - close)) / rank((vwap + close))",
    "alpha043": "ts_rank((volume / adv20), 20) * ts_rank((-1 * delta(close, 7)), 8)",
    "alpha044": "-1 * correlation(high, rank(volume), 5)",
    "alpha045": "-1 * (rank((sum(delay(close, 5), 20) / 20)) * correlation(close, volume, 2) * rank(correlation(sum(close, 5), sum(close, 20), 2)))",
    "alpha046": "(0.25 < (((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10))) ? (-1 * 1) : ((((delay(close, 20) - delay(close, 10)) / 10 - ((delay(close, 10) - close) / 10)) < 0) ? 1 : ((-1 * 1) * (close - delay(close, 1))))",
    "alpha047": "(((rank((1 / close)) * volume) / adv20) * ((high * rank((high - close))) / (sum(high, 5) / 5))) - rank((vwap - delay(vwap, 5)))",
    "alpha048": "indneutralize(((correlation(delta(close, 1), delta(delay(close, 1), 1), 250) * delta(close, 1)) / close), IndClass.subindustry) / sum(((delta(close, 1) / delay(close, 1)) ^ 2), 250)",
    "alpha049": "((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < (-1 * 0.1)) ? 1 : ((-1 * 1) * (close - delay(close, 1)))",
    "alpha050": "-1 * ts_max(rank(correlation(rank(volume), rank(vwap), 5)), 5)",
    "alpha051": "((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < (-1 * 0.05)) ? 1 : ((-1 * 1) * (close - delay(close, 1)))",
    "alpha052": "(((-1 * ts_min(low, 5)) + delay(ts_min(low, 5), 5)) * rank(((sum(returns, 240) - sum(returns, 20)) / 220))) * ts_rank(volume, 5)",
    "alpha053": "-1 * delta((((close - low) - (high - close)) / (close - low)), 9)",
    "alpha054": "(-1 * ((low - close) * (open ^ 5))) / ((low - high) * (close ^ 5))",
    "alpha055": "-1 * correlation(rank(((close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low, 12)))), rank(volume), 6)",
    "alpha056": "0 - (1 * (rank((sum(returns, 10) / sum(sum(returns, 2), 3))) * rank((returns * cap))))",
    "alpha057": "0 - (1 * ((close - vwap) / decay_linear(rank(ts_argmax(close, 30)), 2)))",
    "alpha058": "-1 * ts_rank(decay_linear(correlation(indneutralize(vwap, IndClass.sector), volume, 3.92795), 7.89291), 5.50322)",
    "alpha059": "-1 * ts_rank(decay_linear(correlation(indneutralize(((vwap * 0.728317) + (vwap * (1 - 0.728317))), IndClass.industry), volume, 4.25197), 16.2289), 8.19648)",
    "alpha060": "0 - (1 * ((2 * scale(rank(((((close - low) - (high - close)) / (high - low)) * volume)))) - scale(rank(ts_argmax(close, 10)))))",
    "alpha061": "rank((vwap - ts_min(vwap, 16.1219))) < rank(correlation(vwap, adv180, 17.9282))",
    "alpha062": "(rank(correlation(vwap, sum(adv20, 22.4101), 9.91009)) < rank(((rank(open) + rank(open)) < (rank(((high + low) / 2)) + rank(high))))) * -1",
    "alpha063": "(rank(decay_linear(delta(indneutralize(close, IndClass.industry), 2.25164), 8.22237)) - rank(decay_linear(correlation(((vwap * 0.318108) + (open * (1 - 0.318108))), sum(adv180, 37.2467), 13.557), 12.2883))) * -1",
    "alpha064": "(rank(correlation(sum(((open * 0.178404) + (low * (1 - 0.178404))), 12.7054), sum(adv120, 12.7054), 16.6208)) < rank(delta((((high + low) / 2) * 0.178404) + (vwap * (1 - 0.178404)), 3.69741))) * -1",
    "alpha065": "(rank(correlation(((open * 0.00817205) + (vwap * (1 - 0.00817205))), sum(adv60, 8.6911), 6.40374)) < rank((open - ts_min(open, 13.635)))) * -1",
    "alpha066": "(rank(decay_linear(delta(vwap, 3.51013), 7.23052)) + ts_rank(decay_linear((((low * 0.96633) + (low * (1 - 0.96633))) - vwap) / (open - ((high + low) / 2)), 11.4157), 6.72611)) * -1",
    "alpha067": "(rank((high - ts_min(high, 2.14593))) ^ rank(correlation(indneutralize(vwap, IndClass.sector), indneutralize(adv20, IndClass.subindustry), 6.02936))) * -1",
    "alpha068": "(ts_rank(correlation(rank(high), rank(adv15), 8.91644), 13.9333) < rank(delta(((close * 0.518371) + (low * (1 - 0.518371))), 1.06157))) * -1",
    "alpha069": "(rank(ts_max(delta(indneutralize(vwap, IndClass.industry), 2.72412), 4.79344)) ^ ts_rank(correlation(((close * 0.490655) + (vwap * (1 - 0.490655))), adv20, 4.92416), 9.0615)) * -1",
    "alpha070": "(rank(delta(vwap, 1.29456)) ^ ts_rank(correlation(indneutralize(close, IndClass.industry), adv50, 17.8256), 17.9171)) * -1",
    "alpha071": "max(ts_rank(decay_linear(correlation(ts_rank(close, 3.43976), ts_rank(adv180, 12.0647), 18.0175), 4.20501), 15.6948), ts_rank(decay_linear((rank(((low + open) - (vwap + vwap))) ^ 2), 16.4662), 4.4388))",
    "alpha072": "rank(decay_linear(correlation(((high + low) / 2), adv40, 8.93345), 10.1519)) / rank(decay_linear(correlation(ts_rank(vwap, 3.72469), ts_rank(volume, 18.5188), 6.86671), 2.95011))",
    "alpha073": "max(rank(decay_linear(delta(vwap, 4.72775), 2.91864)), ts_rank(decay_linear((delta(((open * 0.147155) + (low * (1 - 0.147155))), 2.03608) / ((open * 0.147155) + (low * (1 - 0.147155)))) * -1, 3.33829), 16.7411)) * -1",
    "alpha074": "(rank(correlation(close, sum(adv30, 37.4843), 15.1365)) < rank(correlation(rank(((high * 0.0261661) + (vwap * (1 - 0.0261661)))), rank(volume), 11.4791))) * -1",
    "alpha075": "(rank(correlation(vwap, volume, 4.24304)) < rank(correlation(rank(low), rank(adv50), 12.4413)))",
    "alpha076": "max(rank(decay_linear(delta(vwap, 1.24383), 11.8259)), ts_rank(decay_linear(ts_rank(correlation(indneutralize(low, IndClass.sector), adv81, 8.14941), 19.569), 17.1543), 19.383)) * -1",
    "alpha077": "min(rank(decay_linear((((high + low) / 2 + high) - (vwap + high)), 20.0451)), rank(decay_linear(correlation(((high + low) / 2), adv40, 3.1614), 5.64125)))",
    "alpha078": "rank(correlation(sum(((low * 0.352233) + (vwap * (1 - 0.352233))), 19.7428), sum(adv40, 19.7428), 6.83313)) ^ rank(correlation(rank(vwap), rank(volume), 5.77492))",
    "alpha079": "(rank(delta(indneutralize(((close * 0.60733) + (open * (1 - 0.60733))), IndClass.sector), 1.23438)) < rank(correlation(ts_rank(vwap, 3.60973), ts_rank(adv150, 9.18637), 14.6644)))",
    "alpha080": "(rank(sign(delta(indneutralize(((open * 0.868128) + (high * (1 - 0.868128))), IndClass.industry), 4.04545))) ^ ts_rank(correlation(high, adv10, 5.11456), 5.53756)) * -1",
    "alpha081": "(rank(log(product(rank((rank(correlation(vwap, sum(adv10, 49.6054), 8.47743)) ^ 4)), 14.9655))) < rank(correlation(rank(vwap), rank(volume), 5.07914))) * -1",
    "alpha082": "min(rank(decay_linear(delta(open, 1.46063), 14.8717)), ts_rank(decay_linear(correlation(indneutralize(volume, IndClass.sector), ((open * 0.634196) + (open * (1 - 0.634196))), 17.4842), 6.92131), 13.4283)) * -1",
    "alpha083": "(rank(delay(((high - low) / (sum(close, 5) / 5)), 2)) * rank(rank(volume))) / (((high - low) / (sum(close, 5) / 5)) / (vwap - close))",
    "alpha084": "signedpower(ts_rank((vwap - ts_max(vwap, 15.3217)), 20.7127), delta(close, 4.96796))",
    "alpha085": "rank(correlation(((high * 0.876703) + (close * (1 - 0.876703))), adv30, 9.61331)) ^ rank(correlation(ts_rank(((high + low) / 2), 3.70596), ts_rank(volume, 10.1595), 7.11408))",
    "alpha086": "(ts_rank(correlation(close, sum(adv20, 14.7444), 6.00049), 20.4195) < rank(((open + close) - (vwap + open)))) * -1",
    "alpha087": "max(rank(decay_linear(delta(((close * 0.369701) + (vwap * (1 - 0.369701))), 1.91233), 2.65461)), ts_rank(decay_linear(abs(correlation(indneutralize(adv81, IndClass.industry), close, 13.4132)), 4.89768), 14.4535)) * -1",
    "alpha088": "min(rank(decay_linear(((rank(open) + rank(low)) - (rank(high) + rank(close))), 8.06882)), ts_rank(decay_linear(correlation(ts_rank(close, 8.44728), ts_rank(adv60, 20.6966), 8.01266), 6.65053), 2.61957))",
    "alpha089": "ts_rank(decay_linear(correlation(((low * 0.967285) + (low * (1 - 0.967285))), adv10, 6.94279), 5.51607), 3.79744) - ts_rank(decay_linear(delta(indneutralize(vwap, IndClass.industry), 3.48158), 10.1466), 15.3012)",
    "alpha090": "(rank((close - ts_max(close, 4.66719))) ^ ts_rank(correlation(indneutralize(adv40, IndClass.subindustry), low, 5.38375), 3.21856)) * -1",
    "alpha091": "(ts_rank(decay_linear(decay_linear(correlation(indneutralize(close, IndClass.industry), volume, 9.74928), 16.398), 3.83219), 4.8667) - rank(decay_linear(correlation(vwap, adv30, 4.01303), 2.6809))) * -1",
    "alpha092": "min(ts_rank(decay_linear(((((high + low) / 2) + close) < (low + open)), 14.7221), 18.8683), ts_rank(decay_linear(correlation(rank(low), rank(adv30), 7.58555), 6.94024), 6.80584))",
    "alpha093": "ts_rank(decay_linear(correlation(indneutralize(vwap, IndClass.industry), adv81, 17.4193), 19.848), 7.54455) / rank(decay_linear(delta(((close * 0.524434) + (vwap * (1 - 0.524434))), 2.77377), 16.2664))",
    "alpha094": "(rank((vwap - ts_min(vwap, 11.5783))) ^ ts_rank(correlation(ts_rank(vwap, 19.6462), ts_rank(adv60, 4.02992), 18.0926), 2.70756)) * -1",
    "alpha095": "rank((open - ts_min(open, 12.4105))) < ts_rank((rank(correlation(sum(((high + low) / 2), 19.1351), sum(adv40, 19.1351), 12.8742)) ^ 5), 11.7584)",
    "alpha096": "max(ts_rank(decay_linear(correlation(rank(vwap), rank(volume), 3.83878), 4.16783), 8.38151), ts_rank(decay_linear(ts_argmax(correlation(ts_rank(close, 7.45404), ts_rank(adv60, 4.13242), 3.65459), 12.6556), 14.0365), 13.4143)) * -1",
    "alpha097": "(rank(decay_linear(delta(indneutralize(((low * 0.721001) + (vwap * (1 - 0.721001))), IndClass.industry), 3.3705), 20.4523)) - ts_rank(decay_linear(ts_rank(correlation(ts_rank(low, 7.87871), ts_rank(adv60, 17.255), 4.97547), 18.5925), 15.7152), 6.71659)) * -1",
    "alpha098": "rank(decay_linear(correlation(vwap, sum(adv5, 26.4719), 4.58418), 7.18088)) - rank(decay_linear(ts_rank(ts_argmin(correlation(rank(open), rank(adv15), 20.8187), 8.62571), 6.95668), 8.07206))",
    "alpha099": "(rank(correlation(sum(((high + low) / 2), 19.8975), sum(adv60, 19.8975), 8.8136)) < rank(correlation(low, volume, 6.28259))) * -1",
    "alpha100": "0 - (1 * ((1.5 * scale(indneutralize(indneutralize(rank(((((close - low) - (high - close)) / (high - low)) * volume)), IndClass.subindustry), IndClass.subindustry)) - scale(indneutralize((correlation(close, rank(adv20), 5) - rank(ts_argmin(close, 30))), IndClass.subindustry))) * (volume / adv20)))",
    "alpha101": "(close - open) / ((high - low) + 0.001)",
}


class QwenClient:
    """阿里云千问模型客户端"""

    def __init__(self, api_key: str, model: str, temperature: float, max_tokens: int):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def send_prompt(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        try:
            response = self.session.post(
                self.api_url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload,
                timeout=60,
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            raise RuntimeError(f"API调用失败：{str(e)}")


def create_qwen_client(cfg: Config) -> QwenClient:
    """Create the Qwen client (same logic as factor11.15.py)."""
    api_key = os.environ.get("QWEN_API_KEY") or "sk-0a6f8d4b2b6e42cc9c88e25410c8853b"
    if not api_key:
        raise ValueError("请设置环境变量QWEN_API_KEY")
    return QwenClient(api_key, cfg.model_name, cfg.temperature, cfg.max_tokens)


def generate_new_factors(base_factors: List[str], client: QwenClient, cfg: Config) -> List[str]:
    """根据基础因子生成变异因子和交叉因子。"""
    # 收集生成出来的新因子表达式
    new_factors: List[str] = []
    for idx, base in enumerate(base_factors, 1):
        # 针对每个基础因子构造提示词，并请求大模型生成变异因子
        prompt = _build_mutation_prompt(base, cfg.n_mutations)
        try:
            generated = client.send_prompt(prompt).split("\n")
            valid = []
            for f in generated:
                f_clean = f.strip()
                if len(f_clean) >= 10 and "rank(" in f_clean and not any(word in f_clean for word in ["以下是", "根据", "生成", "因子"]):
                    cleaned = _clean_factor_expression(f_clean)
                    if cleaned and cleaned.startswith("rank("):
                        valid.append(cleaned)
            new_factors.extend(valid[: cfg.n_mutations])
        except Exception as e:
            print(f"基础因子 {idx} 生成失败：{e}")

    for i in range(cfg.n_crossovers):
        # 随机抽取两个基础因子，构造交叉提示词
        f1, f2 = random.sample(base_factors, 2)
        prompt = _build_crossover_prompt(f1, f2)
        try:
            generated = client.send_prompt(prompt).strip()
            cleaned = _clean_factor_expression(generated)
            if cleaned and cleaned.startswith("rank(") and "zscore(" in cleaned:
                new_factors.append(cleaned)
        except Exception as e:
            print(f"交叉因子 {i+1} 生成失败：{e}")

    # 使用最长公共子串相似度去重，避免重复因子
    unique = _deduplicate_factors(new_factors, cfg.factor_similarity_threshold)
    return unique


def _build_mutation_prompt(base: str, n: int) -> str:
    examples = "\n".join([
        "输入：rank(zscore(momentum(close, 20)))\n输出：rank(zscore(ema(momentum(close, 20), 5)))",
        "输入：rank(zscore((close - open) / open))\n输出：rank(zscore((close - open) / open * 100))",
    ])
    return (
        f"基于基础因子生成{n}个变异因子，可使用close、volume、open、high、low字段，"
        f"函数包括momentum/rolling_mean/ema等，必须包含zscore和rank。\n\n"
        "重要：只输出因子表达式，不要包含任何说明文字。\n\n"
        f"示例：\n{examples}\n\n"
        f"基础因子：{base}\n\n"
        "输出因子（每行一个表达式）："
    )


def _build_crossover_prompt(f1: str, f2: str) -> str:
    examples = "\n".join([
        "因子1：rank(zscore(momentum(close, 20)))\n因子2：rank(zscore((high - low) / open))\n输出：rank(zscore(momentum(close,20)) + zscore((high - low)/open))",
    ])
    return (
        "融合两个因子，用+/-/*/÷运算，可使用close、volume、open、high、low字段，"
        "必须包含zscore和rank。\n\n"
        "重要：只输出因子表达式，不要包含任何说明文字。\n\n"
        f"示例：\n{examples}\n\n"
        f"因子1：{f1}\n因子2：{f2}\n\n"
        "输出因子（仅表达式）："
    )


def _deduplicate_factors(factors: List[str], threshold: float) -> List[str]:
    # 保存最终保留的因子表达式
    unique: List[str] = []
    for f in factors:
        norm = re.sub(r"\s+", "", f).lower().replace("×", "*").replace("÷", "/")
        duplicate = False
        for u in unique:
            u_norm = re.sub(r"\s+", "", u).lower().replace("×", "*").replace("÷", "/")
            lcs = _longest_common_substring_length(norm, u_norm)
            sim = lcs / max(len(norm), len(u_norm)) if max(len(norm), len(u_norm)) > 0 else 0
            if sim >= threshold:
                duplicate = True
                break
        if not duplicate:
            unique.append(f)
    return unique


def _longest_common_substring_length(s1: str, s2: str) -> int:
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    max_len = 0
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                max_len = max(max_len, dp[i][j])
    return max_len


def _run_factor(expr: str, data: Dict[str, pd.DataFrame], groups: Optional[pd.DataFrame] = None) -> Optional[pd.DataFrame]:
    # 取出基础宽表字段，后续所有运算都依赖这些原始数据
    open_w = data["open"]
    high_w = data["high"]
    low_w = data["low"]
    close_w = data["close"]
    volume_w = data["volume"]

    # 派生字段：收益率、VWAP、美元成交量（用于 adv 系列因子）
    returns_w = close_w / close_w.shift(1) - 1
    vwap_w = (open_w + high_w + low_w + close_w) / 4.0
    dollar_vol_w = close_w * volume_w

    # 根据表达式中出现的 advN 提前计算滚动平均成交额，避免重复计算
    adv_windows = {int(n) for n in re.findall(r"adv(\d+)", expr)}
    adv_dict: Dict[str, pd.DataFrame] = {}
    for n in adv_windows:
        adv_dict[f"adv{n}"] = dollar_vol_w.rolling(window=n, min_periods=max(1, int(n * 0.5))).mean()

    def momentum(x: pd.DataFrame, n: int) -> pd.DataFrame:
        return x / x.shift(n) - 1

    def rolling_mean(x: pd.DataFrame, n: int) -> pd.DataFrame:
        return x.rolling(window=n, min_periods=max(1, int(n * 0.5))).mean()

    def rolling_std(x: pd.DataFrame, n: int) -> pd.DataFrame:
        return x.rolling(window=n, min_periods=max(1, int(n * 0.5))).std()

    def ema(x: pd.DataFrame, n: int) -> pd.DataFrame:
        return x.ewm(span=n, adjust=False).mean()

    def delta(x: pd.DataFrame, n: float) -> pd.DataFrame:
        n_int = int(math.floor(n)) if not isinstance(n, int) else n
        return x - x.shift(n_int)

    def delay(x: pd.DataFrame, n: float) -> pd.DataFrame:
        n_int = int(math.floor(n)) if not isinstance(n, int) else n
        return x.shift(n_int)

    def ts_rank(x: pd.DataFrame, n: float) -> pd.DataFrame:
        n_int = int(math.floor(n)) if not isinstance(n, int) else n
        return x.rolling(window=n_int, min_periods=max(1, int(n_int * 0.5))).rank(ascending=False)

    def ts_sum(x: pd.DataFrame, n: float) -> pd.DataFrame:
        n_int = int(math.floor(n)) if not isinstance(n, int) else n
        return x.rolling(window=n_int, min_periods=max(1, int(n_int * 0.5))).sum()

    def ts_min(x: pd.DataFrame, n: float) -> pd.DataFrame:
        n_int = int(math.floor(n)) if not isinstance(n, int) else n
        return x.rolling(window=n_int, min_periods=max(1, int(n_int * 0.5))).min()

    def ts_max(x: pd.DataFrame, n: float) -> pd.DataFrame:
        n_int = int(math.floor(n)) if not isinstance(n, int) else n
        return x.rolling(window=n_int, min_periods=max(1, int(n_int * 0.5))).max()

    def ts_argmax(x: pd.DataFrame, n: float) -> pd.DataFrame:
        n_int = int(math.floor(n)) if not isinstance(n, int) else n
        return x.rolling(window=n_int, min_periods=max(1, int(n_int * 0.5))).apply(lambda arr: np.argmax(arr) + 1, raw=True)

    def ts_argmin(x: pd.DataFrame, n: float) -> pd.DataFrame:
        n_int = int(math.floor(n)) if not isinstance(n, int) else n
        return x.rolling(window=n_int, min_periods=max(1, int(n_int * 0.5))).apply(lambda arr: np.argmin(arr) + 1, raw=True)

    def stddev(x: pd.DataFrame, n: float) -> pd.DataFrame:
        return rolling_std(x, n)

    def signedpower(x: pd.DataFrame, a: float) -> pd.DataFrame:
        return np.sign(x) * (np.abs(x) ** a)

    def decay_linear(x: pd.DataFrame, n: float) -> pd.DataFrame:
        n_int = int(math.floor(n)) if not isinstance(n, int) else n

        def _decay(arr: pd.Series) -> float:
            length = len(arr)
            w = np.arange(1, length + 1, dtype=float)
            return float(np.dot(arr, w) / w.sum())

        return x.rolling(window=n_int, min_periods=max(1, int(n_int * 0.5))).apply(_decay, raw=False)

    def correlation(x: pd.DataFrame, y: pd.DataFrame, n: float) -> pd.DataFrame:
        n_int = int(math.floor(n)) if not isinstance(n, int) else n
        return x.rolling(window=n_int, min_periods=max(1, int(n_int * 0.5))).corr(y)

    def covariance(x: pd.DataFrame, y: pd.DataFrame, n: float) -> pd.DataFrame:
        n_int = int(math.floor(n)) if not isinstance(n, int) else n
        return x.rolling(window=n_int, min_periods=max(1, int(n_int * 0.5))).cov(y)

    def product(x: pd.DataFrame, n: float) -> pd.DataFrame:
        n_int = int(math.floor(n)) if not isinstance(n, int) else n
        return x.rolling(window=n_int, min_periods=max(1, int(n_int * 0.5))).apply(np.prod, raw=True)

    def rank(x: pd.DataFrame) -> pd.DataFrame:
        return x.rank(axis=1, ascending=False, method="min")

    def zscore(x: pd.DataFrame) -> pd.DataFrame:
        mean = x.mean(axis=1, skipna=True).values.reshape(-1, 1)
        std = x.std(axis=1, skipna=True).replace(0, 1).values.reshape(-1, 1)
        return (x - mean) / std

    def scale(x: pd.DataFrame, a: float = 1.0) -> pd.DataFrame:
        denom = x.abs().sum(axis=1, skipna=True).replace(0, np.nan).values.reshape(-1, 1)
        return (x / denom) * a

    def indneutralize(x: pd.DataFrame, g: Optional[str] = None) -> pd.DataFrame:
        if groups is not None and g is not None and g in groups:
            out = x.copy()
            for date in x.index:
                date_groups = groups.loc[date]
                date_values = x.loc[date]
                means = date_values.groupby(date_groups).transform("mean")
                out.loc[date] = date_values - means
            return out
        return x.subtract(x.mean(axis=1, skipna=True), axis=0)

    def abs_(x: pd.DataFrame) -> pd.DataFrame:
        return x.abs()

    def sign(x: pd.DataFrame) -> pd.DataFrame:
        return np.sign(x)

    def log(x: pd.DataFrame) -> pd.DataFrame:
        return np.log(x)

    # 将数据字段与数学函数注入到局部上下文，供表达式动态求值
    local_ctx: Dict[str, object] = {
        "open": open_w,
        "high": high_w,
        "low": low_w,
        "close": close_w,
        "volume": volume_w,
        "returns": returns_w,
        "vwap": vwap_w,
    }
    local_ctx.update(adv_dict)
    local_ctx.update({
        "momentum": momentum,
        "rolling_mean": rolling_mean,
        "rolling_std": rolling_std,
        "ema": ema,
        "delta": delta,
        "delay": delay,
        "ts_rank": ts_rank,
        "ts_sum": ts_sum,
        "sum": ts_sum,
        "ts_min": ts_min,
        "min": ts_min,
        "ts_max": ts_max,
        "max": ts_max,
        "ts_argmax": ts_argmax,
        "ts_argmin": ts_argmin,
        "stddev": stddev,
        "signedpower": signedpower,
        "decay_linear": decay_linear,
        "correlation": correlation,
        "covariance": covariance,
        "product": product,
        "rank": rank,
        "zscore": zscore,
        "scale": scale,
        "indneutralize": indneutralize,
        "abs": abs_,
        "sign": sign,
        "log": log,
    })

    def convert_expression(expression: str) -> str:
        expr_py = expression.replace("^", "**").replace("||", " or ").replace("&&", " and ")
        ternary_pattern = re.compile(r"([^?]+)\?([^:]+):([^?]+)")
        for _ in range(10):
            match = ternary_pattern.search(expr_py)
            if not match:
                break
            cond, true_expr, false_expr = match.groups()
            replacement = f"({true_expr.strip()} if {cond.strip()} else {false_expr.strip()})"
            expr_py = expr_py[: match.start()] + replacement + expr_py[match.end():]
        return expr_py

    expr_clean = convert_expression(expr)
    try:
        code = compile(expr_clean, "<expr>", "eval")
        result = eval(code, {"__builtins__": None}, local_ctx)
        if not isinstance(result, pd.DataFrame):
            return None
        return result.dropna(how="all")
    except Exception:
        return None


def calculate_ic(
    factor_expr: str,
    data: Dict[str, pd.DataFrame],
    cfg: Config,
    groups: Optional[pd.DataFrame] = None,
) -> float:
    # 先根据表达式计算因子值宽表；若计算失败直接返回 0
    factor_wide = _run_factor(factor_expr, data, groups=groups)
    if factor_wide is None or factor_wide.empty:
        return 0.0

    # 计算未来 1 日收益率，作为因子预测目标
    close_w = data["close"]
    fwd_ret = close_w.shift(-1) / close_w - 1

    # 对齐日期和股票代码，保证因子值与收益率在同一维度
    common_dates = factor_wide.index.intersection(fwd_ret.index)
    common_cols = factor_wide.columns.intersection(fwd_ret.columns)
    factor_aligned = factor_wide.loc[common_dates, common_cols]
    fwd_ret_aligned = fwd_ret.loc[common_dates, common_cols]

    # 单股票：时间序列 IC，直接对一支股票的时间序列做秩相关
    if len(common_cols) <= 1:
        series_factor = factor_aligned.iloc[:, 0].dropna()
        series_ret = fwd_ret_aligned.iloc[:, 0].dropna()
        aligned = pd.concat([series_factor, series_ret], axis=1, join="inner").dropna()
        if len(aligned) < cfg.min_samples_for_ic:
            return 0.0
        ic = aligned.rank().corr(method="pearson").iloc[0, 1]
        return float(ic) if not np.isnan(ic) else 0.0

    # 多股票：横截面 IC，逐日计算截面秩相关并求均值
    fac_long = factor_aligned.stack(future_stack=True)
    ret_long = fwd_ret_aligned.stack(future_stack=True)
    df = pd.concat([fac_long, ret_long], axis=1, keys=["factor", "ret"]).dropna()
    if df.empty:
        return 0.0
    ic_list: List[float] = []
    for _, group in df.groupby(level=0):
        if len(group) < cfg.min_samples_for_ic:
            continue
        fac = group["factor"].rank(method="average")
        ret = group["ret"].rank(method="average")
        if fac.nunique() <= 1 or ret.nunique() <= 1:
            continue
        corr = fac.corr(ret)
        if not np.isnan(corr):
            ic_list.append(corr)
    return float(np.mean(ic_list)) if ic_list else 0.0


def _clean_factor_expression(expr: str) -> str:
    if not expr:
        return ""
    # 去掉 Markdown 粗体/代码块等格式符号
    expr = re.sub(r"\*\*([^*]+)\*\*", r"\1", expr)
    expr = re.sub(r"`([^`]+)`", r"\1", expr)
    # 去除大模型常见的提示性文字，只保留表达式
    expr = re.sub(r"(以下是|根据|生成|因子)[^a-zA-Z]*", "", expr, flags=re.IGNORECASE)
    # 转换全角标点为半角，避免解析失败
    full_to_half = {
        "，": ",",
        "。": ".",
        "（": "(",
        "）": ")",
        "：": ":",
        "；": ";",
        "！": "!",
        "？": "?",
        "【": "[",
        "】": "]",
    }
    for full, half in full_to_half.items():
        expr = expr.replace(full, half)
    # 截取从首个函数开始到括号匹配结束的部分，尽量剔除多余文本
    if "(" in expr:
        start_pos = -1
        for func_name in ["rank(", "zscore(", "momentum(", "rolling_mean(", "ema(", "delta(", "ts_rank(", "clip("]:
            pos = expr.find(func_name)
            if pos >= 0:
                start_pos = pos
                break
        if start_pos >= 0:
            expr = expr[start_pos:]
            open_count = 0
            end_pos = -1
            for i, char in enumerate(expr):
                if char == "(":
                    open_count += 1
                elif char == ")":
                    open_count -= 1
                    if open_count == 0:
                        end_pos = i + 1
                        break
            if end_pos > 0:
                expr = expr[:end_pos]
            else:
                expr = expr + ")" * open_count
    expr = re.sub(r"\)\s*([a-zA-Z_]+\(", r") * \1", expr)
    open_count = expr.count("(")
    close_count = expr.count(")")
    if open_count > close_count:
        expr += ")" * (open_count - close_count)
    elif close_count > open_count:
        for _ in range(close_count - open_count):
            last_close = expr.rfind(")")
            if last_close >= 0:
                expr = expr[:last_close] + expr[last_close + 1 :]
    return re.sub(r"\s+", " ", expr).strip()


def main() -> None:
    # 1. 准备配置：填入待读取的股票数据文件路径
    cfg = Config()
    cfg.excel_paths = [
        "D:/因子挖掘/股票数据/华泰证券数据.xlsx",
        "D:/因子挖掘/股票数据/中国平安数据.xlsx",
        "D:/因子挖掘/股票数据/中信证券数据.xlsx",
        "D:/因子挖掘/股票数据/招商银行数据.xlsx",
    ]
    print("启动因子挖掘流程...")
    # 2. 读取并清洗数据，得到宽表格式的行情数据
    data = read_stock_data_from_excel(cfg)
    print(f"成功读取数据，包含字段：{list(data.keys())}")
    # 3. 初始化千问客户端，用于生成变异/交叉因子
    client = create_qwen_client(cfg)
    print("千问客户端已初始化")
    # 4. 基于基础因子生成新的因子表达式
    new_factors = generate_new_factors(BASE_FACTORS, client, cfg)
    all_factors = list(ALPHA101_FACTORS.values()) + new_factors
    # 5. 逐个计算因子 IC，并记录有效性
    results = []
    for idx, factor in enumerate(all_factors, 1):
        ic = calculate_ic(factor, data, cfg)
        results.append(
            {
                "序号": idx,
                "因子表达式": factor,
                "IC值": round(ic, 6),
                "有效": "是" if abs(ic) >= cfg.ic_threshold else "否",
            }
        )
        if idx % 5 == 0:
            print(f"已评估 {idx}/{len(all_factors)} 个因子")
    result_df = pd.DataFrame(results).sort_values("IC值", ascending=False)
    print("评估完成，IC最高的前 10 个因子：")
    print(result_df[["序号", "IC值", "有效", "因子表达式"]].head(10))
    # 6. 保存结果到 Excel，文件名为“因子评估结果_中文版.xlsx”
    save_dir = os.path.dirname(cfg.get_excel_paths()[0]) if cfg.get_excel_paths() else os.getcwd()
    save_path = os.path.join(save_dir, "因子评估结果_中文版.xlsx")
    result_df.to_excel(save_path, index=False, engine="openpyxl")
    print(f"结果已保存至：{save_path}")


if __name__ == "__main__":
    main()

# 多股票因子挖掘的基本逻辑：
# 1. 横截面因子：在每个时间点上，对所有股票进行排序、分组、计算相对指标等。例如，计算每只股票在当天所有股票中的市值排名、收益率排名等。
# 2. 时间序列因子：对每只股票单独计算其时间序列指标，例如过去20日的收益率、波动率等。
# 之后将这些因子与下一期的收益率进行关联，用于评估预测能力。
# 具体步骤：数据准备 -> 因子计算 -> 因子与收益率的关联分析 -> 因子组合。
