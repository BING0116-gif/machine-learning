"""文件IO工具模块

提供常用的文件读写操作工具函数。
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import joblib
import pandas as pd


def ensure_dir(path: Union[str, Path]) -> Path:
    """确保目录存在，不存在则创建

    Args:
        path: 目录路径

    Returns:
        Path对象
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: Dict, path: Union[str, Path], indent: int = 2) -> None:
    """保存数据为JSON文件

    Args:
        data: 要保存的字典数据
        path: 保存路径
        indent: 缩进空格数
    """
    path = Path(path)
    ensure_dir(path.parent)

    def default_serializer(obj):
        if isinstance(obj, (pd.Timestamp,)):
            return obj.isoformat()
        if hasattr(obj, "tolist"):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, default=default_serializer, ensure_ascii=False)


def load_json(path: Union[str, Path]) -> Dict:
    """加载JSON文件

    Args:
        path: 文件路径

    Returns:
        加载的字典数据
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_dataframe(
    df: pd.DataFrame,
    path: Union[str, Path],
    format: str = "csv",
    **kwargs,
) -> None:
    """保存DataFrame到文件

    Args:
        df: 要保存的DataFrame
        path: 保存路径
        format: 保存格式 ('csv', 'pkl', 'parquet')
        **kwargs: 传递给对应保存函数的额外参数
    """
    path = Path(path)
    ensure_dir(path.parent)

    if format == "csv":
        df.to_csv(path, index=True, **kwargs)
    elif format == "pkl":
        joblib.dump(df, path)
    elif format == "parquet":
        df.to_parquet(path, **kwargs)
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'csv', 'pkl', or 'parquet'.")


def load_dataframe(path: Union[str, Path], format: str = "auto") -> pd.DataFrame:
    """加载DataFrame从文件

    Args:
        path: 文件路径
        format: 文件格式 ('auto', 'csv', 'pkl', 'parquet')

    Returns:
        加载的DataFrame
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    if format == "auto":
        suffix = path.suffix.lower()
        if suffix == ".csv":
            format = "csv"
        elif suffix in (".pkl", ".pickle"):
            format = "pkl"
        elif suffix == ".parquet":
            format = "parquet"
        else:
            format = "csv"

    if format == "csv":
        return pd.read_csv(path)
    elif format == "pkl":
        return joblib.load(path)
    elif format == "parquet":
        return pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported format: {format}")


def load_config(path: Union[str, Path]) -> Dict:
    """加载YAML配置文件

    Args:
        path: 配置文件路径

    Returns:
        配置字典
    """
    import yaml

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
