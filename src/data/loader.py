"""数据加载模块

提供统一的数据加载接口，支持Dry Bean Dataset的三个子集。
"""

from typing import Dict, Optional
import pandas as pd
from pathlib import Path
import chardet

from src.utils.logger import get_logger

logger = get_logger("dry_bean.data")


class DataLoader:
    """数据加载器

    负责从CSV文件加载数据，自动检测编码，
    并返回标准化的DataFrame。

    Attributes:
        data_dir: 数据目录路径
        encoding: 文件编码
    """

    FEATURE_COLUMNS = [
        'Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength',
        'AspectRatio', 'Eccentricity', 'ConvexArea', 'EquivDiameter',
        'Extent', 'Solidity', 'roundness', 'Compactness',
        'ShapeFactor1', 'ShapeFactor2', 'ShapeFactor3', 'ShapeFactor4'
    ]

    TARGET_COLUMN = 'Class'

    VALID_CLASSES = ['SEKER', 'BARBUNYA', 'BOMBAY', 'CALI',
                     'HOROZ', 'SIRA', 'DERMASON']

    def __init__(self, data_dir: str = 'DryBeanDataset', encoding: str = 'auto'):
        """初始化数据加载器

        Args:
            data_dir: 数据目录路径
            encoding: 文件编码，'auto'表示自动检测
        """
        self.data_dir = Path(data_dir)
        self.encoding = encoding

    def _detect_encoding(self, file_path: Path) -> str:
        """检测文件编码"""
        if self.encoding != 'auto':
            return self.encoding

        with open(file_path, 'rb') as f:
            result = chardet.detect(f.read(10000))
        return result.get('encoding', 'utf-8') or 'utf-8'

    def load_csv(self, filename: str) -> pd.DataFrame:
        """加载单个CSV文件

        Args:
            filename: CSV文件名

        Returns:
            加载的DataFrame

        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 数据格式错误
        """
        file_path = self.data_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {file_path}")

        encoding = self._detect_encoding(file_path)
        logger.info(f"加载数据: {file_path} (编码: {encoding})")

        df = pd.read_csv(file_path, encoding=encoding, na_values=['?', 'NA', 'N/A', 'nan', 'NaN', ''])

        # 修正原始数据集中的拼写错误列名
        column_rename_map = {'AspectRation': 'AspectRatio'}
        existing_wrong_cols = [c for c in column_rename_map if c in df.columns]
        if existing_wrong_cols:
            df = df.rename(columns={c: column_rename_map[c] for c in existing_wrong_cols})
            logger.info(f"修正列名拼写: {existing_wrong_cols} → {[column_rename_map[c] for c in existing_wrong_cols]}")

        # 确保特征列为数值类型
        for col in self.FEATURE_COLUMNS:
            if not pd.api.types.is_numeric_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # 验证必要的列是否存在
        required_cols = self.FEATURE_COLUMNS + [self.TARGET_COLUMN]
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"缺少必要的列: {missing_cols}")

        logger.info(f"加载完成: {df.shape[0]}行, {df.shape[1]}列")
        return df

    def load_train(self) -> pd.DataFrame:
        """加载训练集"""
        return self.load_csv('Dry_Bean_Dataset_Dirty_train.csv')

    def load_val(self) -> pd.DataFrame:
        """加载验证集"""
        return self.load_csv('Dry_Bean_Dataset_Dirty_val.csv')

    def load_test(self) -> pd.DataFrame:
        """加载测试集"""
        return self.load_csv('Dry_Bean_Dataset_Dirty_test.csv')

    def load_all(self) -> Dict[str, pd.DataFrame]:
        """加载所有数据集

        Returns:
            包含train/val/test的字典
        """
        return {
            'train': self.load_train(),
            'val': self.load_val(),
            'test': self.load_test()
        }

    def get_data_info(self, df: pd.DataFrame) -> Dict:
        """获取数据集基本信息

        Args:
            df: 输入DataFrame

        Returns:
            包含数据统计信息的字典
        """
        info = {
            'shape': list(df.shape),
            'columns': list(df.columns),
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'missing_values': {
                col: int(count)
                for col, count in df.isnull().sum().items()
                if count > 0
            },
            'missing_percentage': {
                col: round(float(pct), 4)
                for col, pct in (df.isnull().sum() / len(df) * 100).items()
                if pct > 0
            },
            'class_distribution': {
                cls: int(count)
                for cls, count in df[self.TARGET_COLUMN].value_counts().items()
            },
            'numeric_stats': df[self.FEATURE_COLUMNS].describe().to_dict(),
            'memory_usage_mb': round(
                df.memory_usage(deep=True).sum() / 1024 / 1024, 2
            ),
        }
        return info
