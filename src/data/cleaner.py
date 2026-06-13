"""数据清洗模块

清洗脏数据，处理缺失值、异常值、标签错误。
"""

from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from scipy import stats

from src.utils.logger import get_logger

logger = get_logger("dry_bean.cleaner")


class DataCleaner:
    """数据清洗器

    提供多种数据清洗策略，支持配置化。
    """

    LABEL_CORRECTIONS = {
        'D3RMAS0N': 'DERMASON',
        'S3K3R': 'SEKER',
        'B0MBAY': 'BOMBAY',
        'H0R0Z': 'HOROZ',
    }

    VALID_CLASSES = {'SEKER', 'BARBUNYA', 'BOMBAY', 'CALI', 'HOROZ', 'SIRA', 'DERMASON'}

    def __init__(self, strategy_config: Dict = None):
        """初始化数据清洗器

        Args:
            strategy_config: 策略配置字典
        """
        default_config = {
            'missing_value': 'median',
            'outlier': 'iqr',
            'label_correction': True,
            'iqr_factor': 1.5,
            'zscore_threshold': 3,
            'knn_n_neighbors': 5,
        }
        self.config = strategy_config or default_config

    def clean(self, df: pd.DataFrame,
              feature_columns: List[str],
              target_column: str) -> Tuple[pd.DataFrame, Dict]:
        """执行完整的清洗流程

        Args:
            df: 原始DataFrame
            feature_columns: 特征列名列表
            target_column: 目标列名

        Returns:
            (cleaned_df, cleaning_report) 元组
        """
        report = {
            'original_shape': list(df.shape),
            'steps_applied': [],
            'changes_made': {}
        }

        df_cleaned = df.copy()

        # Step 1: 标签纠错
        if self.config.get('label_correction', True):
            df_cleaned, label_report = self._correct_labels(df_cleaned, target_column)
            report['steps_applied'].append('label_correction')
            report['changes_made']['label_corrections'] = label_report

        # Step 2: 缺失值处理
        df_cleaned, missing_report = self._handle_missing_values(df_cleaned, feature_columns)
        report['steps_applied'].append('missing_value_handling')
        report['changes_made']['missing_values'] = missing_report

        # Step 3: 异常值处理
        df_cleaned, outlier_report = self._handle_outliers(df_cleaned, feature_columns)
        report['steps_applied'].append('outlier_handling')
        report['changes_made']['outliers'] = outlier_report

        report['final_shape'] = list(df_cleaned.shape)

        logger.info(f"数据清洗完成: {df.shape} -> {df_cleaned.shape}")
        return df_cleaned, report

    def _correct_labels(self, df: pd.DataFrame,
                        target_column: str) -> Tuple[pd.DataFrame, Dict]:
        """纠正标签错误

        处理三类标签问题：
        1. 已知的OCR/拼写错误（如 D3RMAS0N -> DERMASON）
        2. 大小写错误（如 barbunya -> BARBUNYA）
        3. 前后空格（如 'CALI ' -> 'CALI'）
        """
        corrections = {}

        # Step 1: 去除前后空格
        df[target_column] = df[target_column].str.strip()
        stripped_count = 0
        for val in df[target_column].unique():
            if val != val.strip():
                count = int((df[target_column] == val).sum())
                stripped_count += count
        if stripped_count > 0:
            corrections['strip_whitespace'] = {'count': stripped_count}

        # Step 2: 转大写
        original_values = df[target_column].copy()
        df[target_column] = df[target_column].str.upper()
        case_corrected = int((original_values != df[target_column]).sum())
        if case_corrected > 0:
            corrections['case_correction'] = {'count': case_corrected}
            logger.info(f"标签大小写纠错: {case_corrected} 条")

        # Step 3: 已知的OCR/拼写错误
        for wrong_label, correct_label in self.LABEL_CORRECTIONS.items():
            mask = df[target_column] == wrong_label
            count = int(mask.sum())
            if count > 0:
                df.loc[mask, target_column] = correct_label
                corrections[wrong_label] = {
                    'corrected_to': correct_label,
                    'count': count
                }
                logger.info(f"标签纠错: '{wrong_label}' -> '{correct_label}'，共 {count} 条")

        # Step 4: 验证所有标签都是有效类别
        invalid_mask = ~df[target_column].isin(self.VALID_CLASSES)
        invalid_count = int(invalid_mask.sum())
        if invalid_count > 0:
            invalid_labels = df.loc[invalid_mask, target_column].unique().tolist()
            logger.warning(f"发现 {invalid_count} 条无效标签: {invalid_labels}")
            corrections['invalid_labels'] = {
                'count': invalid_count,
                'labels': invalid_labels
            }

        return df, corrections

    def _handle_missing_values(self, df: pd.DataFrame,
                               columns: List[str]) -> Tuple[pd.DataFrame, Dict]:
        """处理缺失值"""
        strategy = self.config.get('missing_value', 'median')
        report = {'strategy': strategy, 'columns_affected': {}}

        # 检查是否有缺失值
        total_missing = df[columns].isnull().sum().sum()
        if total_missing == 0:
            report['columns_affected'] = {}
            return df, report

        if strategy == 'knn':
            imputer = KNNImputer(n_neighbors=self.config.get('knn_n_neighbors', 5))
            df[columns] = imputer.fit_transform(df[columns])
            report['columns_affected'] = {col: int(df[col].isnull().sum()) for col in columns if df[col].isnull().sum() > 0}
            # KNN处理后所有缺失值都被填充
            report['columns_affected'] = {col: 'filled_by_knn' for col in columns}
        else:
            for col in columns:
                missing_count = df[col].isnull().sum()
                if missing_count == 0:
                    continue

                if strategy == 'median':
                    fill_value = df[col].median()
                    df[col] = df[col].fillna(fill_value)
                elif strategy == 'mean':
                    fill_value = df[col].mean()
                    df[col] = df[col].fillna(fill_value)
                elif strategy == 'drop':
                    df = df.dropna(subset=[col])
                else:
                    fill_value = df[col].median()
                    df[col] = df[col].fillna(fill_value)

                report['columns_affected'][col] = {
                    'missing_count': int(missing_count),
                    'strategy': strategy,
                    'fill_value': float(fill_value) if strategy in ('median', 'mean') else None,
                }
                logger.info(f"缺失值处理: {col} 列，策略={strategy}，数量={missing_count}")

        return df, report

    def _handle_outliers(self, df: pd.DataFrame,
                         columns: List[str]) -> Tuple[pd.DataFrame, Dict]:
        """处理异常值"""
        method = self.config.get('outlier', 'iqr')
        report = {'method': method, 'columns_affected': {}}

        if method == 'none':
            return df, report

        for col in columns:
            if df[col].isnull().all():
                continue

            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                factor = self.config.get('iqr_factor', 1.5)
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR

                outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                outlier_count = int(outlier_mask.sum())

                # Winsorization: 截断到边界值
                df[col] = df[col].clip(lower_bound, upper_bound)

                if outlier_count > 0:
                    report['columns_affected'][col] = {
                        'method': 'winsorize_iqr',
                        'outliers_clipped': outlier_count,
                        'bounds': (float(lower_bound), float(upper_bound)),
                    }

            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(df[col].dropna()))
                threshold = self.config.get('zscore_threshold', 3)
                outlier_mask = z_scores > threshold
                outlier_count = int(outlier_mask.sum())

                mean_val = df[col].mean()
                std_val = df[col].std()
                lower_bound = mean_val - threshold * std_val
                upper_bound = mean_val + threshold * std_val

                df[col] = df[col].clip(lower_bound, upper_bound)

                if outlier_count > 0:
                    report['columns_affected'][col] = {
                        'method': 'winsorize_zscore',
                        'outliers_clipped': outlier_count,
                        'bounds': (float(lower_bound), float(upper_bound)),
                    }

        total_clipped = sum(v.get('outliers_clipped', 0) for v in report['columns_affected'].values())
        if total_clipped > 0:
            logger.info(f"异常值处理: 共截断 {total_clipped} 个异常值")

        return df, report
