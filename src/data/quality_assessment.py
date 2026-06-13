"""数据质量评估模块

对输入数据进行全面的质量评估，识别数据污染情况。
"""

from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from scipy import stats

from src.utils.logger import get_logger

logger = get_logger("dry_bean.quality")


class QualityAssessor:
    """数据质量评估器

    对数据进行全面的质量评估，包括缺失值、异常值、标签一致性等。
    """

    # 已知的标签错误映射
    KNOWN_LABEL_ERRORS = {
        'D3RMAS0N': 'DERMASON',
    }

    def __init__(self):
        """初始化质量评估器"""
        pass

    def assess(self, df: pd.DataFrame, feature_columns: List[str],
               target_column: str) -> Dict:
        """执行完整的数据质量评估

        Args:
            df: 输入DataFrame
            feature_columns: 特征列名列表
            target_column: 目标列名

        Returns:
            质量评估报告字典
        """
        report = {
            'dataset_shape': list(df.shape),
            'issues_found': [],
            'missing_values': self._analyze_missing_values(df, feature_columns),
            'outliers': self._detect_outliers(df, feature_columns),
            'label_issues': self._check_labels(df, target_column),
            'data_type_issues': self._validate_data_types(df, feature_columns),
            'class_distribution': df[target_column].value_counts().to_dict(),
        }

        # 汇总问题
        if report['missing_values']['total_missing'] > 0:
            report['issues_found'].append(
                f"发现 {report['missing_values']['total_missing']} 个缺失值"
            )

        if report['label_issues']['error_count'] > 0:
            report['issues_found'].append(
                f"发现 {report['label_issues']['error_count']} 个标签错误"
            )

        outlier_total = sum(
            v['count'] for v in report['outliers']['columns'].values()
        )
        if outlier_total > 0:
            report['issues_found'].append(
                f"发现 {outlier_total} 个潜在异常值"
            )

        logger.info(
            f"数据质量评估完成，发现 {len(report['issues_found'])} 类问题"
        )
        return report

    def _analyze_missing_values(self, df: pd.DataFrame,
                                 columns: List[str]) -> Dict:
        """分析缺失值"""
        missing = df[columns].isnull().sum()
        missing_cols = {
            col: int(count) for col, count in missing.items() if count > 0
        }

        result = {
            'total_missing': int(missing.sum()),
            'columns_with_missing': missing_cols,
            'missing_percentage': {
                col: round(float(count / len(df) * 100), 4)
                for col, count in missing_cols.items()
            },
        }

        if missing_cols:
            for col, count in missing_cols.items():
                logger.warning(
                    f"缺失值: {col} 列有 {count} 个缺失值 "
                    f"({count/len(df)*100:.2f}%)"
                )

        return result

    def _detect_outliers(self, df: pd.DataFrame,
                          columns: List[str]) -> Dict:
        """使用IQR方法检测异常值"""
        result = {
            'method': 'iqr',
            'iqr_factor': 1.5,
            'columns': {},
        }

        for col in columns:
            if df[col].isnull().all():
                continue
            # 跳过非数值列
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue

            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            outlier_count = int(outlier_mask.sum())

            if outlier_count > 0:
                result['columns'][col] = {
                    'count': outlier_count,
                    'percentage': round(
                        float(outlier_count / len(df) * 100), 2
                    ),
                    'lower_bound': float(lower_bound),
                    'upper_bound': float(upper_bound),
                    'min_value': float(df[col].min()),
                    'max_value': float(df[col].max()),
                }

        return result

    def _check_labels(self, df: pd.DataFrame,
                       target_column: str) -> Dict:
        """检查标签一致性"""
        result = {
            'unique_labels': list(df[target_column].unique()),
            'label_counts': df[target_column].value_counts().to_dict(),
            'errors': {},
            'error_count': 0,
        }

        # 检查已知错误
        for wrong_label, correct_label in self.KNOWN_LABEL_ERRORS.items():
            if wrong_label in df[target_column].values:
                count = int((df[target_column] == wrong_label).sum())
                result['errors'][wrong_label] = {
                    'correct_label': correct_label,
                    'count': count,
                }
                result['error_count'] += count
                logger.warning(
                    f"标签错误: '{wrong_label}' 应为 '{correct_label}'，"
                    f"共 {count} 条"
                )

        # 检查未知标签（不在VALID_CLASSES中的）
        from src.data.loader import DataLoader
        unknown_labels = (
            set(df[target_column].unique()) - set(DataLoader.VALID_CLASSES)
        )
        if unknown_labels:
            for label in unknown_labels:
                if label not in self.KNOWN_LABEL_ERRORS:
                    count = int((df[target_column] == label).sum())
                    result['errors'][label] = {
                        'correct_label': 'UNKNOWN',
                        'count': count,
                    }
                    result['error_count'] += count

        return result

    def _validate_data_types(self, df: pd.DataFrame,
                              columns: List[str]) -> Dict:
        """验证数据类型"""
        result = {
            'issues': [],
        }

        for col in columns:
            # 检查是否为数值类型
            if not pd.api.types.is_numeric_dtype(df[col]):
                non_numeric = df[col].apply(
                    lambda x: not isinstance(
                        x, (int, float, np.integer, np.floating)
                    ) if pd.notna(x) else False
                )
                if non_numeric.any():
                    result['issues'].append({
                        'column': col,
                        'issue': 'non_numeric_values',
                        'count': int(non_numeric.sum()),
                    })

        return result
