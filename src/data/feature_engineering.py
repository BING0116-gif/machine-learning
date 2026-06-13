"""特征工程模块

负责特征的标准化、归一化、选择等操作。
所有转换器都会被保存以便后续复用。
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
import joblib

from src.utils.logger import get_logger

logger = get_logger("dry_bean.feature_engineering")


class FeatureEngineer:
    """特征工程处理器

    负责特征的标准化、归一化、选择等操作。
    所有转换器都会被保存以便后续复用。
    """

    def __init__(self, config: Dict = None):
        """初始化特征工程器

        Args:
            config: 配置字典
        """
        default_config = {
            'scaling': 'standard',
            'feature_selection': 'importance',
            'n_features_to_select': 16,
            'correlation_threshold': 0.95,
        }
        self.config = config or default_config
        self.scalers = {}
        self.selector = None
        self.selected_features = None
        self.label_encoder = LabelEncoder()

    def fit_transform(self, X_train: pd.DataFrame,
                      X_val: pd.DataFrame = None,
                      X_test: pd.DataFrame = None,
                      y_train: pd.Series = None,
                      y_val: pd.Series = None,
                      y_test: pd.Series = None) -> Tuple:
        """在训练集上拟合并转换所有数据集

        Args:
            X_train: 训练集特征
            X_val: 验证集特征
            X_test: 测试集特征
            y_train: 训练集标签
            y_val: 验证集标签
            y_test: 测试集标签

        Returns:
            转换后的数据集元组 (X_train, [X_val, [X_test, [y_train_encoded, [y_val_encoded, y_test_encoded]]]])
        """
        # Step 1: 标签编码（对 train/val/test 统一编码）
        y_train_encoded = None
        y_val_encoded = None
        y_test_encoded = None

        if y_train is not None:
            y_train_encoded = pd.Series(
                self.label_encoder.fit_transform(y_train),
                index=y_train.index,
                name=y_train.name
            )
            # 使用已拟合的编码器转换 val 和 test 标签
            if y_val is not None:
                y_val_encoded = pd.Series(
                    self.label_encoder.transform(y_val),
                    index=y_val.index,
                    name=y_val.name
                )
            if y_test is not None:
                y_test_encoded = pd.Series(
                    self.label_encoder.transform(y_test),
                    index=y_test.index,
                    name=y_test.name
                )

        # Step 2: 特征缩放
        X_train_scaled = self._fit_scale(X_train)

        results = [X_train_scaled]

        if X_val is not None:
            X_val_scaled = self._transform_scale(X_val)
            results.append(X_val_scaled)

        if X_test is not None:
            X_test_scaled = self._transform_scale(X_test)
            results.append(X_test_scaled)

        # Step 3: 特征选择
        if y_train is not None and self.config.get('feature_selection', 'none') != 'none':
            results = self._fit_select(results[0], y_train, results[1:])

        # 将标签编码结果也加到返回中
        final_results = list(results)

        logger.info(f"特征工程完成: 缩放={self.config.get('scaling')}, "
                    f"特征选择={self.config.get('feature_selection')}")

        # 返回格式: (X_train, [X_val, [X_test, y_train_encoded, [y_val_encoded, y_test_encoded]]])
        final_results.append(y_train_encoded)
        if y_val_encoded is not None:
            final_results.append(y_val_encoded)
        if y_test_encoded is not None:
            final_results.append(y_test_encoded)

        return tuple(final_results)

    def _fit_scale(self, df: pd.DataFrame) -> pd.DataFrame:
        """拟合并执行特征缩放"""
        method = self.config.get('scaling', 'standard')

        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        elif method == 'none':
            return df.copy()
        else:
            raise ValueError(f"Unknown scaling method: {method}")

        scaled_data = scaler.fit_transform(df)
        self.scalers['feature_scaler'] = scaler

        return pd.DataFrame(scaled_data, index=df.index, columns=df.columns)

    def _transform_scale(self, df: pd.DataFrame) -> pd.DataFrame:
        """使用已拟合的缩放器转换数据"""
        if 'feature_scaler' not in self.scalers:
            raise RuntimeError("Scaler not fitted. Call fit_transform first.")

        scaled_data = self.scalers['feature_scaler'].transform(df)
        return pd.DataFrame(scaled_data, index=df.index, columns=df.columns)

    def _fit_select(self, X_train: pd.DataFrame, y_train: pd.Series,
                   other_datasets: List[pd.DataFrame]) -> List[pd.DataFrame]:
        """执行特征选择"""
        method = self.config.get('feature_selection', 'importance')
        n_features = self.config.get('n_features_to_select', 16)

        if method == 'importance':
            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X_train, y_train)
            importances = rf.feature_importances_
            indices = np.argsort(importances)[::-1][:n_features]
            self.selected_features = X_train.columns[indices].tolist()

        elif method == 'correlation':
            corr_matrix = X_train.corr().abs()
            upper_tri = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            threshold = self.config.get('correlation_threshold', 0.95)
            to_drop = [column for column in upper_tri.columns
                      if any(upper_tri[column] > threshold)]
            self.selected_features = [col for col in X_train.columns if col not in to_drop]

        elif method == 'rfe':
            estimator = RandomForestClassifier(n_estimators=50, random_state=42)
            selector = RFE(estimator, n_features_to_select=n_features)
            selector.fit(X_train, y_train)
            self.selector = selector
            self.selected_features = X_train.columns[selector.support_].tolist()

        elif method == 'none':
            self.selected_features = X_train.columns.tolist()

        else:
            raise ValueError(f"Unknown selection method: {method}")

        # 应用选择
        results = [X_train[self.selected_features]]
        for ds in other_datasets:
            results.append(ds[self.selected_features])

        logger.info(f"特征选择: {method}, 选择了 {len(self.selected_features)} 个特征")

        return results

    def get_selected_features(self) -> List[str]:
        """获取选择的特征列表"""
        if self.selected_features is None:
            return []
        return self.selected_features

    def get_label_encoder(self) -> LabelEncoder:
        """获取标签编码器"""
        return self.label_encoder

    def save(self, path: str):
        """保存所有拟合的转换器"""
        joblib.dump({
            'scalers': self.scalers,
            'selector': self.selector,
            'selected_features': self.selected_features,
            'config': self.config,
            'label_encoder_classes': self.label_encoder.classes_.tolist() if hasattr(self.label_encoder, 'classes_') else None,
        }, path)
        logger.info(f"特征工程器已保存: {path}")

    @classmethod
    def load(cls, path: str) -> 'FeatureEngineer':
        """加载保存的转换器"""
        data = joblib.load(path)
        engineer = cls(config=data['config'])
        engineer.scalers = data['scalers']
        engineer.selector = data['selector']
        engineer.selected_features = data['selected_features']
        if data.get('label_encoder_classes'):
            engineer.label_encoder = LabelEncoder()
            engineer.label_encoder.classes_ = np.array(data['label_encoder_classes'])
        return engineer
