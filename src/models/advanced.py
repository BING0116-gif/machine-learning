"""进阶算法模块

实现LightGBM分类器（课堂可能未讲过的进阶梯度提升算法）。
"""

import lightgbm as lgb
import pandas as pd
import numpy as np
import time
import joblib
from typing import Dict, Optional
from sklearn.preprocessing import LabelEncoder

from src.utils.logger import get_logger
from .base import BaseModel

logger = get_logger("dry_bean.models.advanced")


class LightGBMModel(BaseModel):
    """LightGBM分类器

    课堂可能未讲过的进阶梯度提升算法。
    特点：
    - 训练速度快（基于直方图的决策树算法）
    - 内存效率高（GOSS + EFB技术）
    - 支持大规模数据
    - 有良好的可解释性
    """

    DEFAULT_PARAMS = {
        'objective': 'multiclass',
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'max_depth': -1,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'n_jobs': -1,
        'random_state': 42,
        'num_boost_round': 1000,
        'early_stopping_rounds': 50,
    }

    def __init__(self, hyperparams: Dict = None, n_classes: int = 7):
        super().__init__('lightgbm', hyperparams)
        params = {**self.DEFAULT_PARAMS, **(hyperparams or {})}
        params['num_class'] = n_classes
        self.params = params
        self.model = None
        self.label_encoder = LabelEncoder()
        self.n_classes = n_classes

    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: pd.DataFrame = None, y_val: pd.Series = None) -> Dict:
        """训练LightGBM模型"""
        start_time = time.time()

        y_train_encoded = self.label_encoder.fit_transform(y_train)

        # 动态更新类别数
        actual_n_classes = len(self.label_encoder.classes_)
        if actual_n_classes != self.n_classes:
            self.n_classes = actual_n_classes
            self.params['num_class'] = actual_n_classes

        train_data = lgb.Dataset(X_train, label=y_train_encoded)

        valid_sets = [train_data]
        valid_names = ['train']

        if X_val is not None and y_val is not None:
            y_val_encoded = self.label_encoder.transform(y_val)
            val_data = lgb.Dataset(X_val, label=y_val_encoded, reference=train_data)
            valid_sets.append(val_data)
            valid_names.append('valid')

        num_boost_round = self.params.pop('num_boost_round', 1000)
        early_stopping_rounds = self.params.pop('early_stopping_rounds', 50)

        # 更新callbacks中的early_stopping
        callbacks = [
            lgb.early_stopping(stopping_rounds=early_stopping_rounds),
            lgb.log_evaluation(period=100),
        ]

        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )

        training_time = time.time() - start_time

        train_pred = self.model.predict(X_train)
        train_pred_labels = np.argmax(train_pred, axis=1)
        train_acc = float((train_pred_labels == y_train_encoded).mean())

        val_acc = None
        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            val_pred_labels = np.argmax(val_pred, axis=1)
            val_acc = float((val_pred_labels == y_val_encoded).mean())

        self.is_fitted = True
        self.training_history = {
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'training_time_seconds': training_time,
            'best_iteration': getattr(self.model, 'best_iteration', None),
            'num_trees': self.model.num_trees(),
            'feature_importance': self.model.feature_importance().tolist(),
        }

        val_acc_str = f"{val_acc:.4f}" if val_acc is not None else "N/A"
        logger.info(f"LightGBM 训练完成: train_acc={train_acc:.4f}, val_acc={val_acc_str}, time={training_time:.2f}s")
        return self.training_history

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """预测类别"""
        if not self.is_fitted:
            raise RuntimeError("Model not trained yet.")
        proba = self.model.predict(X)
        pred_labels = np.argmax(proba, axis=1)
        return self.label_encoder.inverse_transform(pred_labels)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """预测概率"""
        if not self.is_fitted:
            raise RuntimeError("Model not trained yet.")
        return self.model.predict(X)

    def get_feature_importance(self, importance_type: str = 'gain') -> np.ndarray:
        """获取特征重要性"""
        if not self.is_fitted:
            raise RuntimeError("Model not trained yet.")
        return self.model.feature_importance(importance_type=importance_type)

    def save_model(self, path: str):
        """保存模型"""
        self.model.save_model(path + '_lgbm.txt')
        joblib.dump({
            'params': self.params,
            'label_encoder': self.label_encoder,
            'model_name': self.model_name,
            'hyperparams': self.hyperparams,
            'training_history': self.training_history,
            'n_classes': self.n_classes,
        }, path + '_meta.pkl')
        logger.info(f"模型已保存: {path}")

    @classmethod
    def load_model(cls, path: str) -> 'LightGBMModel':
        """加载模型"""
        meta = joblib.load(path + '_meta.pkl')
        instance = cls(hyperparams=meta['hyperparams'], n_classes=meta['n_classes'])
        instance.params = meta['params']
        instance.model = lgb.Booster(model_file=path + '_lgbm.txt')
        instance.label_encoder = meta['label_encoder']
        instance.training_history = meta['training_history']
        instance.is_fitted = True
        return instance
