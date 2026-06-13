"""传统机器学习模型模块

实现Random Forest和XGBoost分类器。
"""

from typing import Dict, Optional
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import time

from src.utils.logger import get_logger
from .base import BaseModel

logger = get_logger("dry_bean.models")


class RFModel(BaseModel):
    """随机森林分类器"""

    DEFAULT_PARAMS = {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': 42,
        'n_jobs': -1,
        'class_weight': 'balanced',
    }

    def __init__(self, hyperparams: Dict = None):
        super().__init__('random_forest', hyperparams)
        params = {**self.DEFAULT_PARAMS, **(hyperparams or {})}
        self.model = RandomForestClassifier(**params)
        self.label_encoder = LabelEncoder()

    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: pd.DataFrame = None, y_val: pd.Series = None) -> Dict:
        """训练随机森林"""
        start_time = time.time()

        y_train_encoded = self.label_encoder.fit_transform(y_train)

        self.model.fit(X_train, y_train_encoded)

        training_time = time.time() - start_time

        train_pred = self.model.predict(X_train)
        train_acc = float((train_pred == y_train_encoded).mean())

        val_acc = None
        if X_val is not None and y_val is not None:
            y_val_encoded = self.label_encoder.transform(y_val)
            val_pred = self.model.predict(X_val)
            val_acc = float((val_pred == y_val_encoded).mean())

        self.is_fitted = True
        self.training_history = {
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'training_time_seconds': training_time,
            'n_estimators': self.model.n_estimators,
            'max_depth': self.model.max_depth,
            'feature_importances': self.model.feature_importances_.tolist(),
        }

        val_acc_str = f"{val_acc:.4f}" if val_acc is not None else "N/A"
        logger.info(f"Random Forest 训练完成: train_acc={train_acc:.4f}, val_acc={val_acc_str}, time={training_time:.2f}s")
        return self.training_history

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """预测类别"""
        if not self.is_fitted:
            raise RuntimeError("Model not trained yet.")
        pred_encoded = self.model.predict(X)
        return self.label_encoder.inverse_transform(pred_encoded)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """预测概率"""
        if not self.is_fitted:
            raise RuntimeError("Model not trained yet.")
        return self.model.predict_proba(X)

    def save_model(self, path: str):
        """保存模型"""
        joblib.dump({
            'model': self.model,
            'label_encoder': self.label_encoder,
            'model_name': self.model_name,
            'hyperparams': self.hyperparams,
            'training_history': self.training_history,
        }, path)
        logger.info(f"模型已保存: {path}")

    @classmethod
    def load_model(cls, path: str) -> 'RFModel':
        """加载模型"""
        data = joblib.load(path)
        instance = cls(hyperparams=data['hyperparams'])
        instance.model = data['model']
        instance.label_encoder = data['label_encoder']
        instance.training_history = data['training_history']
        instance.is_fitted = True
        return instance


class XGBoostModel(BaseModel):
    """XGBoost分类器"""

    DEFAULT_PARAMS = {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'eval_metric': 'mlogloss',
        'objective': 'multi:softprob',
    }

    def __init__(self, hyperparams: Dict = None, n_classes: int = 7):
        super().__init__('xgboost', hyperparams)
        params = {**self.DEFAULT_PARAMS, **(hyperparams or {})}
        params['num_class'] = n_classes
        self.model_params = params
        from xgboost import XGBClassifier
        self.model = XGBClassifier(**params)
        self.label_encoder = LabelEncoder()
        self.n_classes = n_classes

    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: pd.DataFrame = None, y_val: pd.Series = None) -> Dict:
        """训练XGBoost"""
        start_time = time.time()

        y_train_encoded = self.label_encoder.fit_transform(y_train)

        eval_set = None
        if X_val is not None and y_val is not None:
            y_val_encoded = self.label_encoder.transform(y_val)
            eval_set = [(X_train, y_train_encoded), (X_val, y_val_encoded)]

        self.model.fit(
            X_train, y_train_encoded,
            eval_set=eval_set,
            verbose=False,
        )

        training_time = time.time() - start_time

        train_pred = self.model.predict(X_train)
        train_acc = float((train_pred == y_train_encoded).mean())

        val_acc = None
        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            val_acc = float((val_pred == y_val_encoded).mean())

        # 获取eval结果
        evals_result = getattr(self.model, 'evals_result_', None)

        self.is_fitted = True
        self.training_history = {
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'training_time_seconds': training_time,
            'best_iteration': getattr(self.model, 'best_iteration', None),
        }

        # 保存loss曲线数据
        if evals_result:
            history_data = {}
            for dataset_name, metrics in evals_result.items():
                for metric_name, values in metrics.items():
                    key = f"{dataset_name}_{metric_name}"
                    history_data[key] = [float(v) for v in values]
            self.training_history['evals_result'] = history_data

        val_acc_str = f"{val_acc:.4f}" if val_acc is not None else "N/A"
        logger.info(f"XGBoost 训练完成: train_acc={train_acc:.4f}, val_acc={val_acc_str}, time={training_time:.2f}s")
        return self.training_history

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """预测类别"""
        if not self.is_fitted:
            raise RuntimeError("Model not trained yet.")
        pred_encoded = self.model.predict(X)
        return self.label_encoder.inverse_transform(pred_encoded.astype(int))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """预测概率"""
        if not self.is_fitted:
            raise RuntimeError("Model not trained yet.")
        return self.model.predict_proba(X)

    def save_model(self, path: str):
        """保存模型"""
        self.model.save_model(path + '_xgb.model')
        joblib.dump({
            'label_encoder': self.label_encoder,
            'model_name': self.model_name,
            'hyperparams': self.hyperparams,
            'training_history': self.training_history,
            'n_classes': self.n_classes,
            'model_params': self.model_params,
        }, path + '_meta.pkl')
        logger.info(f"模型已保存: {path}")

    @classmethod
    def load_model(cls, path: str) -> 'XGBoostModel':
        """加载模型"""
        meta = joblib.load(path + '_meta.pkl')
        instance = cls(hyperparams=meta['hyperparams'], n_classes=meta.get('n_classes', 7))
        from xgboost import XGBClassifier
        instance.model = XGBClassifier()
        instance.model.load_model(path + '_xgb.model')
        instance.label_encoder = meta['label_encoder']
        instance.training_history = meta['training_history']
        instance.is_fitted = True
        return instance
