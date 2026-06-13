"""模型集成模块

实现Voting和Stacking集成方法。
"""

from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import joblib
import time

from src.utils.logger import get_logger
from .base import BaseModel

logger = get_logger("dry_bean.models.ensemble")


class VotingEnsemble(BaseModel):
    """投票集成分类器"""

    def __init__(self, config: Dict = None, hyperparams: Dict = None):
        super().__init__('voting', hyperparams)
        self.config = config or {}
        self.label_encoder = LabelEncoder()
        self.model = None
        self.base_models = {}

    def set_base_models(self, models: Dict[str, object]):
        """设置基础模型

        Args:
            models: 模型名称到模型实例的映射
        """
        self.base_models = models

    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: pd.DataFrame = None, y_val: pd.Series = None) -> Dict:
        """训练Voting集成模型"""
        start_time = time.time()

        y_train_encoded = self.label_encoder.fit_transform(y_train)

        # 构建sklearn兼容的estimator列表
        estimators = []
        for name, model in self.base_models.items():
            if hasattr(model, 'model'):
                estimators.append((name, model.model))

        if not estimators:
            raise ValueError("No base models provided for voting ensemble")

        voting = self.config.get('voting', 'soft')
        self.model = VotingClassifier(
            estimators=estimators,
            voting=voting,
        )

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
            'voting_type': voting,
            'n_base_models': len(estimators),
        }

        logger.info(f"Voting Ensemble 训练完成: train_acc={train_acc:.4f}, val_acc={val_acc:.4f if val_acc else 'N/A'}")
        return self.training_history

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model not trained yet.")
        pred_encoded = self.model.predict(X)
        return self.label_encoder.inverse_transform(pred_encoded)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model not trained yet.")
        return self.model.predict_proba(X)

    def save_model(self, path: str):
        joblib.dump({
            'model': self.model,
            'label_encoder': self.label_encoder,
            'model_name': self.model_name,
            'hyperparams': self.hyperparams,
            'training_history': self.training_history,
            'config': self.config,
        }, path)

    @classmethod
    def load_model(cls, path: str) -> 'VotingEnsemble':
        data = joblib.load(path)
        instance = cls(config=data['config'], hyperparams=data['hyperparams'])
        instance.model = data['model']
        instance.label_encoder = data['label_encoder']
        instance.training_history = data['training_history']
        instance.is_fitted = True
        return instance


class StackingEnsemble(BaseModel):
    """堆叠集成分类器"""

    def __init__(self, config: Dict = None, hyperparams: Dict = None):
        super().__init__('stacking', hyperparams)
        self.config = config or {}
        self.label_encoder = LabelEncoder()
        self.model = None
        self.base_models = {}

    def set_base_models(self, models: Dict[str, object]):
        """设置基础模型"""
        self.base_models = models

    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: pd.DataFrame = None, y_val: pd.Series = None) -> Dict:
        """训练Stacking集成模型"""
        start_time = time.time()

        y_train_encoded = self.label_encoder.fit_transform(y_train)

        estimators = []
        for name, model in self.base_models.items():
            if hasattr(model, 'model'):
                estimators.append((name, model.model))

        if not estimators:
            raise ValueError("No base models provided for stacking ensemble")

        final_estimator = LogisticRegression(
            multi_class='multinomial', max_iter=1000, random_state=42
        )

        self.model = StackingClassifier(
            estimators=estimators,
            final_estimator=final_estimator,
            cv=5,
        )

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
            'n_base_models': len(estimators),
            'meta_learner': 'LogisticRegression',
        }

        logger.info(f"Stacking Ensemble 训练完成: train_acc={train_acc:.4f}, val_acc={val_acc:.4f if val_acc else 'N/A'}")
        return self.training_history

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model not trained yet.")
        pred_encoded = self.model.predict(X)
        return self.label_encoder.inverse_transform(pred_encoded)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model not trained yet.")
        return self.model.predict_proba(X)

    def save_model(self, path: str):
        joblib.dump({
            'model': self.model,
            'label_encoder': self.label_encoder,
            'model_name': self.model_name,
            'hyperparams': self.hyperparams,
            'training_history': self.training_history,
            'config': self.config,
        }, path)

    @classmethod
    def load_model(cls, path: str) -> 'StackingEnsemble':
        data = joblib.load(path)
        instance = cls(config=data['config'], hyperparams=data['hyperparams'])
        instance.model = data['model']
        instance.label_encoder = data['label_encoder']
        instance.training_history = data['training_history']
        instance.is_fitted = True
        return instance
