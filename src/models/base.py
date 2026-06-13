"""模型基类模块

定义统一的模型接口，所有模型必须继承此类。
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np


class BaseModel(ABC):
    """模型基类

    所有模型必须继承此类并实现抽象方法。
    提供统一的接口规范。
    """

    def __init__(self, model_name: str, hyperparams: Dict = None):
        """
        Args:
            model_name: 模型名称标识符
            hyperparams: 超参数字典
        """
        self.model_name = model_name
        self.hyperparams = hyperparams or {}
        self.model = None
        self.training_history = {}
        self.is_fitted = False

    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: pd.DataFrame = None, y_val: pd.Series = None) -> Dict:
        """训练模型"""
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """预测类别"""
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """预测概率"""
        pass

    @abstractmethod
    def save_model(self, path: str):
        """保存模型到文件"""
        pass

    @classmethod
    @abstractmethod
    def load_model(cls, path: str) -> 'BaseModel':
        """从文件加载模型"""
        pass

    def get_model_name(self) -> str:
        """获取模型名称"""
        return self.model_name

    def get_hyperparams(self) -> Dict:
        """获取超参数"""
        return self.hyperparams

    def get_training_history(self) -> Dict:
        """获取训练历史"""
        return self.training_history
