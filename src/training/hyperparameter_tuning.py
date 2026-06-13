"""超参数优化模块

使用Optuna进行自动化超参数搜索。
"""

from typing import Dict, Optional
import pandas as pd
import numpy as np

from src.utils.logger import get_logger

logger = get_logger("dry_bean.training.tuning")


class HyperparameterTuner:
    """超参数优化器

    使用Optuna进行贝叶斯超参数优化。
    """

    def __init__(self, model_name: str, config: Dict = None):
        """
        Args:
            model_name: 模型名称
            config: 配置字典
        """
        self.model_name = model_name
        self.config = config or {}

    def optimize(self, X_train: pd.DataFrame, y_train: pd.Series,
                 X_val: pd.DataFrame, y_val: pd.Series,
                 n_trials: int = 50) -> Dict:
        """执行超参数优化

        Args:
            X_train: 训练集特征
            y_train: 训练集标签
            X_val: 验证集特征
            y_val: 验证集标签
            n_trials: 优化轮数

        Returns:
            最佳超参数字典
        """
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            logger.warning("Optuna未安装，使用默认参数")
            return self._get_default_params()

        logger.info(f"开始超参数优化: {self.model_name}, n_trials={n_trials}")

        def objective(trial):
            params = self._suggest_params(trial)
            model = self._create_model_with_params(params)

            try:
                history = model.train(X_train, y_train, X_val, y_val)
                val_acc = history.get('val_accuracy', 0)
                if val_acc is None:
                    val_acc = history.get('train_accuracy', 0)
                return val_acc
            except Exception as e:
                logger.warning(f"Trial failed: {e}")
                return 0.0

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        best_params = study.best_params
        best_value = study.best_value

        logger.info(f"优化完成: best_val_accuracy={best_value:.4f}")
        logger.info(f"最佳参数: {best_params}")

        return best_params

    def _suggest_params(self, trial) -> Dict:
        """为每个算法定义搜索空间"""
        if self.model_name == 'random_forest':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 5, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
            }
        elif self.model_name == 'xgboost':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            }
        elif self.model_name == 'mlp':
            return {
                'hidden_dims': [
                    trial.suggest_int('hidden_0', 64, 512),
                    trial.suggest_int('hidden_1', 32, 256),
                    trial.suggest_int('hidden_2', 16, 128),
                ],
                'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
                'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
            }
        elif self.model_name == 'lightgbm':
            return {
                'num_leaves': trial.suggest_int('num_leaves', 15, 63),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
            }
        else:
            return {}

    def _create_model_with_params(self, params: Dict):
        """用给定参数创建模型"""
        if self.model_name == 'random_forest':
            from src.models.traditional import RFModel
            return RFModel(hyperparams=params)
        elif self.model_name == 'xgboost':
            from src.models.traditional import XGBoostModel
            return XGBoostModel(hyperparams=params)
        elif self.model_name == 'mlp':
            from src.models.deep_learning import PyTorchModel
            return PyTorchModel(model_type='mlp', config=params)
        elif self.model_name == 'lightgbm':
            from src.models.advanced import LightGBMModel
            return LightGBMModel(hyperparams=params)
        else:
            raise ValueError(f"Unknown model: {self.model_name}")

    def _get_default_params(self) -> Dict:
        """返回默认参数"""
        defaults = {
            'random_forest': {'n_estimators': 100, 'max_depth': 10},
            'xgboost': {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1},
            'mlp': {'hidden_dims': [256, 128, 64], 'dropout_rate': 0.3, 'learning_rate': 0.001},
            'lightgbm': {'num_leaves': 31, 'learning_rate': 0.05},
        }
        return defaults.get(self.model_name, {})
