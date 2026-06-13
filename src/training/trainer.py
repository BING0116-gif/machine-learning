"""训练管理器模块

统一管理模型的训练过程，包括训练执行、结果记录、模型保存。
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np

from src.utils.logger import get_logger
from src.utils.io_utils import ensure_dir, save_json
from src.models.base import BaseModel

logger = get_logger("dry_bean.training")


class Trainer:
    """训练管理器

    统一管理模型的训练过程，包括：
    - 训练执行
    - 结果记录
    - 模型保存
    - 实验追踪
    """

    def __init__(self, output_dir: str = './results'):
        """
        Args:
            output_dir: 结果输出目录
        """
        self.output_dir = output_dir
        ensure_dir(output_dir)
        ensure_dir(os.path.join(output_dir, 'models'))
        ensure_dir(os.path.join(output_dir, 'experiments'))

    def train_model(self, model: BaseModel,
                    X_train: pd.DataFrame, y_train: pd.Series,
                    X_val: pd.DataFrame, y_val: pd.Series,
                    experiment_name: str = None,
                    save_model: bool = True) -> Dict:
        """训练单个模型

        Args:
            model: 模型实例
            X_train: 训练特征
            y_train: 训练标签
            X_val: 验证特征
            y_val: 验证标签
            experiment_name: 实验名称
            save_model: 是否保存模型

        Returns:
            包含训练结果的字典
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_name = experiment_name or f"{model.get_model_name()}_{timestamp}"

        logger.info(f"\n{'='*60}")
        logger.info(f"Training: {exp_name}")
        logger.info(f"{'='*60}")
        logger.info(f"Model: {model.get_model_name()}")
        logger.info(f"Train samples: {len(X_train)}")
        logger.info(f"Val samples: {len(X_val)}")
        logger.info(f"{'='*60}\n")

        # 执行训练
        training_history = model.train(X_train, y_train, X_val, y_val)

        # 保存模型
        model_path = None
        if save_model:
            model_dir = os.path.join(self.output_dir, 'models')
            ensure_dir(model_dir)
            model_path = os.path.join(model_dir, exp_name)
            model.save_model(model_path)
            logger.info(f"模型已保存: {model_path}")

        # 记录实验
        experiment_record = {
            'experiment_id': exp_name,
            'timestamp': datetime.now().isoformat(),
            'model_name': model.get_model_name(),
            'hyperparameters': model.get_hyperparams(),
            'metrics': training_history,
            'artifacts': {
                'model_path': model_path,
            },
            'status': 'completed',
        }

        # 保存实验记录
        exp_dir = os.path.join(self.output_dir, 'experiments')
        ensure_dir(exp_dir)
        record_path = os.path.join(exp_dir, f'{exp_name}.json')
        save_json(experiment_record, record_path)

        logger.info(f"实验记录已保存: {record_path}")

        return experiment_record

    def train_multiple_models(self, models: List[BaseModel],
                              X_train: pd.DataFrame, y_train: pd.Series,
                              X_val: pd.DataFrame, y_val: pd.Series,
                              prefix: str = 'experiment') -> List[Dict]:
        """批量训练多个模型

        Args:
            models: 模型列表
            X_train, y_train: 训练数据
            X_val, y_val: 验证数据
            prefix: 实验名称前缀

        Returns:
            实验结果列表
        """
        results = []

        for i, model in enumerate(models):
            exp_name = f'{prefix}_{model.get_model_name()}_{i}'
            try:
                result = self.train_model(
                    model, X_train, y_train, X_val, y_val,
                    experiment_name=exp_name
                )
                results.append(result)
            except Exception as e:
                logger.error(f"训练 {model.get_model_name()} 失败: {e}")
                results.append({
                    'model_name': model.get_model_name(),
                    'status': 'failed',
                    'error': str(e),
                })

        return results
