"""可解释性分析模块

提供SHAP和LIME可解释性分析功能。
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import os

from src.utils.logger import get_logger
from src.utils.io_utils import ensure_dir

logger = get_logger("dry_bean.evaluation.interpretability")


class InterpretabilityAnalyzer:
    """可解释性分析器

    提供SHAP和Feature Importance分析功能。
    """

    def __init__(self, feature_names: List[str] = None,
                 class_names: List[str] = None):
        """
        Args:
            feature_names: 特征名称列表
            class_names: 类别名称列表
        """
        self.feature_names = feature_names
        self.class_names = class_names

    def compute_shap_values(self, model, X: pd.DataFrame,
                           model_type: str = 'tree') -> Dict:
        """计算SHAP值

        Args:
            model: 训练好的模型
            X: 输入数据
            model_type: 模型类型 ('tree' 或 'deep')

        Returns:
            SHAP分析结果字典
        """
        try:
            import shap
        except ImportError:
            logger.warning("SHAP未安装，跳过SHAP分析")
            return {}

        logger.info(f"计算SHAP值 (model_type={model_type})...")

        try:
            if model_type == 'tree':
                # 对于sklearn/xgboost/lightgbm模型
                if hasattr(model, 'model'):
                    inner_model = model.model
                else:
                    inner_model = model

                explainer = shap.TreeExplainer(inner_model)
                shap_values = explainer.shap_values(X)

            elif model_type == 'deep':
                # 对于PyTorch模型
                if hasattr(model, 'model'):
                    inner_model = model.model
                else:
                    inner_model = model

                # 使用KernelExplainer作为通用方法
                def model_predict(x):
                    x_df = pd.DataFrame(x, columns=X.columns)
                    return model.predict_proba(x_df)

                background = shap.sample(X, 100)
                explainer = shap.KernelExplainer(model_predict, background)
                shap_values = explainer.shap_values(X[:50])  # 限制样本数
            else:
                logger.warning(f"不支持的模型类型: {model_type}")
                return {}

            result = {
                'shap_values': shap_values,
                'feature_names': self.feature_names or list(X.columns),
                'class_names': self.class_names,
            }

            logger.info("SHAP值计算完成")
            return result

        except Exception as e:
            logger.error(f"SHAP分析失败: {e}")
            return {}

    def plot_shap_summary(self, shap_result: Dict, save_path: str = None):
        """绘制SHAP Summary Plot"""
        try:
            import shap
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("SHAP或matplotlib未安装")
            return

        shap_values = shap_result.get('shap_values')
        feature_names = shap_result.get('feature_names')

        if shap_values is None:
            return

        try:
            if isinstance(shap_values, list):
                # 多分类情况，绘制第一个类别的summary plot
                shap.summary_plot(shap_values[0], feature_names=feature_names, show=False)
            else:
                shap.summary_plot(shap_values, feature_names=feature_names, show=False)

            if save_path:
                ensure_dir(os.path.dirname(save_path))
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                logger.info(f"SHAP Summary Plot已保存: {save_path}")

            plt.close('all')
        except Exception as e:
            logger.error(f"绘制SHAP Summary Plot失败: {e}")
            plt.close('all')

    def compute_feature_importance(self, model, model_name: str) -> Dict:
        """计算特征重要性

        Args:
            model: 训练好的模型
            model_name: 模型名称

        Returns:
            特征重要性字典
        """
        result = {}

        try:
            if model_name in ('random_forest',):
                if hasattr(model, 'model') and hasattr(model.model, 'feature_importances_'):
                    importances = model.model.feature_importances_
                    result['importances'] = importances.tolist()
                    result['method'] = 'gini_importance'

            elif model_name in ('xgboost',):
                if hasattr(model, 'model') and hasattr(model.model, 'feature_importances_'):
                    importances = model.model.feature_importances_
                    result['importances'] = importances.tolist()
                    result['method'] = 'weight_importance'

            elif model_name == 'lightgbm':
                if hasattr(model, 'model'):
                    importances = model.model.feature_importance()
                    result['importances'] = importances.tolist()
                    result['method'] = 'split_importance'

            if result and self.feature_names:
                result['feature_names'] = self.feature_names
                # 排序
                indices = np.argsort(result['importances'])[::-1]
                result['sorted_features'] = [
                    (self.feature_names[i], result['importances'][i])
                    for i in indices
                ]

        except Exception as e:
            logger.error(f"计算特征重要性失败: {e}")

        return result
