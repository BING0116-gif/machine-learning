"""可视化图表生成模块

生成论文所需的各种高质量图表，包括EDA、训练曲线、评估对比等。
"""

import matplotlib
matplotlib.use('Agg')  # 非交互式后端

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import os

from src.utils.logger import get_logger
from src.utils.io_utils import ensure_dir

logger = get_logger("dry_bean.visualization")

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'


class PlotGenerator:
    """图表生成器

    生成论文所需的各种高质量图表。
    """

    def __init__(self, figsize: Tuple[int, int] = (10, 6), dpi: int = 150):
        """
        Args:
            figsize: 默认图形大小
            dpi: 分辨率
        """
        self.default_figsize = figsize
        self.dpi = dpi

    def generate_eda_plots(self, df: pd.DataFrame, feature_columns: List[str],
                           target_column: str, output_dir: str) -> None:
        """生成所有EDA图表

        Args:
            df: 输入DataFrame
            feature_columns: 特征列名列表
            target_column: 目标列名
            output_dir: 输出目录
        """
        ensure_dir(output_dir)
        logger.info(f"开始生成EDA图表，输出目录: {output_dir}")

        # 1. 类别分布图
        self._plot_class_distribution(df, target_column, output_dir)

        # 2. 特征分布直方图
        self._plot_feature_distributions(df, feature_columns, target_column, output_dir)

        # 3. 相关性热力图
        self._plot_correlation_matrix(df, feature_columns, output_dir)

        # 4. 缺失值热力图
        self._plot_missing_values(df, output_dir)

        # 5. 箱线图（按类别分组）
        self._plot_boxplots(df, feature_columns, target_column, output_dir)

        # 6. 特征对散点图
        self._plot_pairplot(df, feature_columns, target_column, output_dir)

        logger.info(f"EDA图表生成完成")

    def _plot_class_distribution(self, df: pd.DataFrame, target_column: str,
                                  output_dir: str) -> None:
        """绘制类别分布图"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        class_counts = df[target_column].value_counts()

        # 柱状图
        colors = plt.cm.Set3(np.linspace(0, 1, len(class_counts)))
        bars = axes[0].bar(class_counts.index, class_counts.values, color=colors, edgecolor='black')
        for bar, count in zip(bars, class_counts.values):
            axes[0].annotate(f'{count}', xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                           xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
        axes[0].set_xlabel('Class')
        axes[0].set_ylabel('Count')
        axes[0].set_title('Class Distribution (Bar Chart)')
        axes[0].tick_params(axis='x', rotation=45)

        # 饼图
        axes[1].pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%',
                    colors=colors, startangle=90)
        axes[1].set_title('Class Distribution (Pie Chart)')

        plt.tight_layout()
        save_path = os.path.join(output_dir, 'class_distribution.png')
        fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"保存类别分布图: {save_path}")

    def _plot_feature_distributions(self, df: pd.DataFrame, feature_columns: List[str],
                                     target_column: str, output_dir: str) -> None:
        """绘制特征分布直方图"""
        dist_dir = os.path.join(output_dir, 'feature_distributions')
        ensure_dir(dist_dir)

        for col in feature_columns:
            fig, ax = plt.subplots(figsize=self.default_figsize)

            for cls in df[target_column].unique():
                subset = df[df[target_column] == cls][col].dropna()
                ax.hist(subset, bins=30, alpha=0.5, label=cls, density=True)

            ax.set_xlabel(col)
            ax.set_ylabel('Density')
            ax.set_title(f'Distribution of {col}')
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)

            save_path = os.path.join(dist_dir, f'dist_{col}.png')
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close(fig)

        logger.info(f"保存特征分布图: {dist_dir}")

    def _plot_correlation_matrix(self, df: pd.DataFrame, feature_columns: List[str],
                                  output_dir: str) -> None:
        """绘制相关性热力图"""
        corr_matrix = df[feature_columns].corr()

        fig, ax = plt.subplots(figsize=(14, 12))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
                   cmap='RdBu_r', center=0, ax=ax,
                   square=True, linewidths=0.5,
                   cbar_kws={"shrink": 0.8})

        ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()

        save_path = os.path.join(output_dir, 'correlation_matrix.png')
        fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"保存相关性热力图: {save_path}")

    def _plot_missing_values(self, df: pd.DataFrame, output_dir: str) -> None:
        """绘制缺失值热力图"""
        missing = df.isnull()

        if not missing.any().any():
            fig, ax = plt.subplots(figsize=(10, 2))
            ax.text(0.5, 0.5, 'No Missing Values Found', ha='center', va='center', fontsize=14)
            ax.set_title('Missing Values Heatmap')
        else:
            fig, ax = plt.subplots(figsize=(14, 8))
            # 只显示有缺失值的列
            missing_cols = missing.any()
            cols_with_missing = missing_cols[missing_cols].index.tolist()

            if cols_with_missing:
                sns.heatmap(missing[cols_with_missing].T, cmap='YlOrRd',
                           cbar_kws={'label': 'Missing'}, ax=ax)
                ax.set_title('Missing Values Heatmap')
                ax.set_ylabel('Features')
                ax.set_xlabel('Sample Index')
            else:
                ax.text(0.5, 0.5, 'No Missing Values Found', ha='center', va='center', fontsize=14)
                ax.set_title('Missing Values Heatmap')

        plt.tight_layout()
        save_path = os.path.join(output_dir, 'missing_values_heatmap.png')
        fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"保存缺失值热力图: {save_path}")

    def _plot_boxplots(self, df: pd.DataFrame, feature_columns: List[str],
                       target_column: str, output_dir: str) -> None:
        """绘制箱线图（按类别分组）"""
        box_dir = os.path.join(output_dir, 'boxplots')
        ensure_dir(box_dir)

        for col in feature_columns:
            fig, ax = plt.subplots(figsize=(12, 6))

            classes = sorted(df[target_column].unique())
            data_by_class = [df[df[target_column] == cls][col].dropna().values for cls in classes]

            bp = ax.boxplot(data_by_class, labels=classes, patch_artist=True)
            colors = plt.cm.Set3(np.linspace(0, 1, len(classes)))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)

            ax.set_xlabel('Class')
            ax.set_ylabel(col)
            ax.set_title(f'{col} by Class')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3, axis='y')

            save_path = os.path.join(box_dir, f'boxplot_{col}.png')
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close(fig)

        logger.info(f"保存箱线图: {box_dir}")

    def _plot_pairplot(self, df: pd.DataFrame, feature_columns: List[str],
                       target_column: str, output_dir: str) -> None:
        """绘制关键特征对的关系图"""
        # 选择最重要的几个特征绘制pairplot
        key_features = ['Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength',
                        'Eccentricity', 'roundness']
        available_features = [f for f in key_features if f in feature_columns]

        if len(available_features) < 2:
            return

        # 采样以加速绘图
        sample_df = df.groupby(target_column, group_keys=False).apply(
            lambda x: x.sample(min(200, len(x)), random_state=42)
        ).reset_index(drop=True)

        if target_column not in sample_df.columns:
            return

        fig = sns.pairplot(
            sample_df[available_features + [target_column]],
            hue=target_column,
            diag_kind='kde',
            plot_kws={'alpha': 0.5, 's': 10},
            corner=True
        )
        fig.fig.suptitle('Feature Pairwise Relationships', y=1.02, fontsize=14)

        save_path = os.path.join(output_dir, 'pairplot.png')
        fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close('all')
        logger.info(f"保存特征对关系图: {save_path}")

    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                             class_names: List[str],
                             title: str = "Confusion Matrix",
                             normalize: bool = True,
                             save_path: str = None) -> plt.Figure:
        """绘制混淆矩阵"""
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(y_true, y_pred)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        fig, ax = plt.subplots(figsize=(10, 8))

        sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                   xticklabels=class_names, yticklabels=class_names,
                   cmap='Blues', ax=ax)

        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')

        plt.tight_layout()

        if save_path:
            ensure_dir(os.path.dirname(save_path))
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"保存混淆矩阵: {save_path}")

        return fig

    def plot_accuracy_comparison(self, results: Dict[str, Dict],
                                metric: str = 'accuracy',
                                title: str = "Model Accuracy Comparison",
                                save_path: str = None) -> plt.Figure:
        """绘制模型精度对比柱状图"""
        models = list(results.keys())
        scores = [results[m].get(metric, 0) for m in models]

        fig, ax = plt.subplots(figsize=self.default_figsize)

        colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
        bars = ax.bar(models, scores, color=colors, edgecolor='black')

        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.annotate(f'{score:.4f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=11, fontweight='bold')

        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
        ax.set_xlabel('Models', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1.1)

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        if save_path:
            ensure_dir(os.path.dirname(save_path))
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        return fig

    def plot_learning_curves(self, histories: Dict[str, Dict],
                            title: str = "Learning Curves",
                            save_path: str = None) -> plt.Figure:
        """绘制Loss/Accuracy曲线"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        colors = plt.cm.tab10(np.linspace(0, 1, len(histories)))

        for (model_name, history), color in zip(histories.items(), colors):
            epochs = range(1, len(history.get('train_loss', [])) + 1)

            if 'train_loss' in history and history['train_loss']:
                axes[0].plot(epochs, history['train_loss'],
                           color=color, label=f'{model_name} (Train)', linewidth=2)
            if 'val_loss' in history and history['val_loss']:
                axes[0].plot(epochs, history['val_loss'],
                           color=color, linestyle='--', label=f'{model_name} (Val)', linewidth=2)

            if 'train_acc' in history and history['train_acc']:
                axes[1].plot(epochs, history['train_acc'],
                           color=color, label=f'{model_name} (Train)', linewidth=2)
            if 'val_acc' in history and history['val_acc']:
                axes[1].plot(epochs, history['val_acc'],
                           color=color, linestyle='--', label=f'{model_name} (Val)', linewidth=2)

        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Training & Validation Loss', fontsize=13, fontweight='bold')
        axes[0].legend(loc='upper right')
        axes[0].grid(True, alpha=0.3)

        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Accuracy', fontsize=12)
        axes[1].set_title('Training & Validation Accuracy', fontsize=13, fontweight='bold')
        axes[1].legend(loc='lower right')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(0, 1.05)

        plt.tight_layout()

        if save_path:
            ensure_dir(os.path.dirname(save_path))
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        return fig

    def plot_robustness_curve(self, robustness_results: Dict,
                             title: str = "Robustness Analysis",
                             save_path: str = None) -> plt.Figure:
        """绘制鲁棒性下降曲线"""
        fig, ax = plt.subplots(figsize=self.default_figsize)

        if 'noise_levels' in robustness_results:
            x = robustness_results['noise_levels']
            xlabel = 'Noise Level (σ)'
        elif 'missing_rates' in robustness_results:
            x = robustness_results['missing_rates']
            xlabel = 'Missing Rate'
        else:
            x = robustness_results.get('rates', [])
            xlabel = 'Noise Rate'

        baseline = robustness_results.get('baseline_accuracy', 1.0)
        accuracies = robustness_results['accuracies']

        ax.plot(x, accuracies, 'o-', linewidth=2, markersize=8,
               color='#2196F3', label='Noisy Accuracy')
        ax.axhline(y=baseline, color='#F44336', linestyle='--',
                  linewidth=2, label=f'Baseline ({baseline:.4f})')

        ax.fill_between(x, 0, accuracies, alpha=0.2, color='#2196F3')

        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='lower left', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

        plt.tight_layout()

        if save_path:
            ensure_dir(os.path.dirname(save_path))
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        return fig
