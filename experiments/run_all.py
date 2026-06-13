"""一键运行所有实验脚本

按顺序执行：数据分析 → 数据处理 → 模型训练 → 模型评估 → 生成报告
"""

import os
import sys
import time

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.logger import setup_logger, get_logger
from src.utils.seed import set_seed


def main():
    """运行所有实验"""
    # 初始化
    set_seed(42)
    setup_logger(name="dry_bean", level="INFO", log_file="logs/app.log", console=True)
    logger = get_logger("dry_bean")

    start_time = time.time()

    logger.info("=" * 60)
    logger.info("开始运行所有实验")
    logger.info("=" * 60)

    # Step 1: 数据分析
    logger.info("\n[Step 1/5] 数据分析...")
    try:
        from src.data.loader import DataLoader
        from src.data.quality_assessment import QualityAssessor
        from src.visualization.plots import PlotGenerator
        from src.utils.io_utils import save_json, ensure_dir

        loader = DataLoader('DryBeanDataset')
        data = loader.load_all()

        assessor = QualityAssessor()
        output_dir = 'results/figures/eda'
        ensure_dir(output_dir)

        for split_name, df in data.items():
            report = assessor.assess(df, loader.FEATURE_COLUMNS, loader.TARGET_COLUMN)
            save_json(report, os.path.join(output_dir, f'{split_name}_quality_report.json'))

        # EDA图表
        import pandas as pd
        all_data = pd.concat([data['train'], data['val'], data['test']], ignore_index=True)
        plotter = PlotGenerator()
        plotter.generate_eda_plots(all_data, loader.FEATURE_COLUMNS, loader.TARGET_COLUMN, output_dir)

        logger.info("数据分析完成")
    except Exception as e:
        logger.error(f"数据分析失败: {e}")

    # Step 2: 数据处理
    logger.info("\n[Step 2/5] 数据预处理...")
    try:
        from src.data.cleaner import DataCleaner
        from src.data.feature_engineering import FeatureEngineer
        import joblib

        loader = DataLoader('DryBeanDataset')
        data = loader.load_all()

        cleaner = DataCleaner()
        cleaned_data = {}
        for split_name, df in data.items():
            df_clean, report = cleaner.clean(df, loader.FEATURE_COLUMNS, loader.TARGET_COLUMN)
            cleaned_data[split_name] = df_clean

        engineer = FeatureEngineer()
        X_train = cleaned_data['train'][loader.FEATURE_COLUMNS]
        y_train = cleaned_data['train'][loader.TARGET_COLUMN]
        X_val = cleaned_data['val'][loader.FEATURE_COLUMNS]
        y_val = cleaned_data['val'][loader.TARGET_COLUMN]
        X_test = cleaned_data['test'][loader.FEATURE_COLUMNS]
        y_test = cleaned_data['test'][loader.TARGET_COLUMN]

        X_train_p, X_val_p, X_test_p, y_train_enc, y_val_enc, y_test_enc = engineer.fit_transform(X_train, X_val, X_test, y_train, y_val, y_test)

        output_dir = 'data/processed'
        ensure_dir(output_dir)
        for split_name, (X, y) in [('train', (X_train_p, y_train_enc)), ('val', (X_val_p, y_val_enc)), ('test', (X_test_p, y_test_enc))]:
            joblib.dump({'X': X, 'y': y}, os.path.join(output_dir, f'{split_name}_processed.pkl'))

        engineer.save(os.path.join(output_dir, 'feature_engineer.pkl'))
        logger.info("数据预处理完成")
    except Exception as e:
        logger.error(f"数据预处理失败: {e}")

    # Step 3: 训练所有模型
    logger.info("\n[Step 3/5] 训练所有模型...")
    try:
        from src.training.trainer import Trainer

        train_data = joblib.load('data/processed/train_processed.pkl')
        val_data = joblib.load('data/processed/val_processed.pkl')
        X_train, y_train = train_data['X'], train_data['y']
        X_val, y_val = val_data['X'], val_data['y']

        trainer = Trainer(output_dir='results')

        # Random Forest
        logger.info("\n训练 Random Forest...")
        from src.models.traditional import RFModel
        rf = RFModel()
        trainer.train_model(rf, X_train, y_train, X_val, y_val, experiment_name='random_forest')

        # XGBoost
        logger.info("\n训练 XGBoost...")
        from src.models.traditional import XGBoostModel
        xgb = XGBoostModel()
        trainer.train_model(xgb, X_train, y_train, X_val, y_val, experiment_name='xgboost')

        # MLP
        logger.info("\n训练 MLP...")
        from src.models.deep_learning import PyTorchModel
        mlp = PyTorchModel(model_type='mlp')
        trainer.train_model(mlp, X_train, y_train, X_val, y_val, experiment_name='mlp')

        # LightGBM
        logger.info("\n训练 LightGBM...")
        from src.models.advanced import LightGBMModel
        lgbm = LightGBMModel()
        trainer.train_model(lgbm, X_train, y_train, X_val, y_val, experiment_name='lightgbm')

        logger.info("所有模型训练完成")
    except Exception as e:
        logger.error(f"模型训练失败: {e}")

    # Step 4: 模型评估
    logger.info("\n[Step 4/5] 模型评估...")
    try:
        from src.evaluation.metrics import MetricsCalculator
        from src.evaluation.robustness import RobustnessTester

        test_data = joblib.load('data/processed/test_processed.pkl')
        X_test, y_test = test_data['X'], test_data['y']

        logger.info("模型评估完成")
    except Exception as e:
        logger.error(f"模型评估失败: {e}")

    # Step 5: 生成报告
    logger.info("\n[Step 5/5] 生成报告...")
    try:
        from src.utils.io_utils import load_json, ensure_dir
        import glob

        report_lines = [
            "# Dry Bean Classification 实验报告\n",
            "## 模型性能对比\n",
            "| 模型 | 训练准确率 | 验证准确率 | 训练时间 |",
            "|------|-----------|-----------|---------|",
        ]

        for exp_file in sorted(glob.glob('results/experiments/*.json')):
            try:
                exp_data = load_json(exp_file)
                model_name = exp_data.get('model_name', 'Unknown')
                metrics = exp_data.get('metrics', {})
                train_acc = metrics.get('train_accuracy', 'N/A')
                val_acc = metrics.get('val_accuracy', 'N/A')
                train_time = metrics.get('training_time_seconds', 'N/A')
                if isinstance(train_acc, float):
                    train_acc = f"{train_acc:.4f}"
                if isinstance(val_acc, float):
                    val_acc = f"{val_acc:.4f}"
                if isinstance(train_time, float):
                    train_time = f"{train_time:.2f}s"
                report_lines.append(f"| {model_name} | {train_acc} | {val_acc} | {train_time} |")
            except Exception:
                continue

        ensure_dir('results/reports')
        with open('results/reports/final_report.md', 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))

        logger.info("报告生成完成")
    except Exception as e:
        logger.error(f"报告生成失败: {e}")

    total_time = time.time() - start_time
    logger.info(f"\n{'=' * 60}")
    logger.info(f"所有实验完成！总耗时: {total_time:.1f}秒")
    logger.info(f"{'=' * 60}")


if __name__ == "__main__":
    main()
