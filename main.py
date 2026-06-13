"""Dry Bean Classification - 机器学习工程项目 CLI 入口

用法:
    python main.py --help
    python main.py analyze --data-dir ./DryBeanDataset --output-dir ./results/figures/eda
    python main.py process --config config/data_processing.yaml --output-dir ./data/processed
    python main.py train --model random_forest --config config/default.yaml
    python main.py run-all --config config/default.yaml
    python main.py evaluate --results-dir ./results
    python main.py serve --port 8501
    python main.py report --results-dir ./results
"""

import os
import sys

import click
import pandas as pd


@click.group()
@click.option("--verbose", "-v", count=True, help="增加日志详细程度 (0=WARNING, 1=INFO, 2=DEBUG)")
@click.option("--seed", default=42, type=int, help="随机种子")
@click.pass_context
def cli(ctx, verbose, seed):
    """Dry Bean Classification - 机器学习工程项目

    基于Dry Bean Dataset的多分类机器学习系统，
    包含完整的数据分析、模型训练、评估和可视化功能。
    """
    ctx.ensure_object(dict)

    # 设置日志级别
    log_levels = {0: "WARNING", 1: "INFO", 2: "DEBUG"}
    ctx.obj["log_level"] = log_levels.get(verbose, "DEBUG")
    ctx.obj["seed"] = seed

    # 设置随机种子
    from src.utils.seed import set_seed
    set_seed(seed)

    # 配置日志
    from src.utils.logger import setup_logger
    setup_logger(
        name="dry_bean",
        level=ctx.obj["log_level"],
        log_file="logs/app.log",
        console=True,
    )


@cli.command()
@click.option("--data-dir", type=click.Path(exists=True), default="DryBeanDataset", help="原始数据目录")
@click.option("--output-dir", type=click.Path(), default="results/figures/eda", help="输出目录")
@click.pass_context
def analyze(ctx, data_dir, output_dir):
    """运行数据分析和EDA可视化

    对原始数据进行探索性分析，生成数据质量报告和可视化图表。
    """
    from src.utils.logger import get_logger
    logger = get_logger()

    logger.info(f"开始数据分析，数据目录: {data_dir}")

    from src.data.loader import DataLoader
    from src.data.quality_assessment import QualityAssessor
    from src.visualization.plots import PlotGenerator

    # 加载数据
    loader = DataLoader(data_dir)
    data = loader.load_all()

    # 数据质量分析
    assessor = QualityAssessor()
    for split_name, df in data.items():
        logger.info(f"分析 {split_name} 数据集 ({df.shape})")
        report = assessor.assess(df, loader.FEATURE_COLUMNS, loader.TARGET_COLUMN)

        from src.utils.io_utils import save_json, ensure_dir
        ensure_dir(output_dir)
        save_json(report, os.path.join(output_dir, f"{split_name}_quality_report.json"))

    # 生成EDA图表
    plotter = PlotGenerator()
    all_data = pd.concat([data["train"], data["val"], data["test"]], ignore_index=True)
    plotter.generate_eda_plots(all_data, loader.FEATURE_COLUMNS, loader.TARGET_COLUMN, output_dir)

    logger.info(f"数据分析完成，结果保存到: {output_dir}")
    click.echo(f"✅ 数据分析完成！报告和图表已保存到: {output_dir}")


@cli.command()
@click.option("--config", type=click.Path(exists=True), default="config/data_processing.yaml", help="数据处理配置文件")
@click.option("--data-dir", type=click.Path(exists=True), default="DryBeanDataset", help="原始数据目录")
@click.option("--output-dir", type=click.Path(), default="data/processed", help="处理后数据输出目录")
@click.pass_context
def process(ctx, config, data_dir, output_dir):
    """数据预处理

    执行数据清洗和特征工程，生成处理后的数据文件。
    """
    from src.utils.logger import get_logger
    logger = get_logger()

    logger.info("开始数据预处理")

    from src.data.loader import DataLoader
    from src.data.cleaner import DataCleaner
    from src.data.feature_engineering import FeatureEngineer
    from src.utils.io_utils import load_config, ensure_dir, save_json
    import joblib

    # 加载配置
    config_data = load_config(config) if os.path.exists(config) else {}
    cleaning_config = config_data.get("data_cleaning", {})
    fe_config = config_data.get("feature_engineering", {})

    # 加载数据
    loader = DataLoader(data_dir)
    data = loader.load_all()

    # 数据清洗
    cleaner = DataCleaner(strategy_config=cleaning_config)
    cleaned_data = {}
    cleaning_reports = {}
    for split_name, df in data.items():
        df_clean, report = cleaner.clean(df, loader.FEATURE_COLUMNS, loader.TARGET_COLUMN)
        cleaned_data[split_name] = df_clean
        cleaning_reports[split_name] = report
        logger.info(f"{split_name}: {df.shape} -> {df_clean.shape}")

    # 特征工程
    engineer = FeatureEngineer(config=fe_config)

    # 分离特征和标签
    X_train = cleaned_data["train"][loader.FEATURE_COLUMNS]
    y_train = cleaned_data["train"][loader.TARGET_COLUMN]
    X_val = cleaned_data["val"][loader.FEATURE_COLUMNS]
    y_val = cleaned_data["val"][loader.TARGET_COLUMN]
    X_test = cleaned_data["test"][loader.FEATURE_COLUMNS]
    y_test = cleaned_data["test"][loader.TARGET_COLUMN]

    # 拟合和转换
    X_train_processed, X_val_processed, X_test_processed, y_train_encoded, y_val_encoded, y_test_encoded = engineer.fit_transform(
        X_train, X_val, X_test, y_train, y_val, y_test
    )

    # 保存处理后的数据
    ensure_dir(output_dir)
    processed = {
        "train": (X_train_processed, y_train_encoded),
        "val": (X_val_processed, y_val_encoded),
        "test": (X_test_processed, y_test_encoded),
    }

    for split_name, (X, y) in processed.items():
        joblib.dump({"X": X, "y": y}, os.path.join(output_dir, f"{split_name}_processed.pkl"))

    # 保存特征工程器
    engineer.save(os.path.join(output_dir, "feature_engineer.pkl"))

    # 保存清洗报告
    save_json(cleaning_reports, os.path.join(output_dir, "cleaning_reports.json"))

    logger.info(f"数据预处理完成，结果保存到: {output_dir}")
    click.echo(f"✅ 数据预处理完成！处理后数据已保存到: {output_dir}")


@cli.command()
@click.option("--model", type=click.Choice(["random_forest", "xgboost", "mlp", "lightgbm", "voting", "stacking"]), required=True, help="模型类型")
@click.option("--config", type=click.Path(exists=True), default="config/default.yaml", help="配置文件")
@click.option("--data-dir", type=click.Path(), default="data/processed", help="处理后数据目录")
@click.option("--output-dir", type=click.Path(), default="results", help="输出目录")
@click.option("--hyperparameter-tuning/--no-hyperparameter-tuning", default=False, help="是否进行超参数优化")
@click.pass_context
def train(ctx, model, config, data_dir, output_dir, hyperparameter_tuning):
    """训练指定模型

    使用处理后的数据训练指定的机器学习模型。
    """
    from src.utils.logger import get_logger
    logger = get_logger()

    logger.info(f"开始训练模型: {model}")

    import joblib
    from src.utils.io_utils import load_config, ensure_dir

    # 加载配置
    config_data = load_config(config)

    # 加载处理后的数据
    train_data = joblib.load(os.path.join(data_dir, "train_processed.pkl"))
    val_data = joblib.load(os.path.join(data_dir, "val_processed.pkl"))

    X_train, y_train = train_data["X"], train_data["y"]
    X_val, y_val = val_data["X"], val_data["y"]

    # 创建模型
    model_instance = _create_model(model, config_data)

    # 超参数优化
    if hyperparameter_tuning:
        logger.info("开始超参数优化...")
        from src.training.hyperparameter_tuning import HyperparameterTuner
        tuner = HyperparameterTuner(model_name=model, config=config_data)
        best_params = tuner.optimize(X_train, y_train, X_val, y_val)
        logger.info(f"最佳参数: {best_params}")
        model_instance = _create_model(model, config_data, override_params=best_params)

    # 训练
    from src.training.trainer import Trainer
    trainer = Trainer(output_dir=output_dir)
    result = trainer.train_model(
        model_instance, X_train, y_train, X_val, y_val,
        experiment_name=model
    )

    logger.info(f"模型训练完成: {model}")
    click.echo(f"✅ 模型训练完成！")
    click.echo(f"   训练准确率: {result['metrics'].get('train_accuracy', 'N/A')}")
    click.echo(f"   验证准确率: {result['metrics'].get('val_accuracy', 'N/A')}")


@cli.command()
@click.option("--config", type=click.Path(exists=True), default="config/default.yaml", help="配置文件")
@click.option("--data-dir", type=click.Path(), default="data/processed", help="处理后数据目录")
@click.option("--output-dir", type=click.Path(), default="results", help="输出目录")
@click.pass_context
def run_all(ctx, config, data_dir, output_dir):
    """运行所有实验

    按顺序执行：数据分析 → 数据处理 → 模型训练 → 模型评估 → 生成报告。
    """
    from src.utils.logger import get_logger
    logger = get_logger()

    logger.info("=" * 60)
    logger.info("开始运行所有实验")
    logger.info("=" * 60)

    import time
    start_time = time.time()

    # Step 1: 数据分析
    logger.info("\n[Step 1/5] 数据分析...")
    ctx.invoke(analyze, data_dir="DryBeanDataset", output_dir="results/figures/eda")

    # Step 2: 数据处理
    logger.info("\n[Step 2/5] 数据预处理...")
    ctx.invoke(process, config=config, data_dir="DryBeanDataset", output_dir=data_dir)

    # Step 3: 训练所有模型
    logger.info("\n[Step 3/5] 训练所有模型...")
    for model_name in ["random_forest", "xgboost", "mlp", "lightgbm"]:
        logger.info(f"\n训练 {model_name}...")
        ctx.invoke(train, model=model_name, config=config, data_dir=data_dir, output_dir=output_dir, hyperparameter_tuning=False)

    # Step 4: 模型评估
    logger.info("\n[Step 4/5] 模型评估...")
    ctx.invoke(evaluate, results_dir=output_dir, data_dir=data_dir)

    # Step 5: 生成报告
    logger.info("\n[Step 5/5] 生成报告...")
    ctx.invoke(report, results_dir=output_dir)

    total_time = time.time() - start_time
    logger.info(f"\n{'=' * 60}")
    logger.info(f"所有实验完成！总耗时: {total_time:.1f}秒")
    logger.info(f"{'=' * 60}")
    click.echo(f"\n✅ 所有实验完成！总耗时: {total_time:.1f}秒")


@cli.command()
@click.option("--results-dir", type=click.Path(), default="results", help="实验结果目录")
@click.option("--data-dir", type=click.Path(), default="data/processed", help="处理后数据目录")
@click.option("--output-dir", type=click.Path(), default="results/figures/evaluation", help="评估图表输出目录")
@click.pass_context
def evaluate(ctx, results_dir, data_dir, output_dir):
    """评估已训练的模型

    对所有已训练的模型进行全面评估，生成对比报告和图表。
    """
    from src.utils.logger import get_logger
    logger = get_logger()

    logger.info("开始模型评估")

    import joblib
    from src.utils.io_utils import ensure_dir
    from src.evaluation.metrics import MetricsCalculator
    from src.evaluation.comparator import ModelComparator
    from src.visualization.plots import PlotGenerator

    ensure_dir(output_dir)

    # 加载测试数据
    test_data = joblib.load(os.path.join(data_dir, "test_processed.pkl"))
    X_test, y_test = test_data["X"], test_data["y"]

    # 评估所有模型
    comparator = ModelComparator(results_dir=results_dir)
    comparison_results = comparator.compare_all(X_test, y_test)

    # 生成图表
    plotter = PlotGenerator()
    plotter.plot_accuracy_comparison(comparison_results, save_path=os.path.join(output_dir, "accuracy_comparison.png"))

    logger.info("模型评估完成")
    click.echo(f"✅ 模型评估完成！结果保存到: {output_dir}")


@cli.command()
@click.option("--port", default=8501, type=int, help="Web服务端口")
@click.option("--host", default="localhost", help="Web服务主机")
@click.pass_context
def serve(ctx, port, host):
    """启动Streamlit Web界面

    启动交互式Dashboard，展示数据分析、模型对比和可解释性分析结果。
    """
    import subprocess

    click.echo(f"启动Streamlit Web界面: http://{host}:{port}")
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", "app.py",
        "--server.port", str(port),
        "--server.headless", "true",
    ])


@cli.command()
@click.option("--results-dir", type=click.Path(), default="results", help="实验结果目录")
@click.option("--output", type=click.Path(), default="results/reports/final_report.md", help="报告输出路径")
@click.pass_context
def report(ctx, results_dir, output):
    """生成实验报告

    汇总所有实验结果，生成Markdown格式的最终报告。
    """
    from src.utils.logger import get_logger
    logger = get_logger()

    logger.info("开始生成实验报告")

    from src.utils.io_utils import ensure_dir
    ensure_dir(os.path.dirname(output))

    # 收集实验结果
    experiments_dir = os.path.join(results_dir, "experiments")
    if not os.path.exists(experiments_dir):
        click.echo("⚠️ 未找到实验结果，请先运行模型训练")
        return

    # 生成报告
    report_lines = [
        "# Dry Bean Classification 实验报告\n",
        "## 模型性能对比\n",
        "| 模型 | 训练准确率 | 验证准确率 | 训练时间 |",
        "|------|-----------|-----------|---------|",
    ]

    import glob
    for exp_file in sorted(glob.glob(os.path.join(experiments_dir, "*.json"))):
        from src.utils.io_utils import load_json
        try:
            exp_data = load_json(exp_file)
            model_name = exp_data.get("model_name", "Unknown")
            metrics = exp_data.get("metrics", {})
            train_acc = metrics.get("train_accuracy", "N/A")
            val_acc = metrics.get("val_accuracy", "N/A")
            train_time = metrics.get("training_time_seconds", "N/A")
            if isinstance(train_acc, float):
                train_acc = f"{train_acc:.4f}"
            if isinstance(val_acc, float):
                val_acc = f"{val_acc:.4f}"
            if isinstance(train_time, float):
                train_time = f"{train_time:.2f}s"
            report_lines.append(f"| {model_name} | {train_acc} | {val_acc} | {train_time} |")
        except Exception as e:
            logger.warning(f"无法读取实验文件 {exp_file}: {e}")

    report_content = "\n".join(report_lines)

    with open(output, "w", encoding="utf-8") as f:
        f.write(report_content)

    logger.info(f"实验报告已生成: {output}")
    click.echo(f"✅ 实验报告已生成: {output}")


def _create_model(model_name: str, config: dict, override_params: dict = None):
    """创建模型实例的工厂函数"""
    model_config = config.get("models", {}).get(model_name, {})
    if override_params:
        model_config.update(override_params)

    if model_name == "random_forest":
        from src.models.traditional import RFModel
        return RFModel(hyperparams=model_config)
    elif model_name == "xgboost":
        from src.models.traditional import XGBoostModel
        return XGBoostModel(hyperparams=model_config)
    elif model_name == "mlp":
        from src.models.deep_learning import PyTorchModel
        return PyTorchModel(model_type="mlp", config=model_config)
    elif model_name == "lightgbm":
        from src.models.advanced import LightGBMModel
        return LightGBMModel(hyperparams=model_config)
    elif model_name == "voting":
        from src.models.ensemble import VotingEnsemble
        return VotingEnsemble(config=model_config)
    elif model_name == "stacking":
        from src.models.ensemble import StackingEnsemble
        return StackingEnsemble(config=model_config)
    else:
        raise ValueError(f"Unknown model: {model_name}")


if __name__ == "__main__":
    cli()
