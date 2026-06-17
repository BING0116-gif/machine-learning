# Dry Bean Classification

基于 **Dry Bean Dataset** 的多分类机器学习系统，涵盖完整的数据分析、模型训练、评估与可视化流程。

> **课程**: 2026_AIT209 机器学习 (期末大作业)

---

## 目录

- [项目简介](#项目简介)
- [项目结构](#项目结构)
- [快速开始](#快速开始)
- [安装指南](#安装指南)
- [使用说明](#使用说明)
- [CLI 命令参考](#cli-命令参考)
- [配置说明](#配置说明)
- [模型说明](#模型说明)
- [Web 界面 (Streamlit)](#web-界面-streamlit)
- [实验结果](#实验结果)
- [贡献指南](#贡献指南)
- [许可证](#许可证)
- [联系方式](#联系方式)

---

## 项目简介

本项目基于 UCI Dry Bean Dataset，构建了一个完整的多分类机器学习工程系统。系统支持 **4 种主流分类算法 + 2 种集成方法**，并提供 CLI 命令行界面和 Streamlit Web Dashboard 两种交互方式。

### 主要功能

| 功能模块                 | 说明                                               |
| ------------------------ | -------------------------------------------------- |
| **数据分析 (EDA)** | 数据质量评估、分布分析、相关性矩阵、缺失值检测     |
| **数据预处理**     | 缺失值填充、标签纠错、特征标准化、特征工程         |
| **模型训练**       | Random Forest / XGBoost / MLP (PyTorch) / LightGBM |
| **超参数优化**     | 基于 Optuna 的贝叶斯超参数搜索                     |
| **模型集成**       | Voting Ensemble & Stacking Ensemble                |
| **全面评估**       | 精度指标、混淆矩阵、鲁棒性测试、可解释性分析       |
| **可视化**         | EDA图表、训练曲线、精度对比、SHAP/LIME 可解释性    |
| **Web Dashboard**  | 基于 Streamlit 的交互式数据探索与模型对比界面      |

### 数据集信息

| 属性       | 值                                                                                       |
| ---------- | ---------------------------------------------------------------------------------------- |
| 数据集名称 | Dry Bean Dataset                                                                         |
| 来源       | [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Dry+Bean+Dataset) |
| 总样本数   | 13,611                                                                                   |
| 特征数     | 16 (形态学特征)                                                                          |
| 类别数     | 7                                                                                        |
| 类别名称   | SEKER, BARBUNYA, BOMBAY, CALI, HOROZ, SIRA, DERMASON                                     |

### 技术栈

| 类别         | 技术                            |
| ------------ | ------------------------------- |
| 语言         | Python 3.9+                     |
| 传统机器学习 | scikit-learn, XGBoost, LightGBM |
| 深度学习     | PyTorch                         |
| 可视化       | Matplotlib, Seaborn, Plotly     |
| Web 框架     | Streamlit                       |
| CLI 框架     | Click                           |
| 超参数优化   | Optuna                          |
| 可解释性     | SHAP, LIME                      |
| 实验管理     | YAML 配置 + JSON 实验记录       |

---

## 项目结构

```
机器学习期末大作业/
├── config/                          # YAML 配置文件
│   ├── default.yaml                 #   主配置文件
│   ├── data_processing.yaml         #   数据处理配置
│   └── models/                      #   模型参数配置
│       ├── lightgbm.yaml
│       ├── mlp.yaml
│       ├── random_forest.yaml
│       └── xgboost.yaml
├── data/
│   └── processed/                   # 处理后的数据 (.pkl)
├── docs/
│   └── theory.md                    # LightGBM 算法原理详解
├── experiments/
│   └── run_all.py                   # 一键运行所有实验
├── logs/                            # 运行日志
├── notebooks/                       # Jupyter Notebook
├── results/
│   ├── experiments/                 # 实验记录 (.json)
│   ├── figures/
│   │   ├── eda/                     #   EDA 图表
│   │   ├── evaluation/              #   评估图表
│   │   └── training/                #   训练曲线
│   ├── models/                      # 保存的模型文件
│   └── reports/                     # 自动生成的报告
├── src/
│   ├── data/                        # 数据处理模块
│   │   ├── loader.py                #   数据加载器
│   │   ├── cleaner.py               #   数据清洗
│   │   ├── feature_engineering.py   #   特征工程
│   │   └── quality_assessment.py    #   数据质量评估
│   ├── evaluation/                  # 评估模块
│   │   ├── metrics.py               #   评估指标计算
│   │   ├── comparator.py            #   模型对比
│   │   ├── robustness.py            #   鲁棒性测试
│   │   └── interpretability.py      #   可解释性分析
│   ├── models/                      # 模型模块
│   │   ├── base.py                  #   模型基类
│   │   ├── traditional.py           #   Random Forest & XGBoost
│   │   ├── deep_learning.py         #   MLP (PyTorch)
│   │   ├── advanced.py              #   LightGBM
│   │   └── ensemble.py             #   Voting & Stacking
│   ├── training/                    # 训练模块
│   │   ├── trainer.py               #   训练管理器
│   │   └── hyperparameter_tuning.py #   超参数优化
│   ├── utils/                       # 工具模块
│   │   ├── io_utils.py              #   文件读写
│   │   ├── logger.py               #   日志系统
│   │   └── seed.py                  #   随机种子
│   └── visualization/
│       └── plots.py                 # 可视化图表生成
├── tests/                           # 单元测试
├── app.py                           # Streamlit Web Dashboard
├── main.py                          # CLI 入口
├── requirements.txt                 # Python 依赖
├── .env.example                    # 环境变量模板
└── .gitignore
```

---

## 快速开始

```bash
# 1. 创建虚拟环境并激活
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # macOS/Linux

# 2. 安装依赖
pip install -r requirements.txt

# 3. 数据分析
python main.py analyze --data-dir DryBeanDataset

# 4. 数据预处理
python main.py process

# 5. 训练所有模型
python main.py train --model random_forest
python main.py train --model xgboost
python main.py train --model mlp
python main.py train --model lightgbm

# 6. 模型评估
python main.py evaluate

# 7. 生成报告
python main.py report

# 8. 启动 Web 界面
python main.py serve
```

> **一键运行**: `python main.py run-all` 将按顺序执行数据分析 → 数据预处理 → 训练 4 个模型 → 评估 → 生成报告。

---

## 安装指南

### 环境要求

- **Python**: 3.9 或更高版本
- **操作系统**: Windows / macOS / Linux
- **GPU (可选)**: 支持 CUDA 的 NVIDIA GPU (加速 MLP 训练)

### 安装步骤

1. **克隆或下载项目**

   ```bash
   git clone <repository-url>
   cd 机器学习期末大作业
   ```
2. **创建虚拟环境**

   ```bash
   python -m venv venv
   ```
3. **激活虚拟环境**

   ```bash
   # Windows
   venv\Scripts\activate

   # macOS / Linux
   source venv/bin/activate
   ```
4. **安装 Python 依赖**

   ```bash
   pip install -r requirements.txt
   ```
5. **配置环境变量 (可选)**

   ```bash
   cp .env.example .env
   # 编辑 .env 文件，按需修改配置
   ```
6. **准备数据集**

   将 Dry Bean Dataset 的 CSV 文件放入 `DryBeanDataset/` 目录：

   - `Dry_Bean_Dataset_Dirty_train.csv`
   - `Dry_Bean_Dataset_Dirty_val.csv`
   - `Dry_Bean_Dataset_Dirty_test.csv`

### 主要依赖

| 依赖         | 版本要求 | 用途          |
| ------------ | -------- | ------------- |
| numpy        | >=1.24.0 | 数值计算      |
| pandas       | >=2.0.0  | 数据处理      |
| scikit-learn | >=1.3.0  | 传统 ML 算法  |
| xgboost      | >=1.7.0  | XGBoost 模型  |
| lightgbm     | >=4.0.0  | LightGBM 模型 |
| torch        | >=2.0.0  | 深度学习框架  |
| optuna       | >=3.0.0  | 超参数优化    |
| matplotlib   | >=3.7.0  | 图表绘制      |
| seaborn      | >=0.12.0 | 统计可视化    |
| plotly       | >=5.15.0 | 交互式图表    |
| streamlit    | >=1.28.0 | Web 界面      |
| click        | >=8.1.0  | CLI 框架      |
| shap         | >=0.41.0 | SHAP 可解释性 |
| lime         | >=0.2.0  | LIME 可解释性 |

> **注意**: 必须使用 `venv\Scripts\python.exe` (Windows) 或 `venv/bin/python` (macOS/Linux) 执行所有命令，系统 Python 可能存在 numpy/pandas 版本兼容性问题。

---

## 使用说明

### 基本工作流

推荐按以下顺序执行：

```
数据准备 → 数据分析 → 数据预处理 → 模型训练 → 模型评估 → 生成报告
```

#### 1. 数据分析 (EDA)

分析原始数据的质量、分布和特征关系，生成可视化报告。

```bash
python main.py analyze --data-dir DryBeanDataset --output-dir results/figures/eda
```

输出：

- 各数据集的质量报告 (`results/figures/eda/*_quality_report.json`)
- EDA 图表 (分布图、相关性矩阵等)

#### 2. 数据预处理

执行数据清洗（缺失值填充、标签纠错）和特征工程（标准化、特征选择），生成处理后的数据。

```bash
python main.py process --config config/data_processing.yaml --output-dir data/processed
```

输出：

- `data/processed/train_processed.pkl` — 训练集
- `data/processed/val_processed.pkl` — 验证集
- `data/processed/test_processed.pkl` — 测试集
- `data/processed/feature_engineer.pkl` — 特征工程器
- `data/processed/cleaning_reports.json` — 清洗报告

> **注意**: 数据处理只需运行一次，结果会持久化保存到 `data/processed/`。

#### 3. 模型训练

训练指定的机器学习模型。

```bash
# 训练单个模型
python main.py train --model random_forest
python main.py train --model xgboost
python main.py train --model mlp
python main.py train --model lightgbm

# 带超参数优化
python main.py train --model random_forest --hyperparameter-tuning
```

#### 4. 模型评估

对所有已训练模型进行全面评估。

```bash
python main.py evaluate --results-dir results --data-dir data/processed
```

#### 5. 生成报告

汇总所有实验结果，生成 Markdown 格式的最终报告。

```bash
python main.py report --results-dir results --output results/reports/final_report.md
```

#### 6. 一键运行

```bash
python main.py run-all
```

自动执行完整流程，总耗时约 1~3 分钟（含 MLP 训练）。

---

## CLI 命令参考

所有命令均通过 `python main.py` 调用。

### 全局选项

| 选项              | 说明                                              | 默认值 |
| ----------------- | ------------------------------------------------- | ------ |
| `--verbose, -v` | 日志详细程度 (可叠加: 0=WARNING, 1=INFO, 2=DEBUG) | 0      |
| `--seed`        | 随机种子                                          | 42     |

### 子命令

#### `analyze` — 数据分析与 EDA

```bash
python main.py analyze [OPTIONS]
```

| 选项             | 说明         | 默认值                  |
| ---------------- | ------------ | ----------------------- |
| `--data-dir`   | 原始数据目录 | `DryBeanDataset`      |
| `--output-dir` | EDA 输出目录 | `results/figures/eda` |

#### `process` — 数据预处理

```bash
python main.py process [OPTIONS]
```

| 选项             | 说明               | 默认值                          |
| ---------------- | ------------------ | ------------------------------- |
| `--config`     | 数据处理配置文件   | `config/data_processing.yaml` |
| `--data-dir`   | 原始数据目录       | `DryBeanDataset`              |
| `--output-dir` | 处理后数据输出目录 | `data/processed`              |

#### `train` — 模型训练

```bash
python main.py train --model <MODEL> [OPTIONS]
```

| 选项                        | 说明                    | 默认值                  |
| --------------------------- | ----------------------- | ----------------------- |
| `--model`                 | **必需** 模型类型 | 见下方可选值            |
| `--config`                | 配置文件                | `config/default.yaml` |
| `--data-dir`              | 处理后数据目录          | `data/processed`      |
| `--output-dir`            | 结果输出目录            | `results`             |
| `--hyperparameter-tuning` | 启用超参数优化          | `False`               |

`--model` 可选值:

| 值                | 模型                 |
| ----------------- | -------------------- |
| `random_forest` | 随机森林             |
| `xgboost`       | XGBoost              |
| `mlp`           | 多层感知机 (PyTorch) |
| `lightgbm`      | LightGBM             |
| `voting`        | 投票集成             |
| `stacking`      | 堆叠集成             |

#### `evaluate` — 模型评估

```bash
python main.py evaluate [OPTIONS]
```

| 选项              | 说明             | 默认值                         |
| ----------------- | ---------------- | ------------------------------ |
| `--results-dir` | 实验结果目录     | `results`                    |
| `--data-dir`    | 处理后数据目录   | `data/processed`             |
| `--output-dir`  | 评估图表输出目录 | `results/figures/evaluation` |

#### `serve` — 启动 Web 界面

```bash
python main.py serve [OPTIONS]
```

| 选项       | 说明         | 默认值    |
| ---------- | ------------ | --------- |
| `--port` | Web 服务端口 | 8501      |
| `--host` | Web 服务主机 | localhost |

#### `report` — 生成实验报告

```bash
python main.py report [OPTIONS]
```

| 选项              | 说明         | 默认值                              |
| ----------------- | ------------ | ----------------------------------- |
| `--results-dir` | 实验结果目录 | `results`                         |
| `--output`      | 报告输出路径 | `results/reports/final_report.md` |

#### `run-all` — 一键运行完整流程

```bash
python main.py run-all [OPTIONS]
```

按顺序执行：`analyze` → `process` → `train`(4个模型) → `evaluate` → `report`

---

## 配置说明

项目使用 YAML 格式的配置文件，位于 `config/` 目录。

### 主配置文件 (`config/default.yaml`)

```yaml
project:
  name: "Dry Bean Classification"
  version: "1.0.0"
  random_seed: 42

data:
  raw_dir: "DryBeanDataset"
  processed_dir: "data/processed"

# 数据清洗策略
data_cleaning:
  missing_value_strategy: "median"   # 缺失值填充: median/mean/mode/knn
  outlier_method: "iqr"              # 异常值检测: iqr/zscore
  label_correction: true             # 标签纠错
  iqr_factor: 1.5
  zscore_threshold: 3

# 特征工程
feature_engineering:
  scaling: "standard"                # 标准化: standard/minmax
  feature_selection: "importance"    # 特征选择: importance/correlation
  correlation_threshold: 0.95

# 模型超参数 (详见 config/models/*.yaml)
models:
  random_forest: { ... }
  xgboost: { ... }
  mlp: { ... }
  lightgbm: { ... }

training:
  early_stopping_patience: 15

evaluation:
  metrics: ["accuracy", "precision", "recall", "f1"]

logging:
  level: "INFO"
  file: "logs/app.log"
```

### 数据处理配置 (`config/data_processing.yaml`)

独立管理数据清洗和特征工程的参数，与主配置分离。

### 模型配置 (`config/models/`)

每个模型有独立的参数配置文件：

- `random_forest.yaml` — 随机森林参数
- `xgboost.yaml` — XGBoost 参数
- `mlp.yaml` — MLP 网络架构和训练参数
- `lightgbm.yaml` — LightGBM 参数

### 环境变量 (`.env`)

| 变量            | 说明     | 默认值             |
| --------------- | -------- | ------------------ |
| `DATA_DIR`    | 数据目录 | `DryBeanDataset` |
| `LOG_LEVEL`   | 日志级别 | `INFO`           |
| `RANDOM_SEED` | 随机种子 | `42`             |

> 复制 `.env.example` 为 `.env` 并按需修改。

---

## 模型说明

### 实现的模型

| 模型                        | 类型     | 特点                                                                                   |
| --------------------------- | -------- | -------------------------------------------------------------------------------------- |
| **Random Forest**     | 集成学习 | 基于 Bagging 的决策树集成，对噪声鲁棒，训练快                                          |
| **XGBoost**           | 梯度提升 | 基于 Boosting 的决策树集成，高精度，支持正则化                                         |
| **MLP (PyTorch)**     | 深度学习 | 3 层全连接网络 (256→128→64)，BatchNorm + Dropout + Cosine Annealing + Early Stopping |
| **LightGBM**          | 梯度提升 | 基于 Leaf-wise 生长策略，训练速度极快，内存效率高 (详见[docs/theory.md](docs/theory.md))  |
| **Voting Ensemble**   | 集成方法 | 软投票/硬投票组合多个基模型                                                            |
| **Stacking Ensemble** | 集成方法 | 使用 LogisticRegression 作为元学习器                                                   |

### 超参数优化

使用 Optuna 进行贝叶斯超参数搜索：

```bash
python main.py train --model random_forest --hyperparameter-tuning
```

搜索空间：

- **Random Forest**: n_estimators(50-300), max_depth(5-20), min_samples_split(2-10)
- **XGBoost**: n_estimators(50-300), max_depth(3-10), learning_rate(0.01-0.3), subsample(0.6-1.0)
- **MLP**: hidden_dims(64-512), dropout_rate(0.1-0.5), learning_rate(1e-4-1e-2), batch_size(16/32/64)
- **LightGBM**: num_leaves(15-63), max_depth(3-15), learning_rate(0.01-0.3)

### 评估指标

- **Accuracy** — 总体准确率
- **Balanced Accuracy** — 平衡准确率
- **Precision / Recall / F1-Score** — Macro & Weighted 平均
- **Confusion Matrix** — 混淆矩阵
- **每类指标** — 每个类别的 Precision / Recall / F1

---

## Web 界面 (Streamlit)

项目提供基于 Streamlit 的交互式 Web Dashboard。

```bash
# 方式一: CLI 命令
python main.py serve --port 8501

# 方式二: 直接启动
streamlit run app.py --server.port 8501
```

启动后访问 `http://localhost:8501`。

### Dashboard 页面

| 页面               | 功能                                                       |
| ------------------ | ---------------------------------------------------------- |
| **首页**     | 关键指标卡片、模型性能概览、快速开始指引                   |
| **数据探索** | 数据预览、类别分布、特征分布直方图、相关性矩阵、缺失值分析 |
| **模型对比** | 精度对比表 & 柱状图、训练曲线 (Loss & Accuracy)            |
| **可解释性** | 特征重要性排名、SHAP 分析图表                              |
| **关于项目** | 项目介绍、技术栈、数据集信息                               |

---

## 实验结果

训练完成后，自动生成实验报告 (`results/reports/final_report.md`)。以下是某次实际运行结果：

| 模型          | 训练准确率 | 验证准确率 | 训练时间 |
| ------------- | ---------- | ---------- | -------- |
| lightgbm      | 0.9708     | 0.8931     | 0.83s    |
| pytorch_mlp   | 0.9067     | 0.9228     | 24.04s   |
| random_forest | 0.9468     | 0.8842     | 0.34s    |
| xgboost       | 0.9821     | 0.9258     | 0.71s    |

> 验证集最佳模型: **XGBoost** (Accuracy: 0.9258)

---

## 贡献指南

### 分支策略

- `main` — 稳定版本
- `dev` — 开发分支
- `feature/<name>` — 新功能开发
- `fix/<name>` — Bug 修复

### 提交规范

```
<type>: <description>

类型:
  feat     新功能
  fix      Bug 修复
  docs     文档更新
  refactor 代码重构
  test     测试相关
  config   配置变更
```

示例: `feat: add LightGBM model with GOSS and EFB`

### Pull Request 流程

1. Fork 项目并创建功能分支
2. 编写代码并添加测试
3. 确保现有测试通过: `pytest tests/`
4. 提交 PR 并描述变更内容
5. 等待 Code Review

### 代码风格

- 遵循 PEP 8 规范
- 类型注解: 函数参数和返回值使用 typing
- 文档字符串: Google Style Docstring
- 日志: 使用项目内置的 `src.utils.logger`

---

## 联系方式

- **数据集**: [Dry Bean Dataset - UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Dry+Bean+Dataset)
- **项目结构参考**: 机器学习工程项目最佳实践
