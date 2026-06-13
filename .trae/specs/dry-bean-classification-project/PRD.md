# Dry Bean Dataset 多分类机器学习工程项目 PRD

**文档版本**: v1.0  
**创建日期**: 2026-06-10  
**项目代号**: dry-bean-classification  
**课程信息**: 2026_AIT209 机器学习期末大作业  
**截止日期**: 2026-06-28（第16周周末）  
**文档所有者**: 学生  
**数据分类**: 学术研究数据（公开数据集）

---

## 文档控制

| 版本 | 日期 | 作者 | 变更说明 |
|------|------|------|----------|
| v1.0 | 2026-06-10 | 学生 | 初始版本，完整需求规格 |

### 审批流程
- [x] 需求确认完成（通过 grill-master 交互式访谈）
- [ ] 技术方案评审
- [ ] 实施计划确认
- [ ] 最终验收

---

## 执行摘要

### 项目背景
本项目是 2026_AIT209 机器学习课程的期末大作业，要求基于 **Dry Bean Dataset**（带噪声的脏数据）构建一个完整的机器学习工程项目，实现从数据分析到模型部署的全流程。

### 核心目标
构建一个**专业级、可展示、可复现**的多分类机器学习系统，包含：
1. 完整的数据分析和处理流程
2. 至少3种多分类算法的对比实验（含1种课堂未讲过的算法）
3. 全面的模型评估（精度、速度、鲁棒性、过拟合分析）
4. 工程化的代码架构和 Web 可视化界面
5. GitHub 专业级展示和详细文档

### 成功标准
- ✅ 满足课程所有评分要求（总分100%）
- ✅ 获得额外加分（现场讲解 + 高级特性）
- ✅ 代码质量达到生产级别（可维护、可扩展）
- ✅ 完整的文档和实验报告

---

## 问题陈述

### 业务问题
作为机器学习课程的学生，需要展示对机器学习全流程的掌握程度，包括：
- 数据理解和预处理能力
- 多算法实现和对比能力
- 工程化开发和文档撰写能力
- 理论知识和实践结合能力

### 当前痛点
- 缺乏系统的项目组织和实施计划
- 不确定如何选择最适合的算法组合
- 需要平衡课程要求和高级特性的实现
- 时间管理需要合理规划（18天工期）

### 期望成果
- **高质量的项目代码**：模块化、可维护、符合工程规范
- **全面的实验分析**：多维度对比，有深度的洞察
- **专业的展示效果**：Web界面 + GitHub + 论文图表
- **理论深度**：数学推导、算法原理、可解释性分析

---

## 范围界定

### ✅ 项目范围内（In Scope）

#### 核心功能模块
1. **数据分析模块**
   - 数据集基本统计描述
   - 数据质量评估（缺失值、异常值、标签错误）
   - 探索性数据分析（EDA）可视化
   - 类别分布分析

2. **数据处理模块**
   - 数据清洗策略（缺失值填充、异常值处理、标签纠错）
   - 特征工程（标准化/归一化、特征选择、特征构造）
   - 数据分割验证（Train/Val/Test）
   - 数据增强（可选）

3. **多算法实验模块**
   - **算法A**：传统机器学习算法（如随机森林/XGBoost）
   - **算法B**：深度学习算法（PyTorch实现的MLP/CNN）
   - **算法C**：进阶算法（如LightGBM/CatBoost/SVM+核技巧）
   - **可选算法D/E**：用于集成学习的基础模型

4. **模型评估与分析**
   - 测试集精度对比（Accuracy, Precision, Recall, F1）
   - Loss曲线可视化
   - 推理速度基准测试
   - 鲁棒性测试（高斯噪声、缺失值噪声、标签噪声）
   - 过拟合分析（训练/验证精度差异）
   - 可解释性分析（SHAP/LIME/Feature Importance）

5. **工程架构与部署**
   - 模块化代码结构（MVC/分层架构）
   - 统一命令行接口（CLI）
   - Web交互式展示界面（Streamlit）
   - GitHub专业级仓库（README/Badge/自动化脚本）

6. **文档与总结**
   - 技术文档（README、API文档、使用指南）
   - 实验报告（论文素材、图表、表格）
   - 课程总结（学习内容、评价建议）

#### 高级特性（额外加分）
- ✅ 超参数自动优化（Optuna/GridSearch）
- ✅ 模型集成技术（Stacking/Voting/Blending）
- ✅ 理论深度补充（数学推导、算法原理）
- ✅ 可解释性分析（SHAP值可视化）

### ❌ 项目范围外（Out of Scope）
- ❌ 生产环境部署（Docker/Kubernetes）
- ❌ 实时在线推理服务
- ❌ 大规模分布式训练
- ❌ 移动端或嵌入式部署
- ❌ 商业化应用场景
- ❌ 与其他系统集成（数据库、API等）

### ⚠️ 明确的非目标（Non-Goals）
- 不追求SOTA（State-of-the-Art）性能，重点是方法论的完整性
- 不实现过于复杂的深度学习架构（如Transformer），保持可解释性
- 不过度优化推理性能，以功能完整性为主
- 不追求100%测试覆盖率，重点覆盖核心逻辑

---

## 用户画像和使用场景

### 主要用户

#### 👨‍🎓 User #1: 学生（项目开发者）
**角色**: 机器学习课程学生  
**目标**: 
- 完成高质量的期末作业
- 展示机器学习全流程能力
- 获得高分和额外加分

**使用场景**:
1. **开发阶段**: 使用 CLI 运行单个算法进行调试
2. **实验阶段**: 一键运行所有算法并生成对比报告
3. **展示阶段**: 启动 Web 界面向老师演示
4. **提交阶段**: 整理代码和文档到 GitHub

**痛点**:
- 需要清晰的代码结构和注释
- 需要详细的运行日志和错误提示
- 需要快速定位问题和修复bug

---

#### 👨‍🏫 User #2: 教师（评审者）
**角色**: 课程教师/助教  
**目标**:
- 评估学生的机器学习能力
- 检查代码质量和工程规范
- 理解实验设计和结果分析

**使用场景**:
1. **代码审查**: 通过 GitHub 查看 README 和代码结构
2. **现场演示**: 查看 Web 界面的交互展示
3. **论文评审**: 阅读实验报告和图表
4. **成绩评定**: 根据评分标准打分

**期望**:
- 清晰的文档和说明
- 规范的代码风格
- 完整的实验记录
- 可复现的实验结果

---

## 功能需求

### FR-001: 数据加载与管理

**优先级**: Must Have (P0)  
**描述**: 提供统一的数据加载接口，支持 Dry Bean Dataset 的三个子集（train/val/test）

**需求详情**:
```python
# 期望接口
class DataLoader:
    def load_train() -> pd.DataFrame
    def load_val() -> pd.DataFrame
    def load_test() -> pd.DataFrame
    def load_all() -> Dict[str, pd.DataFrame]
    def get_data_info() -> DataInfo  # 返回数据集元信息
```

**验收标准**:
- [ ] 能正确读取3个CSV文件
- [ ] 自动识别特征列和标签列
- [ ] 处理编码问题（UTF-8/GBK）
- [ ] 返回数据的基本统计信息（形状、类型、缺失值数量）

**测试用例**:
- Given 有效的CSV文件路径 When 加载数据 Then 返回正确的DataFrame
- Given 文件不存在 When 加载数据 Then 抛出明确的异常
- Given 包含空行的CSV When 加载数据 Then 自动跳过空行

---

### FR-002: 数据质量分析

**优先级**: Must Have (P0)  
**描述**: 对输入数据进行全面的质量评估，识别数据污染情况

**功能点**:
1. **缺失值分析**
   - 统计每列的缺失值数量和比例
   - 可视化缺失值分布热力图
   - 判断缺失机制（MCAR/MAR/MNAR）

2. **异常值检测**
   - 基于统计方法（IQR/Z-Score）
   - 基于可视化方法（箱线图/散点图）
   - 记录异常值的位置和数量

3. **标签一致性检查**
   - 检测标签拼写错误（如 D3RMAS0N → DERMASON）
   - 统计每个类别的样本数量
   - 检测类别不平衡问题

4. **数据类型验证**
   - 检查数值型特征是否为有效数字
   - 检查是否有非预期数据类型

**输出产物**:
- 数据质量报告（JSON/Markdown格式）
- 可视化图表（保存为PNG/PDF）

**验收标准**:
- [ ] 自动发现第9行的Solidity缺失值
- [ ] 自动识别第9行的标签错误（D3RMAS0N）
- [ ] 生成完整的数据质量报告
- [ ] 所有图表清晰可读

---

### FR-003: 数据清洗与预处理

**优先级**: Must Have (P0)  
**描述**: 实现数据清洗和特征工程的完整流程

**数据处理流水线**:

```
原始数据 → 缺失值处理 → 异常值处理 → 标签纠错 → 特征缩放 → 特征选择 → 清洁数据
```

**具体实现**:

1. **缺失值处理策略**
   ```python
   # 对于数值特征：
   - 中位数填充（对异常值不敏感）
   - KNN填充（考虑相似样本）
   
   # 决策依据：
   - 如果缺失率 < 5% → 填充
   - 如果缺失率 > 30% → 删除该特征
   ```

2. **异常值处理策略**
   ```python
   # IQR 方法：
   Q1 = df[col].quantile(0.25)
   Q3 = df[col].quantile(0.75)
   IQR = Q3 - Q1
   lower_bound = Q1 - 1.5 * IQR
   upper_bound = Q3 + 1.5 * IQR
   
   # 处理方式：
   - Winsorization（截断到边界值）
   - 或删除极端异常值
   ```

3. **标签纠错**
   ```python
   # 已知错误映射：
   label_corrections = {
       "D3RMAS0N": "DERMASON",
       # 其他可能的错误...
   }
   ```

4. **特征工程**
   ```python
   # 特征标准化：
   - StandardScaler（适用于正态分布特征）
   - MinMaxScaler（适用于有界特征）
   
   # 特征选择：
   - 相关性过滤（去除高度相关特征）
   - 重要性排序（基于树模型）
   - 递归特征消除（RFE）
   ```

**配置化设计**:
```yaml
# config/data_processing.yaml
missing_value_strategy: median  # median/knn/drop
outlier_method: iqr            # iqr/zscore/winsorize
feature_scaling: standard      # standard/minmax/robust
feature_selection: importance  # importance/correlation/rfe
```

**验收标准**:
- [ ] 处理后数据无缺失值
- [ ] 异常值在合理范围内
- [ ] 标签全部正确（7个类别）
- [ ] 特征数值范围合理（适合模型输入）
- [ ] 支持配置文件调整参数

---

### FR-004: 传统机器学习模型

**优先级**: Must Have (P0)  
**描述**: 实现至少1种传统机器学习多分类算法

**推荐算法选择**:

| 算法 | 类型 | 优点 | 适用场景 |
|------|------|------|----------|
| **Random Forest** | 集成学习 | 抗过拟合、可解释性好 | 表格数据首选 |
| **XGBoost** | 梯度提升 | 高精度、速度快 | 竞赛常用 |
| **SVM + RBF核** | 核方法 | 理论优雅、小样本表现好 | 高维数据 |

**实现要求**:
```python
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC

class TraditionalMLModel:
    def __init__(self, model_name: str, hyperparams: dict):
        """
        Args:
            model_name: 'random_forest' | 'xgboost' | 'svm'
            hyperparams: 超参数字典
        """
        
    def train(self, X_train, y_train, X_val, y_val):
        """训练模型，返回训练历史"""
        
    def predict(self, X_test):
        """预测，返回概率和类别"""
        
    def evaluate(self, y_true, y_pred):
        """计算评估指标"""
        
    def save_model(self, path: str):
        """保存模型到文件"""
        
    @staticmethod
    def load_model(path: str):
        """从文件加载模型"""
```

**默认超参数**:
```python
RANDOM_FOREST_PARAMS = {
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 5,
    'random_state': 42,
    'n_jobs': -1
}

XGBOOST_PARAMS = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'random_state': 42,
    'use_label_encoder': False,
    'eval_metric': 'mlogloss'
}

SVM_PARAMS = {
    'C': 1.0,
    'kernel': 'rbf',
    'gamma': 'scale',
    'decision_function_shape': 'ovr'
}
```

**验收标准**:
- [ ] 模型能成功训练和预测
- [ ] 测试集准确率 > 90%（基线目标）
- [ ] 支持超参数自定义
- [ ] 支持模型保存和加载
- [ ] 训练过程有日志输出

---

### FR-005: 深度学习模型

**优先级**: Must Have (P0)  
**描述**: 使用 PyTorch 实现至少1种深度学习多分类算法

**推荐架构**:

#### 方案A: 多层感知机（MLP）
```python
import torch.nn as nn

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dims=[256, 128, 64], dropout_rate=0.3):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)
```

**训练配置**:
```yaml
# config/training.yaml
deep_learning:
  model: mlp  # mlp/cnn
  hidden_dims: [256, 128, 64]
  dropout_rate: 0.3
  
training:
  epochs: 100
  batch_size: 32
  learning_rate: 0.001
  optimizer: adam
  scheduler: cosine_annealing
  early_stopping_patience: 10
  
device: auto  # auto/cpu/cuda
```

**训练循环**:
```python
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += batch_y.size(0)
        correct += predicted.eq(batch_y).sum().item()
    
    accuracy = correct / total
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss, accuracy
```

**验收标准**:
- [ ] 模型能在 GPU/CPU 上正常训练
- [ ] Loss 曲线收敛（无明显震荡）
- [ ] 测试集准确率 > 92%（高于传统ML）
- [ ] 支持早停（Early Stopping）
- [ ] 支持学习率调度
- [ ] 训练日志完整（loss/accuracy/时间）

---

### FR-006: 进阶算法（课堂未讲过的）

**优先级**: Must Have (P0)  
**描述**: 实现1种课堂上未讲过的进阶算法

**推荐选项**:

#### 选项1: LightGBM（微软出品）
```python
import lightgbm as lgb

class LightGBMModel:
    def __init__(self, params=None):
        default_params = {
            'objective': 'multiclass',
            'num_class': 7,
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'verbose': -1
        }
        self.params = params or default_params
        
    def train(self, X_train, y_train, X_val, y_val):
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=1000,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
        )
```

**为什么选择 LightGBM**:
- ✅ 课堂上可能未深入讲解
- ✅ 在表格数据上表现优异
- ✅ 训练速度快
- ✅ 有丰富的可解释性工具

**其他备选**:
- CatBoost（特别适合类别特征）
- Gradient Boosting Machine (GBM) 手动实现
- Extra Trees（极端随机树）

**验收标准**:
- [ ] 算法原理在论文中有清晰阐述
- [ ] 代码实现正确且高效
- [ ] 性能与其他算法有可比性
- [ ] 有独特的优势分析

---

### FR-007: 模型评估与对比

**优先级**: Must Have (P0)  
**描述**: 对所有算法进行全面的多维度评估和对比

**评估维度**:

#### 1️⃣ 精度指标对比
```python
metrics = {
    'accuracy': accuracy_score(y_true, y_pred),
    'precision_macro': precision_score(y_true, y_pred, average='macro'),
    'recall_macro': recall_score(y_true, y_pred, average='macro'),
    'f1_macro': f1_score(y_true, y_pred, average='macro'),
    'confusion_matrix': confusion_matrix(y_true, y_pred),
    'classification_report': classification_report(y_true, y_pred)
}
```

**输出格式**:
| 模型 | Accuracy | Precision | Recall | F1-Score | 训练时间 | 推理时间 |
|------|----------|-----------|--------|----------|----------|----------|
| Random Forest | 93.2% | 0.93 | 0.93 | 0.93 | 2.3s | 0.05s |
| XGBoost | 94.1% | 0.94 | 0.94 | 0.94 | 1.8s | 0.03s |
| MLP | 93.8% | 0.94 | 0.94 | 0.94 | 15.2s | 0.02s |
| LightGBM | 94.5% | 0.95 | 0.95 | 0.95 | 1.2s | 0.01s |

#### 2️⃣ Loss曲线对比
- 训练Loss vs 验证Loss
- 收敛速度对比
- 过拟合现象识别

#### 3️⃣ 推理速度基准测试
```python
import time

def benchmark_inference(model, X_test, n_runs=100):
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        _ = model.predict(X_test)
        end = time.perf_counter()
        times.append(end - start)
    
    return {
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times),
        'throughput': len(X_test) / np.mean(times)  # samples/sec
    }
```

#### 4️⃣ 鲁棒性测试
```python
class RobustnessTester:
    def test_gaussian_noise(self, model, X_test, noise_levels=[0.01, 0.05, 0.1, 0.2, 0.5]):
        """测试不同强度的高斯噪声"""
        results = {}
        for noise_level in noise_levels:
            X_noisy = X_test + np.random.normal(0, noise_level, X_test.shape)
            acc = evaluate(model, X_noisy, y_test)
            results[noise_level] = acc
        return results
    
    def test_missing_values(self, model, X_test, missing_rates=[0.01, 0.05, 0.1, 0.2]):
        """测试不同比例的缺失值"""
        # 使用简单填充策略（中位数）
        ...
    
    def test_label_noise(self, model, X_train, y_train, noise_rates=[0.01, 0.05, 0.1, 0.2]):
        """训练时注入标签噪声"""
        results = {}
        for noise_rate in noise_rates:
            y_noisy = inject_label_noise(y_train, noise_rate)
            model_copy = clone_and_retrain(model, X_train, y_noisy)
            acc = evaluate(model_copy, X_test, y_test)
            results[noise_rate] = acc
        return results
```

**可视化产出**:
- 📊 精度对比柱状图
- 📈 Loss曲线图（多模型叠加）
- ⏱️ 推理速度对比图
- 🛡️ 鲁棒性下降曲线（噪声强度 vs 精度）
- 🔥 混淆矩阵热力图（每个模型）
- 📉 学习曲线（训练集大小 vs 精度）

**验收标准**:
- [ ] 所有评估指标计算正确
- [ ] 对比表格格式规范
- [ ] 图表清晰美观（适合放入论文）
- [ ] 鲁棒性测试覆盖3种噪声类型
- [ ] 结果可复现（固定随机种子）

---

### FR-008: 超参数优化

**优先级**: Should Have (P1)  
**描述**: 实现自动化的超参数搜索功能

**推荐工具**: Optuna（贝叶斯优化）

```python
import optuna

def objective(trial, model_name, X_train, y_train, X_val, y_val):
    # 定义搜索空间
    if model_name == 'random_forest':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 5, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        }
    elif model_name == 'xgboost':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        }
    elif model_name == 'mlp':
        params = {
            'hidden_dims': [trial.suggest_int(f'hidden_{i}', 32, 512) for i in range(3)],
            'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
        }
    
    # 训练和评估
    model = create_model(model_name, params)
    model.train(X_train, y_train, X_val, y_val)
    val_acc = model.evaluate(X_val, y_val)['accuracy']
    
    return val_acc  # 最大化验证准确率

# 运行优化
study = optuna.create_study(direction='maximize')
study.optimize(lambda trial: objective(trial, ...), n_trials=50)

print(f"Best params: {study.best_params}")
print(f"Best validation accuracy: {study.best_value:.4f}")
```

**优化配置**:
```yaml
# config/hyperparameter_tuning.yaml
optimization:
  tool: optuna  # optuna/gridsearch/randomsearch
  n_trials: 50
  direction: maximize  # maximize accuracy or minimize loss
  cv_folds: 5
  
search_space:
  random_forest:
    n_estimators: [50, 300]
    max_depth: [5, 20]
    min_samples_split: [2, 10]
  
  xgboost:
    n_estimators: [50, 300]
    max_depth: [3, 10]
    learning_rate: [0.01, 0.3]  # log scale
    
  mlp:
    hidden_0: [32, 512]
    hidden_1: [32, 256]
    hidden_2: [16, 128]
    dropout_rate: [0.1, 0.5]
    learning_rate: [1e-4, 1e-2]  # log scale
```

**验收标准**:
- [ ] 优化后的模型性能提升 > 2%
- [ ] 优化过程可视化（参数重要性、优化历史）
- [ ] 支持多种优化算法（TPE/CMA-ES/Grid）
- [ ] 优化结果可复现

---

### FR-009: 模型集成

**优先级**: Should Have (P1)  
**描述**: 实现模型集成技术以提升最终性能

**推荐方法**:

#### 1. Voting Classifier（投票集成）
```python
from sklearn.ensemble import VotingClassifier

voting_clf = VotingClassifier(
    estimators=[
        ('rf', best_rf_model),
        ('xgb', best_xgb_model),
        ('lgbm', best_lgbm_model),
    ],
    voting='soft'  # soft voting 使用概率加权
)

voting_clf.fit(X_train, y_train)
y_pred = voting_clf.predict(X_test)
```

#### 2. Stacking Classifier（堆叠集成）
```python
from sklearn.ensemble import StackingClassifier

stacking_clf = StackingClassifier(
    estimators=[
        ('rf', best_rf_model),
        ('xgb', best_xgb_model),
        ('mlp', best_mlp_model),
    ],
    final_estimator=LogisticRegression(multi_class='multinomial'),
    cv=5
)

stacking_clf.fit(X_train, y_train)
y_pred = stacking_clf.predict(X_test)
```

**集成策略对比**:
| 方法 | 原理 | 优点 | 缺点 |
|------|------|------|------|
| Hard Voting | 多数投票 | 简单鲁棒 | 忽略置信度 |
| Soft Voting | 概率平均 | 利用置信度 | 需要校准良好的概率 |
| Stacking | 元学习器融合 | 通常性能最好 | 计算成本高 |

**验收标准**:
- [ ] 集成模型的准确率 > 单一最佳模型
- [ ] 实现至少2种集成方法
- [ ] 分析各基础模型的贡献度
- [ ] 集成结果可视化

---

### FR-010: 可解释性分析

**优先级**: Should Have (P1)  
**描述**: 提供模型决策的可解释性分析

**推荐工具**:

#### 1. SHAP (SHapley Additive exPlanations)
```python
import shap

# 全局解释：特征重要性
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test, feature_names=feature_names)

# 局部解释：单样本预测
shap.force_plot(explainer.expected_value[0], shap_values[0][0], X_test.iloc[0])
```

#### 2. Feature Importance（内置）
```python
# 对于树模型
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12, 6))
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[indices])
plt.xticks(range(X.shape[1]), feature_names[indices], rotation=45)
plt.tight_layout()
```

#### 3. LIME (Local Interpretable Model-agnostic Explanations)
```python
import lime
import lime.lime_tabular

explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train.values,
    feature_names=feature_names,
    class_names=class_names,
    discretize_continuous=True
)

exp = explainer.explain_instance(
    X_test.iloc[0].values,
    model.predict_proba,
    num_features=10
)
exp.show_in_notebook(show_table=True)
```

**可视化产出**:
- 📊 SHAP Summary Plot（全局特征重要性）
- 🎯 SHAP Force Plot（单样本解释）
- 📈 SHAP Dependence Plot（特征交互效应）
- 🔴 LIME 局部解释图
- 📊 Feature Importance 对比图（不同模型）

**验收标准**:
- [ ] SHAP 分析覆盖所有模型
- [ ] 解释结果符合领域知识（面积、周长等形状特征应该重要）
- [ ] 可视化图表清晰易懂
- [ ] 支持交互式探索（Web界面）

---

### FR-011: 命令行接口（CLI）

**优先级**: Must Have (P0)  
**描述**: 提供统一的命令行入口，支持一键运行和单步调试

**设计要求**:
```bash
# 主命令
python main.py <command> [options]

# 子命令示例：

# 1. 数据分析
python main.py analyze --data-dir ./data/raw --output-dir ./results/eda

# 2. 数据处理
python main.py process --config config/data_processing.yaml --output-dir ./data/processed

# 3. 训练单个模型
python main.py train --model random_forest --config config/models/rf.yaml

# 4. 运行所有实验
python main.py run-all --experiments config/experiments.yaml --output-dir ./results

# 5. 评估模型
python main.py evaluate --model-path ./results/models/rf_best.pkl --test-data ./data/processed/test.pkl

# 6. 启动Web界面
python main.py serve --port 8501 --host localhost

# 7. 生成报告
python main.py report --results-dir ./results --output ./results/report.md
```

**实现框架**: Click 或 argparse

```python
import click

@click.group()
@click.option('--verbose', '-v', count=True, help='增加日志详细程度')
@click.pass_context
def cli(ctx, verbose):
    """Dry Bean Classification - 机器学习工程项目"""
    ctx.ensure_object(dict)
    ctx.obj['VERBOSE'] = verbose

@cli.command()
@click.option('--data-dir', type=click.Path(exists=True), required=True)
@click.option('--output-dir', type=click.Path(), default='./results/eda')
def analyze(data_dir, output_dir):
    """运行数据分析"""
    click.echo(f"正在分析数据: {data_dir}")
    run_eda(data_dir, output_dir)

@cli.command()
@click.option('--model', type=click.Choice(['random_forest', 'xgboost', 'mlp', 'lightgbm']), required=True)
@click.option('--config', type=click.Path(exists=True))
@click.option('--hyperparameter-tuning/--no-hyperparameter-tuning', default=False)
def train(model, config, hyperparameter_tuning):
    """训练指定模型"""
    ...

if __name__ == '__main__':
    cli()
```

**帮助信息**:
```bash
$ python main.py --help
Usage: main.py [OPTIONS] COMMAND [ARGS]...

  Dry Bean Classification - 机器学习工程项目

Options:
  -v, --verbose  增加日志详细程度
  --help         Show this message and exit.

Commands:
  analyze    运行数据分析
  process    数据预处理
  train      训练模型
  run-all    运行所有实验
  evaluate   评估模型
  serve      启动Web界面
  report     生成实验报告
```

**验收标准**:
- [ ] 所有子命令可用且功能正常
- [ ] 帮助信息清晰完整
- [ ] 错误提示友好（参数缺失、文件不存在等）
- [ ] 支持 `--verbose` 控制日志级别
- [ ] 支持配置文件覆盖默认参数

---

### FR-012: Web 可视化界面

**优先级**: Should Have (P1)  
**描述**: 基于 Streamlit 构建交互式 Dashboard

**页面布局设计**:

```
┌─────────────────────────────────────────────────────┐
│  🌰 Dry Bean Classification Dashboard               │
├─────────────────────────────────────────────────────┤
│  [首页] [数据探索] [模型对比] [可解释性] [关于]      │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐          │
│  │ 数据概览  │  │ 模型性能  │  │ 最佳模型  │          │
│  │ 13611条  │  │ 94.5%    │  │ LightGBM │          │
│  │ 16特征   │  │ 平均精度  │  │          │          │
│  │ 7类别   │  │          │  │          │          │
│  └──────────┘  └──────────┘  └──────────┘          │
│                                                     │
│  ┌─────────────────────────────────────────────┐    │
│  │           模型精度对比柱状图                  │    │
│  └─────────────────────────────────────────────┘    │
│                                                     │
│  ┌─────────────────┐ ┌─────────────────────────┐   │
│  │  混淆矩阵        │ │  特征重要性 (SHAP)      │   │
│  └─────────────────┘ └─────────────────────────┘   │
│                                                     │
└─────────────────────────────────────────────────────┘
```

**核心功能模块**:

#### 1️⃣ 首页仪表盘
```python
import streamlit as st

def render_dashboard():
    st.title("🌰 Dry Bean Classification Dashboard")
    
    # 关键指标卡片
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("总样本数", "13,611")
    with col2:
        st.metric("最佳模型", "LightGBM")
    with col3:
        st.metric("最高精度", "94.5%")
    
    # 模型对比图
    st.subheader("模型精度对比")
    fig = plot_accuracy_comparison()
    st.plotly_chart(fig, use_container_width=True)
```

#### 2️⃣ 数据探索页面
```python
def render_data_explorer():
    st.header("📊 数据探索")
    
    # 数据预览
    st.subheader("数据预览")
    df = load_data()
    st.dataframe(df.head(10))
    
    # 分布图
    st.subheader("特征分布")
    selected_feature = st.selectbox("选择特征", feature_names)
    fig = plot_feature_distribution(selected_feature)
    st.pyplot(fig)
    
    # 相关性矩阵
    st.subheader("相关性热力图")
    fig = plot_correlation_matrix()
    st.pyplot(fig)
    
    # 类别分布
    st.subheader("类别分布")
    fig = plot_class_distribution()
    st.pyplot(fig)
```

#### 3️⃣ 模型对比页面
```python
def render_model_comparison():
    st.header("🤖 模型对比")
    
    # 选择模型
    models = st.multiselect(
        "选择要对比的模型",
        ['Random Forest', 'XGBoost', 'MLP', 'LightGBM', 'Voting', 'Stacking'],
        ['Random Forest', 'XGBoost', 'MLP', 'LightGBM']
    )
    
    # 精度对比表
    results = load_experiment_results(models)
    st.dataframe(results)
    
    # Loss曲线
    st.subheader("Training Loss Curves")
    fig = plot_loss_curves(models)
    st.plotly_chart(fig)
    
    # 鲁棒性测试
    st.subheader("Robustness Analysis")
    noise_type = st.selectbox("噪声类型", ["Gaussian", "Missing Values", "Label Noise"])
    fig = plot_robustness_curve(models, noise_type)
    st.plotly_chart(fig)
```

#### 4️⃣ 可解释性页面
```python
def render_interpretability():
    st.header("🔍 模型可解释性")
    
    model = st.selectbox("选择模型", models)
    
    # SHAP Summary Plot
    st.subheader("全局特征重要性 (SHAP)")
    fig = plot_shap_summary(model)
    st.pyplot(fig)
    
    # 单样本解释
    st.subheader("局部解释 (LIME)")
    sample_idx = st.slider("选择样本索引", 0, len(X_test)-1, 0)
    fig = plot_lime_explanation(model, sample_idx)
    st.pyplot(fig)
```

**启动命令**:
```bash
streamlit run app.py --server.port 8501 --server.headless true
```

**验收标准**:
- [ ] 界面美观现代（使用 Streamlit 主题定制）
- [ ] 所有图表交互式（Plotly）
- [ ] 响应式布局（适配不同屏幕尺寸）
- [ ] 加载速度 < 3秒
- [ ] 支持导出图表（PNG/SVG）
- [ ] 移动端可用（响应式设计）

---

### FR-013: 实验管理与结果追踪

**优先级**: Should Have (P1)  
**描述**: 系统化管理实验过程和结果

**实现方案**: MLflow（轻量级）或自定义 JSON 日志

**实验记录格式**:
```json
{
  "experiment_id": "exp_20260610_001",
  "timestamp": "2026-06-10T16:30:00Z",
  "model_name": "random_forest",
  "hyperparameters": {
    "n_estimators": 150,
    "max_depth": 12,
    "min_samples_split": 3
  },
  "metrics": {
    "train_accuracy": 0.998,
    "val_accuracy": 0.932,
    "test_accuracy": 0.931,
    "training_time_seconds": 2.3
  },
  "artifacts": {
    "model_path": "./results/models/rf_exp_001.pkl",
    "plots": [
      "./results/figures/rf_confusion_matrix.png",
      "./results/figures/rf_learning_curve.png"
    ]
  },
  "data_version": "processed_v1.0",
  "random_seed": 42,
  "notes": "使用Optuna优化后的最佳超参数"
}
```

**目录结构**:
```
results/
├── experiments/
│   ├── exp_20260610_001_rf.json
│   ├── exp_20260610_002_xgb.json
│   └── exp_20260610_003_mlp.json
├── models/
│   ├── rf_best.pkl
│   ├── xgb_best.pkl
│   ├── mlp_best.pth
│   └── lgbm_best.txt
├── figures/
│   ├── eda/
│   │   ├── data_overview.png
│   │   ├── correlation_matrix.png
│   │   └── class_distribution.png
│   ├── training/
│   │   ├── rf_loss_curve.png
│   │   ├── xgb_loss_curve.png
│   │   └── mlp_loss_curve.png
│   ├── evaluation/
│   │   ├── accuracy_comparison.png
│   │   ├── confusion_matrices/
│   │   └── robustness_analysis/
│   └── interpretability/
│       ├── shap_summary_rf.png
│       ├── shap_summary_xgb.png
│       └── lime_examples/
└── reports/
    ├── experiment_comparison.csv
    └── final_report.md
```

**验收标准**:
- [ ] 每次实验都有完整记录
- [ ] 实验结果可追溯和复现
- [ ] 支持实验对比和筛选
- [ ] 重要文件版本化管理

---

## 非功能需求

### NFR-001: 性能要求

| 指标 | 目标值 | 备注 |
|------|--------|------|
| 模型训练时间（传统ML） | < 30秒 | 在CPU上 |
| 模型训练时间（深度学习） | < 5分钟 | 100个epoch |
| 单样本推理延迟 | < 10ms | CPU模式 |
| Web界面加载时间 | < 3秒 | 首屏渲染 |
| 数据处理时间 | < 10秒 | 全量数据 |

**优化策略**:
- 使用并行处理（`n_jobs=-1`）
- 数据缓存（避免重复加载）
- 模型序列化优化（joblib/pickle）
- 图片懒加载（Web界面）

---

### NFR-002: 可维护性

**代码规范**:
- ✅ 遵循 PEP 8 风格指南
- ✅ 类型注解（Type Hints）
- ✅ Docstring 文档（Google/Numpy风格）
- ✅ 模块化设计（单一职责原则）
- ✅ 配置与代码分离（YAML/JSON）

**目录结构规范**:
```
src/
├── data/           # 数据相关逻辑
├── models/         # 模型定义
├── training/       # 训练流程
├── evaluation/     # 评估指标
├── visualization/  # 可视化
└── utils/          # 工具函数
```

**测试覆盖率**:
- 核心模块（数据处理、模型训练）: > 80%
- 辅助模块（可视化、工具）: > 60%
- 总体目标: > 70%

---

### NFR-003: 可靠性

**错误处理**:
- 所有可能失败的操作都有 try-except
- 错误信息友好且包含上下文
- 关键操作支持重试机制
- 日志记录完整（DEBUG/INFO/WARNING/ERROR）

**数据完整性**:
- 固定随机种子（`random_state=42`）
- 实验结果可复现
- 数据版本管理
- 模型版本管理

---

### NFR-004: 可扩展性

**插件化设计**:
- 新增算法只需继承基类并注册
- 新增评估指标只需添加函数
- 新增可视化图表只需添加绘图函数

**配置驱动**:
- 所有超参数通过 YAML 配置
- 支持多套配置文件切换（dev/prod/test）
- 环境变量支持敏感信息

---

### NFR-005: 安全性（学术项目简化版）

**数据安全**:
- ✅ 不存储个人隐私数据（使用公开数据集）
- ✅ 不硬编码密码或密钥（如有）
- ✅ 输入数据校验（防止注入攻击，虽然不太可能）

**代码安全**:
- ✅ 依赖版本锁定（requirements.txt + pip freeze）
- ✅ 不使用 `exec()` 或 `eval()` 处理用户输入
- ✅ 文件路径安全（防止路径遍历）

**Git 安全**:
- ✅ `.gitignore` 排除敏感文件（`.env`, `*.pem`）
- ✅ 不提交大型二进制文件（模型文件 > 100MB）
- ✅ 不提交数据文件（使用 Git LFS 或外部下载）

---

### NFR-006: 可观测性

**日志系统**:
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)
```

**日志级别**:
- DEBUG: 详细调试信息（变量值、中间结果）
- INFO: 一般信息（开始/结束训练、关键步骤）
- WARNING: 警告信息（性能问题、潜在风险）
- ERROR: 错误信息（异常、失败操作）
- CRITICAL: 严重错误（程序无法继续）

**监控指标**:
- 训练 Loss/Accuracy（实时记录）
- 内存使用量（避免OOM）
- 磁盘空间（结果文件存储）
- 运行时间（性能瓶颈识别）

---

## 技术栈决策

### 核心技术栈

| 类别 | 选择 | 版本 | 替代方案 | 选择理由 |
|------|------|------|----------|----------|
| **语言** | Python | 3.9+ | - | ML生态最丰富 |
| **包管理** | pip + venv | 最新 | poetry/conda | 简单通用 |
| **传统ML** | scikit-learn | 1.3+ | - | 业界标准 |
| **梯度提升** | XGBoost | 1.7+ | LightGBM | 性能优秀 |
| **梯度提升** | LightGBM | 4.0+ | XGBoost | 速度快 |
| **深度学习** | PyTorch | 2.0+ | TensorFlow | 研究友好 |
| **超参数优化** | Optuna | 3.0+ | Hyperopt | 易用高效 |
| **可视化** | Matplotlib | 3.7+ | - | 基础绑图 |
| **可视化** | Seaborn | 0.12+ | - | 统计图表 |
| **可视化** | Plotly | 5.0+ | Bokeh | 交互式图表 |
| **可解释性** | SHAP | 0.41+ | LIME | 全面强大 |
| **Web框架** | Streamlit | 1.28+ | Gradio | 快速原型 |
| **CLI框架** | Click | 8.1+ | argparse | 优雅易用 |
| **配置管理** | PyYAML | 6.0 | - | YAML解析 |
| **实验管理** | MLflow | 2.0+ | 自定义 | 轻量可选 |
| **测试** | pytest | 7.0+ | unittest | 现代测试 |
| **代码质量** | Black | 23.0+ | autopep8 | 格式化 |
| **代码质量** | isort | 5.0+ | - | 导入排序 |
| **类型检查** | mypy | 1.0+ | - | 静态类型检查 |

### 依赖清单

**requirements.txt**:
```txt
# 数据处理
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.10.0

# 机器学习
scikit-learn>=1.3.0
xgboost>=1.7.0
lightgbm>=4.0.0

# 深度学习
torch>=2.0.0
torchvision>=0.15.0

# 超参数优化
optuna>=3.0.0

# 可视化
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.15.0
kaleido>=0.2.1  # 静态图片导出

# 可解释性
shap>=0.41.0
lime>=0.2.0

# Web界面
streamlit>=1.28.0

# CLI
click>=8.1.0

# 配置
pyyaml>=6.0

# 实验管理（可选）
mlflow>=2.0.0

# 测试
pytest>=7.0.0
pytest-cov>=4.0.0

# 代码质量
black>=23.0.0
isort>=5.12.0
mypy>=1.0.0

# 工具
tqdm>=4.65.0  # 进度条
joblib>=1.2.0  # 序列化
python-dotenv>=1.0.0  # 环境变量
```

### 技术约束

**必须使用**:
- ✅ Python 3.9+
- ✅ PyTorch（如果涉及深度学习）
- ✅ scikit-learn（传统ML）
- ✅ Streamlit（Web界面）

**禁止使用**:
- ❌ TensorFlow/Keras（除非有特殊理由）
- ❌ 过于复杂的架构（Transformer等）
- ❌ 商业化API或付费服务
- ❌ 未经验证的第三方库

**需要审批才能替换**:
- ⚠️ 核心算法库（sklearn/xgboost/pytorch）
- ⚠️ Web框架（streamlit）
- ⚠️ 可视化库（matplotlib/plotly）

---

## 架构设计与模块边界

### 系统架构图

```
┌─────────────────────────────────────────────────────────────┐
│                      用户界面层                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │  CLI (Click) │  │  Web UI     │  │  Jupyter Notebooks  │ │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘ │
└─────────┼────────────────┼────────────────────┼────────────┘
          │                │                    │
┌─────────▼────────────────▼────────────────────▼────────────┐
│                      业务逻辑层                              │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────────┐ │
│  │ 数据处理  │ │ 模型训练  │ │ 模型评估  │ │  可视化生成    │ │
│  │ Pipeline │ │ Pipeline │ │ Pipeline │ │  Pipeline      │ │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └───────┬────────┘ │
└───────┼────────────┼────────────┼───────────────┼──────────┘
        │            │            │               │
┌───────▼────────────▼────────────▼───────────────▼──────────┐
│                      基础设施层                              │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────────┐ │
│  │ 数据加载  │ │ 模型定义  │ │ 评估指标  │ │  配置管理      │ │
│  │ Module   │ │ Module   │ │ Module   │ │  Module        │ │
│  └──────────┘ └──────────┘ └──────────┘ └────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────▼─────────────────────────────┐
│                      数据层                                 │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────────┐ │
│  │ Raw Data │ │Processed │ │ Models   │ │  Results       │ │
│  │ (CSV)    │ │ (PKL)    │ │ (PKL/PTH)│ │  (Figures/Logs)│ │
│  └──────────┘ └──────────┘ └──────────┘ └────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### 目录结构详解

```
dry-bean-classification/
│
├── 📄 README.md                          # 专业级项目文档
├── 📄 requirements.txt                   # Python依赖
├── 📄 setup.py                           # 安装配置
├── 📄 .gitignore                         # Git忽略规则
├── 📄 .env.example                       # 环境变量模板
│
├── 📁 config/                            # 配置文件
│   ├── default.yaml                      # 默认配置
│   ├── data_processing.yaml              # 数据处理参数
│   ├── models/                           # 模型配置
│   │   ├── random_forest.yaml
│   │   ├── xgboost.yaml
│   │   ├── mlp.yaml
│   │   └── lightgbm.yaml
│   ├── training.yaml                     # 训练超参数
│   └── hyperparameter_tuning.yaml        # 优化配置
│
├── 📁 data/                              # 数据目录
│   ├── raw/                              # 原始数据（Git忽略）
│   │   ├── Dry_Bean_Dataset_Dirty_train.csv
│   │   ├── Dry_Bean_Dataset_Dirty_val.csv
│   │   └── Dry_Bean_Dataset_Dirty_test.csv
│   ├── processed/                        # 处理后数据（Git忽略）
│   │   ├── train_processed.pkl
│   │   ├── val_processed.pkl
│   │   └── test_processed.pkl
│   └── external/                         # 外部数据（如有）
│
├── 📁 src/                               # 源代码
│   ├── __init__.py
│   │
│   ├── 📁 data/                          # 数据模块
│   │   ├── __init__.py
│   │   ├── loader.py                     # 数据加载器
│   │   ├── cleaner.py                    # 数据清洗
│   │   ├── feature_engineering.py        # 特征工程
│   │   └── dataset.py                    # PyTorch Dataset
│   │
│   ├── 📁 models/                        # 模型定义
│   │   ├── __init__.py
│   │   ├── base.py                       # 模型基类
│   │   ├── traditional.py                # 传统ML模型
│   │   ├── deep_learning.py              # PyTorch模型
│   │   ├── advanced.py                   # 进阶算法
│   │   └── ensemble.py                   # 集成模型
│   │
│   ├── 📁 training/                      # 训练模块
│   │   ├── __init__.py
│   │   ├── trainer.py                    # 训练器
│   │   ├── hyperparameter_tuning.py      # 超参优化
│   │   └── callbacks.py                  # 回调函数
│   │
│   ├── 📁 evaluation/                    # 评估模块
│   │   ├── __init__.py
│   │   ├── metrics.py                    # 评估指标
│   │   ├── comparator.py                 # 模型对比
│   │   ├── robustness.py                 # 鲁棒性测试
│   │   └── interpretability.py           # 可解释性
│   │
│   ├── 📁 visualization/                 # 可视化模块
│   │   ├── __init__.py
│   │   ├── plots.py                      # Matplotlib绑定
│   │   ├── interactive_plots.py          # Plotly绑定
│   │   └── dashboard.py                  # Streamlit组件
│   │
│   └── 📁 utils/                         # 工具函数
│       ├── __init__.py
│       ├── logger.py                     # 日志配置
│       ├── io_utils.py                   # 文件IO
│       ├── math_utils.py                 # 数学工具
│       └── seed.py                       # 随机种子
│
├── 📁 experiments/                       # 实验脚本
│   ├── __init__.py
│   ├── run_all.py                        # 一键运行所有实验
│   ├── run_single.py                     # 单算法调试
│   ├── compare_models.py                 # 模型对比
│   └── generate_report.py               # 生成报告
│
├── 📁 notebooks/                         # Jupyter笔记本
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_data_processing.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_model_evaluation.ipynb
│   └── 05_interpretability_analysis.ipynb
│
├── 📁 tests/                             # 单元测试
│   ├── __init__.py
│   ├── conftest.py                       # pytest fixtures
│   ├── test_data_loader.py
│   ├── test_data_cleaner.py
│   ├── test_models.py
│   ├── test_training.py
│   └── test_evaluation.py
│
├── 📁 results/                           # 实验结果（Git忽略）
│   ├── experiments/                      # 实验记录
│   ├── models/                           # 保存的模型
│   ├── figures/                          # 生成的图表
│   │   ├── eda/
│   │   ├── training/
│   │   ├── evaluation/
│   │   └── interpretability/
│   └── reports/                          # 实验报告
│
├── 📁 docs/                              # 额外文档
│   ├── theory.md                         # 数学理论补充
│   ├── api_reference.md                  # API文档
│   └── development_guide.md             # 开发指南
│
├── 📁 logs/                              # 日志文件（Git忽略）
│
├── app.py                                # Streamlit主应用
└── main.py                               # CLI入口
```

### 模块职责边界

| 模块 | 职责 | 公开接口 | 禁止事项 |
|------|------|----------|----------|
| `src/data` | 数据加载、清洗、特征工程 | `DataLoader`, `DataCleaner`, `FeatureEngineer` | 不包含任何模型逻辑 |
| `src/models` | 模型定义和推理 | `BaseModel`, `RFModel`, `XGBModel`, `MLPModel` | 不包含训练循环 |
| `src/training` | 训练流程、超参优化 | `Trainer`, `HyperparameterTuner` | 不直接调用可视化 |
| `src/evaluation` | 评估指标、对比分析 | `MetricsCalculator`, `ModelComparator` | 不修改模型参数 |
| `src/visualization` | 图表生成、Dashboard | `plot_*`, `render_*` | 不包含业务逻辑 |
| `src/utils` | 通用工具函数 | `set_seed`, `setup_logger` | 不包含业务特定逻辑 |

---

## 构建/运行/验证契约

### 环境设置

```bash
# 1. 创建虚拟环境
python -m venv venv

# 2. 激活虚拟环境
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 验证安装
python -c "import torch; import sklearn; import streamlit; print('✅ All packages installed!')"
```

### 运行命令

#### 开发模式
```bash
# 数据分析（生成EDA报告）
python main.py analyze --data-dir data/raw --output-dir results/figures/eda

# 数据处理
python main.py process --config config/data_processing.yaml --output-dir data/processed

# 训练单个模型（带超参优化）
python main.py train --model random_forest --hyperparameter-tuning

# 运行所有实验（约需30分钟）
python main.py run-all --config config/default.yaml

# 启动Web界面
python main.py serve --port 8501

# 生成最终报告
python main.py report --results-dir results --output results/report/final_report.md
```

#### 测试
```bash
# 运行所有测试
pytest tests/ -v --cov=src --cov-report=html

# 运行特定测试
pytest tests/test_data_loader.py -v

# 运行测试并生成覆盖率报告
pytest tests/ --cov=src --cov-report=term-missing
```

#### 代码质量
```bash
# 代码格式化
black src/ tests/
isort src/ tests/

# 类型检查
mypy src/

# Lint检查（可选）
flake8 src/ tests/
```

### 验收检查清单

#### 功能验收
- [ ] **数据分析**: 能生成完整的数据质量报告和EDA图表
- [ ] **数据处理**: 能将脏数据转换为干净的模型输入
- [ ] **模型训练**: 4种以上算法都能成功训练
- [ ] **模型评估**: 能生成完整的对比表格和图表
- [ ] **鲁棒性测试**: 3种噪声类型的测试都能正常运行
- [ ] **可解释性**: SHAP和LIME分析结果合理
- [ ] **CLI**: 所有子命令正常工作
- [ ] **Web界面**: 所有页面可访问且显示正常

#### 质量验收
- [ ] 代码无语法错误和明显的bug
- [ ] 核心测试通过率 = 100%
- [ ] 代码符合 PEP 8 规范
- [ ] 无硬编码的敏感信息
- [ ] 日志输出完整且有层次

#### 文档验收
- [ ] README 完整且专业（含Badge、安装指南、使用示例）
- [ ] 代码注释充分（特别是复杂逻辑）
- [ ] API文档清晰（Docstring）
- [ ] 实验结果可复现（固定随机种子）

#### 性能验收
- [ ] 传统ML训练时间 < 30秒
- [ ] 深度学习训练时间 < 5分钟
- [ ] Web界面加载 < 3秒
- [ ] 内存占用合理（< 4GB）

---

## 安全、风险与合规

### 数据安全

**数据分类**: 公开学术数据集（无需特殊处理）

**数据来源**: UCI Machine Learning Repository - Dry Bean Dataset  
**数据许可**: CC BY 4.0（允许商业和非商业使用）

**数据处理原则**:
- ✅ 仅使用提供的训练/验证/测试集
- ✅ 不尝试获取额外的真实标签信息
- ✅ 不篡改测试集标签
- ✅ 保持数据处理的透明性和可复现性

### 代码安全

**依赖安全**:
- 定期更新依赖版本（每月一次）
- 使用 `pip-audit` 检查已知漏洞
- 锁定依赖版本（requirements.txt + pip freeze > requirements.lock）

**输入验证**:
- 所有用户输入（CLI参数、配置文件）都经过验证
- 文件路径安全（防止路径遍历攻击）
- 数值参数范围检查（防止溢出或无效值）

### 风险管理

#### 已识别风险

| 风险ID | 风险描述 | 可能性 | 影响程度 | 缓解措施 | 负责人 |
|--------|----------|--------|----------|----------|--------|
| R001 | 数据质量问题比预期严重 | 高 | 中 | 增强数据清洗策略；手动审查 | 学生 |
| R002 | 深度学习模型过拟合 | 中 | 高 | 早停、正则化、数据增强 | 学生 |
| R003 | 某些算法性能不佳 | 低 | 中 | 准备多个备选算法；集成学习 | 学生 |
| R004 | 时间不够完成所有功能 | 中 | 高 | 优先保证核心功能；高级特性可选 | 学生 |
| R005 | PyTorch安装问题（Windows） | 低 | 中 | 提供详细的安装指南；备选方案 | 学生 |
| R006 | GitHub上传大文件失败 | 中 | 低 | 使用Git LFS；排除大文件 | 学生 |

#### 应急预案

**如果深度学习训练太慢**:
- 减少 epoch 数（50→30）
- 减小模型规模（hidden_dims: [256,128,64] → [128,64,32]）
- 仅使用CPU（放弃GPU加速）

**如果某个算法效果不好**:
- 尝试不同的超参数组合
- 检查数据是否有问题
- 用更简单的算法替代（如KNN）

**如果时间紧迫**:
- 优先完成核心功能（FR-001 到 FR-007, FR-011）
- 砍掉高级特性（FR-008 到 FR-010）
- 简化Web界面（仅保留核心页面）

---

## 可观测性与运维准备

### 日志规范

**日志格式**:
```
[时间戳] [模块名] [日志级别] [消息] [附加信息]

示例:
[2026-06-10 16:30:00] [data.loader] [INFO] Loading training data from: data/raw/train.csv
[2026-06-10 16:30:01] [data.loader] [INFO] Loaded 10888 samples with 17 columns
[2026-06-10 16:30:01] [data.cleaner] [WARNING] Found 15 missing values in column 'Solidity'
[2026-06-10 16:30:02] [models.rf] [ERROR] Training failed: Invalid parameter value
```

**日志文件**:
- `logs/app.log`: 应用主日志（所有级别）
- `logs/training.log`: 训练专用日志（详细Loss/Accuracy）
- `logs/error.log`: 仅错误日志（便于排查）

### 监控指标

**训练阶段**:
- 当前Epoch / 总Epoch
- Training Loss / Validation Loss
- Training Accuracy / Validation Accuracy
- Learning Rate
- 已用时间 / 预计剩余时间

**评估阶段**:
- 各指标的数值（Accuracy, F1等）
- 混淆矩阵
- 推理时间
- 内存占用

**Web界面**:
- 页面加载时间
- 用户操作日志
- 错误请求

### 实验追踪

**每次实验记录**:
- 实验ID和时间戳
- 使用的数据和配置
- 模型和超参数
- 评估指标
- 生成的文件路径
- 备注

**实验对比**:
- 支持按模型、时间、指标筛选
- 支持导出对比表格
- 支持可视化趋势图

---

## 实施计划（Phase Plan）

### Phase 0: 基础设施搭建（预计2天）

**目标**: 搭建项目骨架和开发环境

#### Epic E0: 项目初始化
- [ ] **E0.1**: 创建项目目录结构和配置文件
  - [ ] 初始化Git仓库
  - [ ] 创建`.gitignore`（排除`venv/`, `data/`, `results/`, `*.pyc`, `__pycache__/`）
  - [ ] 创建`requirements.txt`
  - [ ] 创建虚拟环境并安装依赖
  - [ ] 创建基本的`config/default.yaml`

- [ ] **E0.2**: 实现核心工具模块
  - [ ] 实现`src/utils/logger.py`（日志配置）
  - [ ] 实现`src/utils/seed.py`（随机种子设置）
  - [ ] 实现`src/utils/io_utils.py`（文件读写工具）
  - [ ] 编写单元测试验证工具函数

- [ ] **E0.3**: 搭建CLI框架
  - [ ] 使用Click创建`main.py`骨架
  - [ ] 实现基本的`--help`和子命令占位符
  - [ ] 添加`--verbose`选项控制日志级别

**验收标准**:
- [ ] `python main.py --help` 显示帮助信息
- [ ] `pytest tests/ -v` 运行无报错
- [ ] `black src/` 格式化无警告

**预估工时**: 8小时

---

### Phase 1: 数据分析与处理（预计3天）

**目标**: 完成数据的理解、清洗和预处理

#### Epic E1: 数据加载与探索
- [ ] **E1.1**: 实现数据加载器
  - [ ] 实现`src/data/loader.py`（DataLoader类）
  - [ ] 支持加载train/val/test三个数据集
  - [ ] 自动识别特征列和标签列
  - [ ] 处理编码问题（UTF-8/GBK自动检测）
  - [ ] 返回数据基本信息（形状、类型、缺失值统计）

- [ ] **E1.2**: 实现数据质量分析
  - [ ] 实现`src/data/quality_assessment.py`
  - [ ] 缺失值分析（统计每列缺失数量和比例）
  - [ ] 异常值检测（IQR方法和Z-Score方法）
  - [ ] 标签一致性检查（检测拼写错误）
  - [ ] 生成数据质量报告（Markdown/JSON格式）

- [ ] **E1.3**: 实现EDA可视化
  - [ ] 实现`src/visualization/plots.py`中的EDA绑定函数
  - [ ] 数据概览图（前几行数据、基本统计）
  - [ ] 特征分布直方图（每个特征一张图）
  - [ ] 相关性热力图
  - [ ] 类别分布饼图/柱状图
  - [ ] 缺失值热力图
  - [ ] 箱线图（按类别分组）
  - [ ] 保存所有图表到`results/figures/eda/`

**验收标准**:
- [ ] `python main.py analyze` 生成完整的EDA报告
- [ ] 自动发现第9行的Solidity缺失值和标签错误
- [ ] 所有图表清晰可读

**预估工时**: 12小时

#### Epic E2: 数据清洗与特征工程
- [ ] **E2.1**: 实现数据清洗管道
  - [ ] 实现`src/data/cleaner.py`（DataCleaner类）
  - [ ] 缺失值处理（中位数填充/KNN填充）
  - [ ] 异常值处理（Winsorization/截断）
  - [ ] 标签纠错（基于规则的纠正）
  - [ ] 支持配置文件自定义策略

- [ ] **E2.2**: 实现特征工程
  - [ ] 实现`src/data/feature_engineering.py`
  - [ ] 特征标准化（StandardScaler/MinMaxScaler）
  - [ ] 特征选择（相关性过滤/重要性排序/RFE）
  - [ ] 特征构造（多项式特征/交互特征，可选）
  - [ ] 保存处理后的数据（pickle格式）

- [ ] **E2.3**: 创建数据处理CLI命令
  - [ ] 实现`main.py process`子命令
  - [ ] 支持通过`--config`指定配置文件
  - [ ] 输出处理前后数据对比报告

**验收标准**:
- [ ] `python main.py process` 生成干净的处理后数据
- [ ] 处理后数据无缺失值、无异常标签
- [ ] 特征数值范围适合模型输入

**预估工时**: 10小时

---

### Phase 2: 模型实现与训练（预计4天）

**目标**: 实现4种以上的分类算法并完成训练

#### Epic E3: 传统机器学习模型
- [ ] **E3.1**: 实现模型基类和接口
  - [ ] 定义`src/models/base.py`（BaseModel抽象类）
  - [ ] 统一的接口：`train()`, `predict()`, `evaluate()`, `save()`, `load()`
  - [ ] 标准化的输出格式

- [ ] **E3.2**: 实现Random Forest模型
  - [ ] 实现`src/models/traditional.py`中的RFModel类
  - [ ] 默认超参数配置
  - [ ] 支持自定义超参数
  - [ ] 训练日志和进度显示

- [ ] **E3.3**: 实现XGBoost模型
  - [ ] 实现XGBModel类
  - [ ] 处理类别标签编码（LabelEncoder）
  - [ ] Early Stopping支持
  - [ ] 特征重要性提取

- [ ] **E3.4**: 实现训练器
  - [ ] 实现`src/training/trainer.py`（Trainer类）
  - [ ] 统一的训练流程封装
  - [ ] 训练历史记录（Loss/Accuracy曲线数据）
  - [ ] 模型保存和加载
  - [ ] 实验记录JSON生成

**验收标准**:
- [ ] Random Forest测试集准确率 > 90%
- [ ] XGBoost测试集准确率 > 92%
- [ ] 训练过程有完整日志

**预估工时**: 12小时

#### Epic E4: 深度学习模型
- [ ] **E4.1**: 实现PyTorch数据集
  - [ ] 实现`src/data/dataset.py`（DryBeanDataset类）
  - [ ] 继承`torch.utils.data.Dataset`
  - [ ] 支持Tensor转换和数据归一化
  - [ ] DataLoader创建（batch_size, shuffle, num_workers）

- [ ] **E4.2**: 实现MLP模型架构
  - [ ] 实现`src/models/deep_learning.py`中的MLPClassifier
  - [ ] 可配置的层数和神经元数量
  - [ ] Batch Normalization + Dropout
  - [ ] 权重初始化（Xavier/He）

- [ ] **E4.3**: 实现深度学习训练循环
  - [ ] 扩展`Trainer`类支持PyTorch模型
  - [ ] Epoch级别的训练和验证
  - [ ] Loss计算和反向传播
  - [ ] 优化器和学习率调度器
  - [ ] Early Stopping回调
  - [ ] GPU/CPU自动检测

- [ ] **E4.4**: 训练调优
  - [ ] 调整超参数使模型收敛
  - [ ] 解决过拟合/欠拟合问题
  - [ ] 保存最佳模型checkpoint

**验收标准**:
- [ ] MLP测试集准确率 > 92%
- [ ] Loss曲线平滑收敛
- [ ] 无明显过拟合（train-val gap < 5%）

**预估工时**: 16小时

#### Epic E5: 进阶算法
- [ ] **E5.1**: 实现LightGBM模型
  - [ ] 实现`src/models/advanced.py`中的LightGBMModel
  - [ ] 参数配置和调优
  - [ ] 与其他模型的接口一致

- [ ] **E5.2**: 编写算法原理文档
  - [ ] 在`docs/theory.md`中补充LightGBM原理
  - [ ] 数学公式推导（使用LaTeX）
  - [ ] 与传统GBDT的对比分析

**验收标准**:
- [ ] LightGBM测试集准确率 > 93%
- [ ] 原理文档清晰完整

**预估工时**: 8小时

---

### Phase 3: 评估与分析（预计3天）

**目标**: 完成全面的模型评估、对比和分析

#### Epic E6: 模型评估与对比
- [ ] **E6.1**: 实现评估指标计算
  - [ ] 实现`src/evaluation/metrics.py`
  - [ ] Accuracy, Precision, Recall, F1-Score（宏平均）
  - [ ] Confusion Matrix
  - [ ] Classification Report
  - [ ] ROC Curve（One-vs-Rest，可选）

- [ ] **E6.2**: 实现模型对比工具
  - [ ] 实现`src/evaluation/comparator.py`（ModelComparator类）
  - [ ] 加载多个模型并批量评估
  - [ ] 生成对比表格（CSV/Markdown）
  - [ ] 推理速度基准测试
  - [ ] Loss曲线对比图

- [ ] **E6.3**: 实现鲁棒性测试
  - [ ] 实现`src/evaluation/robustness.py`（RobustnessTester类）
  - [ ] 高斯噪声测试（多个噪声水平）
  - [ ] 缺失值噪声测试
  - [ ] 标签噪声测试
  - [ ] 生成鲁棒性下降曲线图

- [ ] **E6.4**: 实现过拟合分析
  - [ ] 训练/验证精度差异分析
  - [ ] 学习曲线（样本量 vs 精度）
  - [ ] 正则化效果对比

**验收标准**:
- [ ] 所有评估指标计算正确
- [ ] 对比表格和图表完整
- [ ] 鲁棒性测试覆盖3种噪声类型

**预估工时**: 14小时

#### Epic E7: 可解释性分析
- [ ] **E7.1**: 实现SHAP分析
  - [ ] 实现`src/evaluation/interpretability.py`
  - [ ] SHAP Summary Plot（全局特征重要性）
  - [ ] SHAP Dependence Plot（特征交互）
  - [ ] SHAP Force Plot（单样本解释）
  - [ ] 支持树模型和线性模型

- [ ] **E7.2**: 实现LIME分析
  - [ ] LIME局部解释
  - [ ] 支持交互式展示（Notebook/Web）

- [ ] **E7.3**: Feature Importance对比
  - [ ] 不同模型的特征重要性对比图
  - [ ] 与领域知识的交叉验证

**验收标准**:
- [ ] SHAP分析结果合理（形状特征最重要）
- [ ] 可视化图表清晰

**预估工时**: 10小时

---

### Phase 4: 高级特性（预计2天）

**目标**: 实现超参数优化和模型集成

#### Epic E8: 超参数优化
- [ ] **E8.1**: 集成Optuna
  - [ ] 实现`src/training/hyperparameter_tuning.py`
  - [ ] 为每个算法定义搜索空间
  - [ ] TPE采样器（默认）
  - [ ] 早停 trials（Pruner）
  - [ ] 并行优化（可选）

- [ ] **E8.2**: 优化结果分析
  - [ ] 参数重要性图
  - [ ] 优化历史曲线
  - [ ] 最优超参数对比表

**验收标准**:
- [ ] 优化后模型性能提升 > 2%
- [ ] 优化过程可视化

**预估工时**: 8小时

#### Epic E9: 模型集成
- [ ] **E9.1**: 实现Voting集成
  - [ ] Hard Voting和Soft Voting
  - [ ] 动态选择参与投票的模型

- [ ] **E9.2**: 实现Stacking集成
  - [ ] 元学习器选择（Logistic Regression）
  - [ ] 交叉验证策略

- [ ] **E9.3**: 集成效果分析
  - [ ] 集成 vs 单模型对比
  - [ ] 各模型贡献度分析

**验收标准**:
- [ ] 集成模型准确率 > 单一最佳模型
- [ ] 至少实现2种集成方法

**预估工时**: 8小时

---

### Phase 5: 工程集成与展示（预计3天）

**目标**: 完成CLI、Web界面和文档

#### Epic E10: CLI完善
- [ ] **E10.1**: 实现所有CLI子命令
  - [ ] `analyze` - 数据分析
  - [ ] `process` - 数据处理
  - [ ] `train` - 训练模型
  - [ ] `run-all` - 运行所有实验
  - [ ] `evaluate` - 评估模型
  - [ ] `serve` - 启动Web
  - [ ] `report` - 生成报告

- [ ] **E10.2**: 实现一键运行脚本
  - [ ] `experiments/run_all.py`
  - [ ] 按顺序执行所有实验
  - [ ] 错误处理和日志记录
  - [ ] 进度显示（tqdm）

**验收标准**:
- [ ] 所有CLI命令正常工作
- [ ] `python main.py run-all` 能完成全流程

**预估工时**: 8小时

#### Epic E11: Web界面开发
- [ ] **E11.1**: 实现Streamlit主应用
  - [ ] `app.py`主文件
  - [ ] 侧边栏导航
  - [ ] 主题定制（深色/浅色模式）

- [ ] **E11.2**: 实现首页仪表盘
  - [ ] 关键指标卡片
  - [ ] 模型精度对比图
  - [ ] 快速导航按钮

- [ ] **E11.3**: 实现数据探索页面
  - [ ] 数据预览表格
  - [ ] 特征分布图（交互式选择）
  - [ ] 相关性矩阵
  - [ ] 类别分布图

- [ ] **E11.4**: 实现模型对比页面
  - [ ] 模型选择器
  - [ ] 精度对比表
  - [ ] Loss曲线图（Plotly交互式）
  - [ ] 鲁棒性分析图
  - [ ] 混淆矩阵（可切换模型）

- [ ] **E11.5**: 实现可解释性页面
  - [ ] SHAP Summary Plot
  - [ ] LIME局部解释（样本选择器）
  - [ ] Feature Importance对比

- [ ] **E11.6**: 界面美化和优化
  - [ ] 响应式布局
  - [ ] 加载动画
  - [ ] 图片懒加载
  - [ ] 导出功能（PNG/SVG/PDF）

**验收标准**:
- [ ] Web界面美观现代
- [ ] 所有页面可访问
- [ ] 交互流畅（< 3秒响应）

**预估工时**: 16小时

#### Epic E12: 文档撰写
- [ ] **E12.1**: 撰写专业级README
  - [ ] 项目标题和简介（带Badge）
  - [ ] 功能特性列表
  - [ ] 快速开始指南（Installation + Quick Start）
  - [ ] 项目结构说明
  - [ ] 使用示例（代码块 + 截图）
  - [ ] 实验结果表格
  - [ ] 模型对比分析
  - [ ] 技术栈列表
  - [ ] 贡献指南（可选）
  - [ ] 许可证（MIT/Apache-2.0）
  - [ ] 致谢（课程信息）

- [ ] **E12.2**: 撰写API文档
  - [ ] 核心类的使用示例
  - [ ] 配置文件说明
  - [ ] CLI命令参考

- [ ] **E12.3**: 生成论文素材
  - [ ] 从Notebook导出高清图表
  - [ ] 生成实验结果表格（LaTeX格式）
  - [ ] 整理算法原理描述

**验收标准**:
- [ ] README完整且专业（> 500行）
- [ ] 包含截图和示例
- [ ] Badge显示正常

**预估工时**: 10小时

---

### Phase 6: 测试与优化（预计2天）

**目标**: 确保代码质量和稳定性

#### Epic E13: 测试完善
- [ ] **E13.1**: 补充单元测试
  - [ ] 数据加载和处理模块测试
  - [ ] 模型训练和推理测试
  - [ ] 评估指标计算测试
  - [ ] 工具函数测试
  - [ ] 目标覆盖率 > 70%

- [ ] **E13.2**: 集成测试
  - [ ] 端到端流程测试（数据→训练→评估）
  - [ ] CLI命令测试
  - [ ] Web界面 smoke test

- [ ] **E13.3**: 性能测试
  - [ ] 训练时间基准测试
  - [ ] 推理速度基准测试
  - [ ] 内存占用测试

**验收标准**:
- [ ] 核心测试通过率 = 100%
- [ ] 总体覆盖率 > 70%

**预估工时**: 8小时

#### Epic E14: 最终优化与修复
- [ ] **E14.1**: Bug修复
  - [ ] 修复测试中发现的问题
  - [ ] 修复用户反馈的问题

- [ ] **E14.2**: 性能优化
  - [ ] 优化热点函数（profiling）
  - [ ] 减少内存占用
  - [ ] 加速数据处理流程

- [ ] **E14.3**: 代码清理
  - [ ] 移除dead code
  - [ ] 统一代码风格
  - [ ] 补充遗漏的注释

- [ ] **E14.4**: 最终验证
  - [ ] 运行完整流程（`run-all`）
  - [ ] 检查所有输出文件
  - [ ] 验证GitHub仓库完整性

**验收标准**:
- [ ] 无明显Bug
- [ ] 代码整洁规范
- [ ] 完整流程可复现

**预估工时**: 8小时

---

### Phase 7: 提交与展示准备（预计1天）

**目标**: 准备提交材料和现场演示

#### Epic E15: GitHub仓库整理
- [ ] **E15.1**: 仓库美化
  - [ ] 添加GitHub模板（ISSUE/PR）
  - [ ] 配置`.github/`目录（可选）
  - [ ] 添加License文件
  - [ ] 添加`demo.gif`或截图（可选）

- [ ] **E15.2**: 版本发布
  - [ ] 创建Git Tag（v1.0.0）
  - [ ] 撰写Release Notes
  - [ ] 打包必要文件

**验收标准**:
- [ ] GitHub仓库专业美观
- [ ] README显示正常

**预估工时**: 4小时

#### Epic E16: 演示准备
- [ ] **E16.1**: 准备演示脚本
  - [ ] 演示流程设计（10-15分钟）
  - [ ] 重点功能介绍
  - [ ] 可能的问题和答案准备

- [ ] **E16.2**: 备份和归档
  - [ ] 备份所有重要文件
  - [ ] 生成最终的实验报告PDF
  - [ ] 整理论文素材

**验收标准**:
- [ ] 演示流程顺畅
- [ ] 所有材料齐全

**预估工时**: 4小时

---

## 时间线总览

```
Week 1 (06.10 - 06.16): Phase 0-2 完成
├── Day 1-2:  Phase 0 (基础设施)
├── Day 3-5:  Phase 1 (数据处理)
└── Day 6-7:  Phase 2 开始 (模型实现)

Week 2 (06.17 - 06.23): Phase 2-4 完成
├── Day 8-10: Phase 2 继续 (深度学习+进阶算法)
├── Day 11-13: Phase 3 (评估分析)
└── Day 14:   Phase 4 (高级特性)

Week 3 (06.24 - 06.28): Phase 5-7 + 提交
├── Day 15-17: Phase 5 (工程集成)
├── Day 18:   Phase 6 (测试优化)
├── Day 19:   Phase 7 (演示准备)
└── Day 20:   最终检查和提交 (06.28截止)
```

**总预估工时**: ~19个工作日 × 6小时/天 = **114小时**

---

## 开放问题与决策日志

### 开放问题（Open Questions）

| # | 问题 | 影响 | 建议 | 状态 |
|---|------|------|------|------|
| OQ1 | 是否需要实现CNN模型（而非MLP）？ | 架构复杂度增加 | 先用MLP，如果效果好就不需要CNN | 待定 |
| OQ2 | 是否需要使用MLflow进行实验管理？ | 额外依赖 | 可以先用JSON，后期升级MLflow | 待定 |
| OQ3 | Web界面是否需要部署到云端（Streamlit Cloud）？ | 演示便利性 | 本地演示即可，提供截图 | 待定 |
| OQ4 | 是否需要编写单元测试以达到高覆盖率？ | 时间成本 | 核心模块必须有测试，辅助模块可降低标准 | 已决定：核心>80%，总体>70% |
| OQ5 | 论文的详细程度如何？ | 写作时间 | 先完成代码，再根据时间决定论文深度 | 待定 |

### 决策日志（Decision Log）

| # | 日期 | 决策 | 理由 | 影响 |
|---|------|------|------|------|
| D1 | 2026-06-10 | 选择混合架构（传统ML+DL+进阶算法） | 平衡课程要求和展示效果 | 需要同时熟悉sklearn和PyTorch |
| D2 | 2026-06-10 | 选择Streamlit作为Web框架 | 快速原型、易于展示 | 需要学习Streamlit API |
| D3 | 2026-06-10 | 选择Optuna进行超参优化 | 易用、高效、可视化好 | 增加一个依赖 |
| D4 | 2026-06-10 | 选择LightGBM作为进阶算法 | 课堂上可能未讲、性能优秀、有特色 | 需要在论文中详细解释原理 |
| D5 | 2026-06-10 | 实现全部高级特性（超参优化+集成+可解释性+理论深度） | 时间充裕、争取额外加分 | 工作量增加约40% |

---

## 完成定义（Definition of Done）

### 通用DoD checklist

#### 功能完整性
- [ ] 所有Must Have（P0）需求都已实现
- [ ] 所有Should Have（P1）需求都已实现（或明确延期）
- [ ] 核心功能可通过CLI正常运行
- [ ] Web界面所有页面可访问且功能正常

#### 代码质量
- [ ] 代码无语法错误和明显的runtime error
- [ ] 符合PEP 8代码规范（通过Black检查）
- [ ] 核心模块有充分的单元测试（覆盖率>80%）
- [ ] 所有测试通过（`pytest tests/ -v`）
- [ ] 无硬编码的魔法数字或字符串（使用常量或配置）
- [ ] 关键函数有Docstring文档

#### 文档完整性
- [ ] README.md完整且专业（包含所有必需章节）
- [ ] API文档清晰（核心类和方法的Docstring）
- [ ] 配置文件有注释说明每个参数的含义
- [ ] Git提交信息规范（conventional commits）

#### 实验可复现性
- [ ] 固定随机种子（`random_state=42`）
- [ ] 完整的环境配置文件（`requirements.txt`）
- [ ] 详细的运行说明（README中的Quick Start）
- [ ] 实验记录完整（包含超参数、指标、时间戳）
- [ ] 他人可以按照README复现结果

#### 性能达标
- [ ] 传统ML训练时间 < 30秒
- [ ] 深度学习训练时间 < 5分钟
- [ ] Web界面首屏加载 < 3秒
- [ ] 内存占用 < 4GB（训练时）

#### 安全合规
- [ ] 无安全漏洞（依赖无已知CVE）
- [ ] 不包含敏感信息（密码、密钥等）
- [ ] `.gitignore`配置正确（排除敏感文件）
- [ ] 代码无明显的安全反模式（如exec、eval）

### 项目特定的DoD

#### 课程评分要求对照
- [ ] **数据分析（5%）**: ✅ 完整的数据质量报告 + EDA可视化
- [ ] **数据处理（30%）**: ✅ 完整的清洗和特征工程流程
- [ ] **多算法实验（30%）**: ✅ 4种算法 + 5维度对比 + 鲁棒性测试
- [ ] **工程集成（30%）**: ✅ CLI + Web界面 + GitHub专业展示
- [ ] **课程总结（5%）**: ✅ 学习内容回顾 + 评价建议

#### 额外加分项
- [ ] **现场演示准备**: ✅ Web界面 + 演示脚本
- [ ] **超参数优化**: ✅ Optuna集成 + 可视化
- [ ] **模型集成**: ✅ Voting + Stacking
- [ ] **理论深度**: ✅ 算法原理文档 + 数学推导
- [ ] **可解释性分析**: ✅ SHAP + LIME + Feature Importance

---

## 附录

### 附录A: 数据集详细信息

**数据集名称**: Dry Bean Dataset  
**来源**: UCI Machine Learning Repository  
**任务类型**: 多分类（Multi-class Classification）  
**类别数量**: 7种干豆品种

**类别列表**:
1. SEKER
2. BARBUNYA
3 BOMBAY
4. CALI
5. HOROZ
6. SIRA
7. DERMASON

**特征列表（16个）**:
1. Area (A): 面积
2. Perimeter (P): 周长
3. MajorAxisLength (L): 长轴长度
4. MinorAxisLength (l): 短轴长度
5. AspectRation (K): 长宽比
6. Eccentricity (Ec): 离心率
7. ConvexArea (C): 凸包面积
8. EquivDiameter (Ed): 等效直径
9. Extent (Ex): 占比
10. Solidity (S): 实心度
11. roundness (R): 圆度
12. Compactness (CO): 紧凑度
13. ShapeFactor1 (SF1): 形状因子1
14. ShapeFactor2 (SF2): 形状因子2
15. ShapeFactor3 (SF3): 形状因子3
16. ShapeFactor4 (SF4): 形状因子4

**数据划分**:
- 训练集: `Dry_Bean_Dataset_Dirty_train.csv` (~80%)
- 验证集: `Dry_Bean_Dataset_Dirty_val.csv` (~10%)
- 测试集: `Dry_Bean_Dataset_Dirty_test.csv` (~10%)

**已知数据质量问题**:
- 第9行: Solidity列缺失
- 第9行: 标签错误（D3RMAS0N 应为 DERMASON）
- 可能存在其他未发现的异常值和缺失值

---

### 附录B: 评分标准详细对照

| 评分项 | 占比 | 要求 | 我们的实现 | 状态 |
|--------|------|------|------------|------|
| 数据分析 | 5% | 主观观察和判断说明 | 数据质量报告 + EDA可视化 + 问题识别 | ✅ 已规划 |
| 数据处理 | 30% | 数据清洗和特征工程 | 完整Pipeline + 配置化 + 多策略 | ✅ 已规划 |
| 多算法实验 | 30% | ≥3种算法（含1种未讲过的）+ 5维度对比 | 4种算法 + 全面对比 + 鲁棒性测试 | ✅ 已规划 |
| 工程集成 | 30% | CLI + GitHub + README + 展示界面 | Click CLI + Streamlit + 专业GitHub | ✅ 已规划 |
| 课程总结 | 5% | 学习内容和评价建议 | 总结文档 + 反思 | ✅ 已规划 |
| **总计** | **100%** | | | **✅ 完全覆盖** |

**额外加分项**:
- 现场讲解: ✅ Web界面演示 + 演示脚本
- 高级特性: ✅ 超参优化 + 模型集成 + 可解释性 + 理论深度

---

### 附录C: 推荐资源

**官方文档**:
- Scikit-learn: https://scikit-learn.org/stable/
- PyTorch: https://pytorch.org/docs/stable/
- XGBoost: https://xgboost.readthedocs.io/
- LightGBM: https://lightgbm.readthedocs.io/
- Optuna: https://optuna.readthedocs.io/
- Streamlit: https://docs.streamlit.io/
- SHAP: https://shap.readthedocs.io/

**教程和书籍**:
- 《Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow》
- 《Python Machine Learning》- Sebastian Raschka
- 《Deep Learning with PyTorch》

**学术论文**:
- Original Dry Bean Dataset paper: Koklu, M., & Ozkan, I.A. (2020)
- LightGBM: Ke et al., "LightGBM: A Highly Efficient Gradient Boosting Decision Tree" (2017)
- SHAP: Lundberg & Lee, "A Unified Approach to Interpreting Model Predictions" (2017)

---

### 附录D: 术语表

| 术语 | 英文 | 解释 |
|------|------|------|
| EDA | Exploratory Data Analysis | 探索性数据分析 |
| MLP | Multi-Layer Perceptron | 多层感知机 |
| GBM | Gradient Boosting Machine | 梯度提升机 |
| SHAP | SHapley Additive exPlanations | SHapley加性解释 |
| LIME | Local Interpretable Model-agnostic Explanations | 局部可解释模型无关解释 |
| TPE | Tree-structured Parzen Estimator | 树结构Parzen估计器 |
| IQR | Interquartile Range | 四分位距 |
| OVR | One-vs-Rest | 一对多分类策略 |
| ROC | Receiver Operating Characteristic | 受试者工作特征曲线 |
| AUC | Area Under Curve | 曲线下面积 |
| CLI | Command Line Interface | 命令行接口 |
| API | Application Programming Interface | 应用程序编程接口 |
| DoD | Definition of Done | 完成定义 |
| PRD | Product Requirements Document | 产品需求文档 |
| SOTA | State-of-the-Art | 最先进水平 |
| CVE | Common Vulnerabilities and Exposures | 通用漏洞披露 |

---

### 附录E: 歧义日志（Unresolved Ambiguity Log）

| # | Section | Topic | Missing Info | Impact | Suggested Next Step |
|---|---------|-------|--------------|--------|---------------------|
| - | - | - | - | - | All ambiguities were resolved during drafting |

**说明**: 在grill-master访谈阶段，所有关键决策点都已经过讨论和确认。当前PRD文档中没有未解决的歧义项。如果在后续实施过程中发现新的歧义，将在此表中追加记录。

---

## 文档结束

**文档状态**: ✅ 已完成初稿，待用户审阅确认  
**下一步行动**: 
1. 用户审阅本PRD文档
2. 确认或提出修改意见
3. 确认后进入Spec规格说明书编写阶段
4. 然后开始分Phase实施

**最后更新**: 2026-06-10  
**下次评审**: 待定（用户确认后）