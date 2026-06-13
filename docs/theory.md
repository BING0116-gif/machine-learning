# LightGBM 算法原理详解

## 1. 概述

LightGBM (Light Gradient Boosting Machine) 是微软于2017年提出的一种梯度提升决策树(GBDT)算法框架。相比传统的GBDT算法（如XGBoost），LightGBM在训练速度和内存使用上有显著优势，同时在精度上保持竞争力。

### 核心创新点

1. **GOSS** (Gradient-based One-Side Sampling): 基于梯度的单边采样
2. **EFB** (Exclusive Feature Bundling): 互斥特征捆绑
3. **Leaf-wise生长策略**: 替代传统的Level-wise策略

## 2. 梯度提升决策树基础

### 2.1 GBDT基本原理

GBDT是一种加法模型，通过逐步添加决策树来减小残差。第$t$轮的预测值为：

$$\hat{y}_i^{(t)} = \hat{y}_i^{(t-1)} + f_t(x_i)$$

其中$f_t$是第$t$棵决策树。

### 2.2 目标函数

LightGBM的目标函数与XGBoost类似：

$$\mathcal{L}^{(t)} = \sum_{i=1}^{n} l(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)) + \Omega(f_t)$$

其中$l$是损失函数，$\Omega$是正则化项：

$$\Omega(f) = \gamma T + \frac{1}{2}\lambda \|\omega\|^2$$

$T$是叶子节点数，$\omega$是叶子权重向量。

### 2.3 二阶泰勒展开

对损失函数进行二阶泰勒展开：

$$\mathcal{L}^{(t)} \approx \sum_{i=1}^{n} \left[ g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i) \right] + \Omega(f_t)$$

其中：
- $g_i = \partial_{\hat{y}^{(t-1)}} l(y_i, \hat{y}^{(t-1)})$ 是一阶梯度
- $h_i = \partial_{\hat{y}^{(t-1)}}^2 l(y_i, \hat{y}^{(t-1)})$ 是二阶梯度（Hessian）

## 3. GOSS: 基于梯度的单边采样

### 3.1 动机

在GBDT中，梯度较小的样本已经得到了较好的训练，而梯度较大的样本还有较大的优化空间。因此，我们可以：

- **保留**所有大梯度的样本
- **随机采样**小梯度的样本

### 3.2 算法流程

1. 将样本按梯度绝对值降序排列
2. 选取前$a \times 100\%$的大梯度样本
3. 从剩余样本中随机采样$b \times 100\%$
4. 对小梯度样本乘以系数$\frac{1-a}{b}$进行放大，以保持数据分布不变

### 3.3 数学证明

设$I$为全部样本集，$I_{top}$为大梯度样本，$I_{rand}$为随机采样的小梯度样本。

传统方法计算方差增益：

$$V_{j}(d) = \frac{1}{n} \left( \frac{(\sum_{x_i \in I_l} g_i)^2}{n_l} + \frac{(\sum_{x_i \in I_r} g_i)^2}{n_r} \right)$$

GOSS估计方差增益：

$$\tilde{V}_{j}(d) = \frac{1}{n} \left( \frac{(\sum_{x_i \in I_{l,top}} g_i + \frac{1-a}{b} \sum_{x_i \in I_{l,rand}} g_i)^2}{n_l} + \frac{(\sum_{x_i \in I_{r,top}} g_i + \frac{1-a}{b} \sum_{x_i \in I_{r,rand}} g_i)^2}{n_r} \right)$$

可以证明，GOSS的估计方差有上界：

$$\mathbb{E}[\tilde{V}(d)] \leq (1 + \frac{1}{b})^2 V(d)$$

### 3.4 优势

- 训练速度提升约2倍（仅使用部分样本）
- 精度损失极小（大梯度样本全部保留）
- 比随机采样更稳定

## 4. EFB: 互斥特征捆绑

### 4.1 动机

在稀疏特征空间中，许多特征是互斥的（不会同时取非零值）。将这些互斥特征捆绑为一个特征，可以减少特征数量，从而降低计算复杂度。

### 4.2 算法流程

1. **构建特征冲突图**: 两个特征同时取非零值的样本数为冲突数
2. **图着色问题**: 将冲突数小于阈值的特征分配相同颜色（可捆绑）
3. **特征合并**: 将同一颜色组的特征合并为一个特征

### 4.3 合并策略

对于互斥特征$i$和$j$，创建新特征$k$：

$$x_k = x_i \times offset + x_j$$

其中$offset$是特征$i$的取值范围上界，确保可以从合并特征中恢复原始值。

### 4.4 复杂度分析

- 原始特征数: $M$
- 捆绑后特征数: $M_{bundle}$
- 特征直方图构建复杂度: $O(M_{bundle} \times n) \ll O(M \times n)$

## 5. Leaf-wise vs Level-wise

### 5.1 Level-wise (XGBoost默认)

按层生长，每层分裂所有叶子节点：

```
        [Root]          Level 0
       /      \
    [L]        [R]      Level 1
   /   \      /   \
  [L]  [R]  [L]  [R]   Level 2
```

- 优点: 不容易过拟合
- 缺点: 效率低（某些叶子节点分裂增益很小）

### 5.2 Leaf-wise (LightGBM默认)

按叶子增益最大优先生长：

```
        [Root]
       /      \
    [L]        [R]      ← 选择增益最大的叶子继续分裂
   /   \
  [L]  [R]              ← 只分裂增益最大的叶子
 /
[L]                     ← 深度可能不均匀
```

- 优点: 收敛更快，精度更高
- 缺点: 容易过拟合（需要限制max_depth或num_leaves）

### 5.3 防过拟合策略

LightGBM通过以下参数控制Leaf-wise的过拟合：
- `num_leaves`: 最大叶子数
- `max_depth`: 最大深度
- `min_data_in_leaf`: 叶子最小样本数
- `lambda_l1`, `lambda_l2`: L1/L2正则化

## 6. 直方图加速

### 6.1 算法

LightGBM将连续特征值离散化为直方图：

1. 将连续特征值分桶（默认256个桶）
2. 统计每个桶的梯度和与样本数
3. 在直方图上寻找最优分裂点

### 6.2 复杂度

- 传统方法: $O(n \times M)$ 每个特征
- 直方图方法: $O(k \times M)$ 其中$k$是桶数（$k \ll n$）

### 6.3 直方图差加速

父节点的直方图 = 左子节点直方图 + 右子节点直方图

只需计算一个子节点的直方图，另一个通过减法得到，减少一半计算量。

## 7. 与XGBoost的对比

| 特性 | LightGBM | XGBoost |
|------|----------|---------|
| 树生长策略 | Leaf-wise | Level-wise |
| 采样策略 | GOSS | 随机采样 |
| 特征降维 | EFB | 无 |
| 分裂点搜索 | 直方图 | 直方图/精确 |
| 训练速度 | 更快 | 较慢 |
| 内存使用 | 更少 | 较多 |
| 过拟合风险 | 较高（需调参） | 较低 |
| 类别特征支持 | 原生支持 | 需编码 |

## 8. 多分类扩展

对于$K$类分类问题，LightGBM使用One-vs-Rest策略：

- 训练$K$组提升树，每组对应一个类别
- 每组的目标是区分该类别与其他类别
- 最终预测使用Softmax：

$$P(y=k|x) = \frac{e^{f_k(x)}}{\sum_{j=1}^{K} e^{f_j(x)}}$$

## 9. 参考文献

1. Ke, G., et al. "LightGBM: A Highly Efficient Gradient Boosting Decision Tree." Advances in Neural Information Processing Systems (NIPS), 2017.
2. Friedman, J.H. "Greedy Function Approximation: A Gradient Boosting Machine." Annals of Statistics, 2001.
3. Chen, T. and Guestrin, C. "XGBoost: A Scalable Tree Boosting System." KDD, 2016.
