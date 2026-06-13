"""Dry Bean Classification - Streamlit Dashboard

交互式Dashboard，展示数据分析、模型对比和可解释性分析结果。

启动方式:
    streamlit run app.py --server.port 8501
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
import joblib

# 页面配置
st.set_page_config(
    page_title="Dry Bean Classification",
    page_icon="🫘",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def load_raw_data():
    """加载原始数据"""
    try:
        from src.data.loader import DataLoader
        loader = DataLoader('DryBeanDataset')
        return loader.load_all()
    except Exception as e:
        st.warning(f"无法加载数据: {e}")
        return None


@st.cache_data
def load_processed_data():
    """加载处理后的数据"""
    try:
        data = {}
        for split in ['train', 'val', 'test']:
            path = f'data/processed/{split}_processed.pkl'
            if os.path.exists(path):
                data[split] = joblib.load(path)
        return data if data else None
    except Exception:
        return None


@st.cache_data
def load_experiment_results():
    """加载实验结果"""
    results = {}
    exp_dir = 'results/experiments'
    if os.path.exists(exp_dir):
        for exp_file in glob.glob(os.path.join(exp_dir, '*.json')):
            try:
                import json
                with open(exp_file, 'r', encoding='utf-8') as f:
                    exp_data = json.load(f)
                model_name = exp_data.get('model_name', 'unknown')
                if model_name not in results or exp_data.get('status') == 'completed':
                    results[model_name] = exp_data
            except Exception:
                continue
    return results


def render_sidebar():
    """渲染侧边栏"""
    st.sidebar.title("🫘 Dry Bean Classification")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "导航",
        ["🏠 首页", "📊 数据探索", "🤖 模型对比", "🔍 可解释性", "📖 关于"],
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 项目信息")
    st.sidebar.info(
        "课程: 2026_AIT209 机器学习\n"
        "数据集: Dry Bean Dataset\n"
        "类别数: 7\n"
        "特征数: 16"
    )

    return page


def render_homepage():
    """渲染首页仪表盘"""
    st.title("🫘 Dry Bean Classification Dashboard")
    st.markdown("基于 Dry Bean Dataset 的多分类机器学习系统")

    # 加载数据
    raw_data = load_raw_data()
    exp_results = load_experiment_results()

    # 关键指标卡片
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_samples = sum(len(df) for df in raw_data.values()) if raw_data else 0
        st.metric("总样本数", f"{total_samples:,}")

    with col2:
        st.metric("特征数", "16")

    with col3:
        st.metric("类别数", "7")

    with col4:
        if exp_results:
            best_model = max(
                exp_results.items(),
                key=lambda x: x[1].get('metrics', {}).get('val_accuracy', 0) or 0
            )
            best_acc = best_model[1].get('metrics', {}).get('val_accuracy', 0)
            st.metric("最佳验证精度", f"{best_acc:.4f}" if best_acc else "N/A")
        else:
            st.metric("最佳验证精度", "待训练")

    st.markdown("---")

    # 模型性能概览
    if exp_results:
        st.subheader("模型性能概览")

        overview_data = []
        for model_name, exp_data in exp_results.items():
            metrics = exp_data.get('metrics', {})
            overview_data.append({
                '模型': model_name,
                '训练准确率': f"{metrics.get('train_accuracy', 0):.4f}" if metrics.get('train_accuracy') else 'N/A',
                '验证准确率': f"{metrics.get('val_accuracy', 0):.4f}" if metrics.get('val_accuracy') else 'N/A',
                '训练时间': f"{metrics.get('training_time_seconds', 0):.2f}s" if metrics.get('training_time_seconds') else 'N/A',
            })

        if overview_data:
            st.dataframe(pd.DataFrame(overview_data), use_container_width=True)
    else:
        st.info("暂无实验结果。请先运行 `python main.py train --model random_forest` 训练模型。")

    # 快速导航
    st.markdown("---")
    st.subheader("快速开始")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.code("python main.py analyze --data-dir DryBeanDataset", language="bash")
        st.caption("数据分析")
    with col2:
        st.code("python main.py process --config config/data_processing.yaml", language="bash")
        st.caption("数据预处理")
    with col3:
        st.code("python main.py train --model random_forest", language="bash")
        st.caption("模型训练")


def render_data_explorer():
    """渲染数据探索页面"""
    st.title("📊 数据探索")

    raw_data = load_raw_data()
    if raw_data is None:
        st.error("无法加载数据，请检查数据目录")
        return

    # 数据集选择
    split = st.selectbox("选择数据集", ["train", "val", "test"], index=0)
    df = raw_data[split]

    # 数据预览
    st.subheader("数据预览")
    st.dataframe(df.head(20), use_container_width=True)

    # 基本统计
    st.subheader("基本统计")
    st.dataframe(df.describe(), use_container_width=True)

    # 类别分布
    st.subheader("类别分布")

    col1, col2 = st.columns(2)
    with col1:
        class_counts = df['Class'].value_counts()
        st.bar_chart(class_counts)

    with col2:
        st.dataframe(
            pd.DataFrame({
                '类别': class_counts.index,
                '数量': class_counts.values,
                '占比': (class_counts.values / len(df) * 100).round(2),
            }),
            use_container_width=True,
        )

    # 特征分布
    st.subheader("特征分布")
    feature_cols = [col for col in df.columns if col != 'Class']
    selected_feature = st.selectbox("选择特征", feature_cols, index=0)

    if selected_feature:
        import plotly.express as px
        fig = px.histogram(df, x=selected_feature, color='Class',
                          marginal='box', opacity=0.7,
                          title=f'{selected_feature} 分布')
        st.plotly_chart(fig, use_container_width=True)

    # 相关性矩阵
    st.subheader("相关性矩阵")
    if st.checkbox("显示相关性矩阵"):
        corr = df[feature_cols].corr()
        import plotly.express as px
        fig = px.imshow(corr, text_auto='.2f', aspect='auto',
                       title='特征相关性矩阵', color_continuous_scale='RdBu_r')
        st.plotly_chart(fig, use_container_width=True)

    # 缺失值
    st.subheader("缺失值分析")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        st.warning(f"发现缺失值:")
        st.dataframe(pd.DataFrame({
            '列名': missing.index,
            '缺失数量': missing.values,
            '缺失比例': (missing.values / len(df) * 100).round(2),
        }), use_container_width=True)
    else:
        st.success("无缺失值")


def render_model_comparison():
    """渲染模型对比页面"""
    st.title("🤖 模型对比")

    exp_results = load_experiment_results()
    if not exp_results:
        st.info("暂无实验结果。请先训练模型。")
        st.code("python main.py train --model random_forest", language="bash")
        return

    # 模型选择
    available_models = list(exp_results.keys())
    selected_models = st.multiselect(
        "选择要对比的模型",
        available_models,
        default=available_models[:4] if len(available_models) >= 4 else available_models,
    )

    if not selected_models:
        st.warning("请至少选择一个模型")
        return

    # 精度对比表
    st.subheader("精度对比")

    comparison_data = []
    for model_name in selected_models:
        metrics = exp_results[model_name].get('metrics', {})
        comparison_data.append({
            '模型': model_name,
            '训练准确率': metrics.get('train_accuracy', 0),
            '验证准确率': metrics.get('val_accuracy', 0),
            '训练时间(s)': metrics.get('training_time_seconds', 0),
        })

    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True)

    # 精度对比图
    st.subheader("精度对比图")
    import plotly.express as px

    fig = px.bar(comparison_df, x='模型', y=['训练准确率', '验证准确率'],
                barmode='group', title='模型精度对比')
    st.plotly_chart(fig, use_container_width=True)

    # 训练曲线（如果有）
    st.subheader("训练曲线")

    for model_name in selected_models:
        metrics = exp_results[model_name].get('metrics', {})
        history = metrics.get('history', {})

        if history and ('train_loss' in history or 'train_acc' in history):
            st.markdown(f"**{model_name}**")

            col1, col2 = st.columns(2)

            if 'train_loss' in history:
                with col1:
                    loss_df = pd.DataFrame({
                        'Epoch': range(1, len(history['train_loss']) + 1),
                        'Train Loss': history['train_loss'],
                    })
                    if 'val_loss' in history and history['val_loss']:
                        loss_df['Val Loss'] = history['val_loss']
                    fig = px.line(loss_df, x='Epoch', y=loss_df.columns[1:],
                                 title=f'{model_name} - Loss Curve')
                    st.plotly_chart(fig, use_container_width=True)

            if 'train_acc' in history:
                with col2:
                    acc_df = pd.DataFrame({
                        'Epoch': range(1, len(history['train_acc']) + 1),
                        'Train Accuracy': history['train_acc'],
                    })
                    if 'val_acc' in history and history['val_acc']:
                        acc_df['Val Accuracy'] = history['val_acc']
                    fig = px.line(acc_df, x='Epoch', y=acc_df.columns[1:],
                                 title=f'{model_name} - Accuracy Curve')
                    st.plotly_chart(fig, use_container_width=True)


def render_interpretability():
    """渲染可解释性页面"""
    st.title("🔍 模型可解释性")

    exp_results = load_experiment_results()
    if not exp_results:
        st.info("暂无实验结果。请先训练模型。")
        return

    # 模型选择
    model_name = st.selectbox("选择模型", list(exp_results.keys()))

    metrics = exp_results[model_name].get('metrics', {})

    # 特征重要性
    st.subheader("特征重要性")

    feature_importances = metrics.get('feature_importances', None)
    if feature_importances:
        from src.data.loader import DataLoader
        feature_names = DataLoader.FEATURE_COLUMNS

        importance_df = pd.DataFrame({
            '特征': feature_names[:len(feature_importances)],
            '重要性': feature_importances,
        }).sort_values('重要性', ascending=False)

        import plotly.express as px
        fig = px.bar(importance_df, x='重要性', y='特征', orientation='h',
                    title=f'{model_name} - 特征重要性')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("该模型暂无特征重要性数据")

    # SHAP分析
    st.subheader("SHAP分析")
    st.info("SHAP分析图表可在运行实验后查看。使用 `python main.py evaluate` 生成。")

    # 检查是否有SHAP图片
    shap_dir = 'results/figures/interpretability'
    if os.path.exists(shap_dir):
        shap_images = glob.glob(os.path.join(shap_dir, '*.png'))
        if shap_images:
            for img_path in shap_images:
                st.image(img_path, caption=os.path.basename(img_path))


def render_about():
    """渲染关于页面"""
    st.title("📖 关于项目")

    st.markdown("""
    ## Dry Bean Classification 项目

    本项目是 **2026_AIT209 机器学习** 课程的期末大作业，基于 Dry Bean Dataset 构建多分类机器学习系统。

    ### 项目特点

    - **完整的数据分析流程**: 数据质量评估、EDA可视化、数据清洗、特征工程
    - **多算法对比**: Random Forest、XGBoost、MLP、LightGBM
    - **全面评估**: 精度、推理速度、鲁棒性、可解释性
    - **工程化架构**: CLI接口、Web界面、模块化代码

    ### 技术栈

    | 类别 | 技术 |
    |------|------|
    | 语言 | Python 3.9+ |
    | 传统ML | scikit-learn, XGBoost, LightGBM |
    | 深度学习 | PyTorch |
    | 可视化 | Matplotlib, Seaborn, Plotly |
    | Web框架 | Streamlit |
    | CLI | Click |
    | 超参数优化 | Optuna |
    | 可解释性 | SHAP, LIME |

    ### 数据集信息

    - **名称**: Dry Bean Dataset
    - **来源**: UCI Machine Learning Repository
    - **样本数**: 13,611
    - **特征数**: 16
    - **类别数**: 7 (SEKER, BARBUNYA, BOMBAY, CALI, HOROZ, SIRA, DERMASON)

    ### 使用方式

    ```bash
    # 数据分析
    python main.py analyze --data-dir DryBeanDataset

    # 数据预处理
    python main.py process

    # 训练模型
    python main.py train --model random_forest
    python main.py train --model xgboost
    python main.py train --model mlp
    python main.py train --model lightgbm

    # 运行所有实验
    python main.py run-all

    # 启动Web界面
    python main.py serve
    ```
    """)


# 主函数
def main():
    page = render_sidebar()

    if page == "🏠 首页":
        render_homepage()
    elif page == "📊 数据探索":
        render_data_explorer()
    elif page == "🤖 模型对比":
        render_model_comparison()
    elif page == "🔍 可解释性":
        render_interpretability()
    elif page == "📖 关于":
        render_about()


if __name__ == "__main__":
    main()
