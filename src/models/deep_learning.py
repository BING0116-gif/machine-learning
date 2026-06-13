"""深度学习模型模块

实现基于PyTorch的MLP分类器。
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import time
import os

from src.utils.logger import get_logger
from .base import BaseModel

logger = get_logger("dry_bean.models.deep_learning")


class MLPNet(nn.Module):
    """多层感知机网络

    架构: Input → Linear → BN → ReLU → Dropout → ... → Linear(Output)
    """

    def __init__(self, input_dim: int, num_classes: int,
                 hidden_dims: List[int] = None,
                 dropout_rate: float = 0.3,
                 use_batch_norm: bool = True):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))

        self.network = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self):
        """He初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class PyTorchModel(BaseModel):
    """PyTorch模型包装器

    将PyTorch模型封装为统一接口。
    """

    DEFAULT_CONFIG = {
        'hidden_dims': [256, 128, 64],
        'dropout_rate': 0.3,
        'epochs': 100,
        'batch_size': 32,
        'learning_rate': 0.001,
        'optimizer': 'adam',
        'weight_decay': 1e-4,
        'scheduler': 'cosine',
        'early_stopping_patience': 15,
        'device': 'auto',
    }

    def __init__(self, model_type: str = 'mlp',
                 config: Dict = None,
                 hyperparams: Dict = None):
        super().__init__(f'pytorch_{model_type}', hyperparams)
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}
        self.model_type = model_type
        self.model = None
        self.device = self._get_device()
        self.label_encoder = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.input_dim = None

    def _get_device(self) -> torch.device:
        """获取计算设备"""
        device_str = self.config.get('device', 'auto')
        if device_str == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device_str)

    def _build_model(self, input_dim: int, num_classes: int):
        """构建模型架构"""
        if self.model_type == 'mlp':
            self.model = MLPNet(
                input_dim=input_dim,
                num_classes=num_classes,
                hidden_dims=self.config.get('hidden_dims', [256, 128, 64]),
                dropout_rate=self.config.get('dropout_rate', 0.3),
            ).to(self.device)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        self.input_dim = input_dim
        self.criterion = nn.CrossEntropyLoss()

        if self.config.get('optimizer', 'adam') == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.get('learning_rate', 0.001),
                weight_decay=self.config.get('weight_decay', 1e-4),
            )
        else:
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config.get('learning_rate', 0.001),
                weight_decay=self.config.get('weight_decay', 1e-4),
            )

        if self.config.get('scheduler') == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.config.get('epochs', 100)
            )
        elif self.config.get('scheduler') == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=30, gamma=0.1
            )

    def _prepare_dataloader(self, X: pd.DataFrame, y: pd.Series = None) -> DataLoader:
        """准备DataLoader"""
        X_array = np.array(X.values if isinstance(X, pd.DataFrame) else X, dtype=np.float32)
        X_tensor = torch.FloatTensor(X_array).to(self.device)

        if y is not None:
            from sklearn.preprocessing import LabelEncoder
            if self.label_encoder is None:
                self.label_encoder = LabelEncoder()
                y_encoded = self.label_encoder.fit_transform(y)
            else:
                y_encoded = self.label_encoder.transform(y)
            y_tensor = torch.LongTensor(y_encoded).to(self.device)
            dataset = TensorDataset(X_tensor, y_tensor)
            shuffle = True
        else:
            dataset = TensorDataset(X_tensor)
            shuffle = False

        return DataLoader(
            dataset,
            batch_size=self.config.get('batch_size', 32),
            shuffle=shuffle,
            num_workers=0,
            drop_last=False,
        )

    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: pd.DataFrame = None, y_val: pd.Series = None) -> Dict:
        """训练PyTorch模型"""
        start_time = time.time()

        from sklearn.preprocessing import LabelEncoder
        self.label_encoder = LabelEncoder()
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        num_classes = len(self.label_encoder.classes_)

        # 构建模型
        input_dim = X_train.shape[1] if isinstance(X_train, pd.DataFrame) else X_train.shape[1]
        self._build_model(input_dim, num_classes)

        # 准备数据
        train_loader = self._prepare_dataloader(X_train, y_train)
        val_loader = self._prepare_dataloader(X_val, y_val) if X_val is not None else None

        # 训练循环
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
        }

        best_val_acc = 0
        patience_counter = 0
        best_model_state = None
        epochs = self.config.get('epochs', 100)
        patience = self.config.get('early_stopping_patience', 15)

        for epoch in range(epochs):
            # 训练
            self.model.train()
            total_loss = 0
            correct = 0
            total = 0

            for batch_data in train_loader:
                batch_x, batch_y = batch_data
                self.optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item() * batch_x.size(0)
                _, predicted = outputs.max(1)
                total += batch_y.size(0)
                correct += predicted.eq(batch_y).sum().item()

            train_loss = total_loss / total
            train_acc = correct / total
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)

            # 验证
            if val_loader is not None:
                self.model.eval()
                val_loss = 0
                val_correct = 0
                val_total = 0

                with torch.no_grad():
                    for batch_data in val_loader:
                        batch_x, batch_y = batch_data
                        outputs = self.model(batch_x)
                        loss = self.criterion(outputs, batch_y)
                        val_loss += loss.item() * batch_x.size(0)
                        _, predicted = outputs.max(1)
                        val_total += batch_y.size(0)
                        val_correct += predicted.eq(batch_y).sum().item()

                val_loss = val_loss / val_total
                val_acc = val_correct / val_total
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)

                # 学习率调度
                if self.scheduler:
                    self.scheduler.step()

                # Early Stopping
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                    best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info(f"Early stopping at epoch {epoch+1}")
                        break

            # 日志
            if (epoch + 1) % 10 == 0 or epoch == 0:
                log_msg = f"Epoch [{epoch+1}/{epochs}] Loss: {train_loss:.4f}, Acc: {train_acc:.4f}"
                if val_loader is not None:
                    log_msg += f" | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
                logger.info(log_msg)

        # 恢复最佳模型
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            self.model.to(self.device)

        training_time = time.time() - start_time

        self.is_fitted = True
        self.training_history = {
            'train_accuracy': float(history['train_acc'][-1]),
            'val_accuracy': float(history['val_acc'][-1]) if history['val_acc'] else None,
            'training_time_seconds': training_time,
            'total_epochs': len(history['train_loss']),
            'history': {k: [float(v) for v in vals] for k, vals in history.items()},
        }

        val_acc_str = f"{history['val_acc'][-1]:.4f}" if history['val_acc'] else "N/A"
        logger.info(f"MLP 训练完成: train_acc={history['train_acc'][-1]:.4f}, "
                    f"val_acc={val_acc_str}, "
                    f"time={training_time:.2f}s")
        return self.training_history

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """预测类别"""
        if not self.is_fitted:
            raise RuntimeError("Model not trained yet.")

        self.model.eval()
        X_array = np.array(X.values if isinstance(X, pd.DataFrame) else X, dtype=np.float32)
        X_tensor = torch.FloatTensor(X_array).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = outputs.max(1)
            all_preds = predicted.cpu().numpy()

        return self.label_encoder.inverse_transform(all_preds)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """预测概率"""
        if not self.is_fitted:
            raise RuntimeError("Model not trained yet.")

        self.model.eval()
        X_array = np.array(X.values if isinstance(X, pd.DataFrame) else X, dtype=np.float32)
        X_tensor = torch.FloatTensor(X_array).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_tensor)
            probs = torch.softmax(outputs, dim=1)
            return probs.cpu().numpy()

    def save_model(self, path: str):
        """保存模型"""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_type': self.model_type,
            'config': self.config,
            'input_dim': self.input_dim,
            'label_encoder_classes': self.label_encoder.classes_.tolist() if self.label_encoder else None,
            'training_history': self.training_history,
            'hyperparams': self.hyperparams,
        }, path)
        logger.info(f"模型已保存: {path}")

    @classmethod
    def load_model(cls, path: str) -> 'PyTorchModel':
        """加载模型"""
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)

        instance = cls(
            model_type=checkpoint['model_type'],
            config=checkpoint['config'],
            hyperparams=checkpoint['hyperparams'],
        )

        from sklearn.preprocessing import LabelEncoder
        instance.label_encoder = LabelEncoder()
        instance.label_encoder.classes_ = np.array(checkpoint['label_encoder_classes'])

        num_classes = len(checkpoint['label_encoder_classes'])
        input_dim = checkpoint.get('input_dim', 16)
        instance._build_model(input_dim, num_classes)
        instance.model.load_state_dict(checkpoint['model_state_dict'])
        instance.training_history = checkpoint['training_history']
        instance.is_fitted = True
        instance.input_dim = input_dim

        return instance
