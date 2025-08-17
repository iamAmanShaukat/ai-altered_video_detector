import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, roc_auc_score, f1_score, confusion_matrix
from collections import Counter
import numpy as np
import logging
from pathlib import Path

class Trainer:
    def __init__(self, config, model, train_loader=None, val_loader=None):
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device(config["training"]["device"])
        self.criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(config["training"]["class_weights"], dtype=torch.float32, device=self.device)
        )
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"]
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=config["training"]["scheduler_factor"],
            patience=config["training"]["scheduler_patience"]
        )
        self.early_stopping_patience = config["training"]["early_stopping_patience"]
        self.early_stopping_min_delta = config["training"]["early_stopping_min_delta"]
        self.best_val_accuracy = 0.0
        self.epochs_no_improve = 0
        self.checkpoint_dir = Path(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), config["training"]["checkpoint_dir"]))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)

    def save_checkpoint(self, epoch, val_accuracy):
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_accuracy': val_accuracy
        }, checkpoint_path)
        self.logger.info(f"Saved checkpoint: {checkpoint_path}")
        # Also save as best_epoch.pth if this is the best
        best_path = self.checkpoint_dir / "best_epoch.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_accuracy': val_accuracy
        }, best_path)
        self.logger.info(f"Saved best checkpoint: {best_path}")

    def train(self):
        self.model.train()
        self.best_val_accuracy = 0.0
        self.best_epoch = -1
        self.epochs_no_improve = 0
        for epoch in range(self.config["training"]["epochs"]):
            self.model.train()
            total_loss = 0
            for batch in self.train_loader:
                data, labels = batch
                data = {k: v.to(self.device) for k, v in data.items()}
                labels = labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(data)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            val_accuracy = self.evaluate(self.val_loader)
            is_best = val_accuracy > (self.best_val_accuracy + self.early_stopping_min_delta)
            if is_best:
                self.best_val_accuracy = val_accuracy
                self.epochs_no_improve = 0
                self.save_checkpoint(epoch, val_accuracy)
                self.logger.info(f"New best epoch: {epoch} (Val Acc: {val_accuracy:.2f}%)")
            else:
                self.epochs_no_improve += 1
                self.logger.info(f"Epoch {epoch}, Loss: {total_loss:.4f}, Val Accuracy: {val_accuracy:.2f}% (No significant improvement over best {self.best_val_accuracy:.2f}%)")
            if self.epochs_no_improve >= self.early_stopping_patience:
                self.logger.info(f"Early stopping triggered after {epoch + 1} epochs (patience: {self.early_stopping_patience}). Best Val Accuracy: {self.best_val_accuracy:.2f}%")
                break

    def evaluate(self, val_loader):
        self.model.eval()
        correct = 0
        total = 0
        predictions = []
        true_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                data, labels = batch
                data = {k: v.to(self.device) for k, v in data.items()}
                labels = labels.to(self.device)
                outputs = self.model(data)
                
                # Get probabilities for AUC-ROC
                probabilities = torch.softmax(outputs, dim=1)
                all_probabilities.extend(probabilities.cpu().numpy())
                
                # Get predictions for accuracy and F1
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                predictions.extend(predicted.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
                
                if batch_idx == 0:
                    self.logger.info(f"Val batch 0 predictions: {predicted[:20].cpu().numpy()}")
                    self.logger.info(f"Val batch 0 true labels: {labels[:20].cpu().numpy()}")
        
        # Calculate metrics
        accuracy = 100. * correct / total
        
        # Convert to numpy arrays for sklearn metrics
        true_labels = np.array(true_labels)
        predictions = np.array(predictions)
        all_probabilities = np.array(all_probabilities)
        
        # Calculate F1-score (macro average for multi-class)
        f1_macro = f1_score(true_labels, predictions, average='macro')
        f1_weighted = f1_score(true_labels, predictions, average='weighted')
        
        # Calculate AUC-ROC (one-vs-rest for multi-class)
        try:
            auc_roc = roc_auc_score(true_labels, all_probabilities, multi_class='ovr', average='macro')
        except ValueError as e:
            self.logger.warning(f"Could not calculate AUC-ROC: {e}")
            auc_roc = None
        
        # Calculate per-class F1 scores
        f1_per_class = f1_score(true_labels, predictions, average=None)
        
        # Log comprehensive metrics
        self.logger.info("=" * 60)
        self.logger.info("EVALUATION METRICS")
        self.logger.info("=" * 60)
        self.logger.info(f"Accuracy: {accuracy:.2f}%")
        self.logger.info(f"F1-Score (Macro): {f1_macro:.4f}")
        self.logger.info(f"F1-Score (Weighted): {f1_weighted:.4f}")
        if auc_roc is not None:
            self.logger.info(f"AUC-ROC (Macro): {auc_roc:.4f}")
        else:
            self.logger.info("AUC-ROC: Could not calculate")
        
        # Per-class F1 scores
        class_names = list(self.config["data"]["class_map"].keys())
        self.logger.info("\nPer-Class F1 Scores:")
        for i, (class_name, f1) in enumerate(zip(class_names, f1_per_class)):
            self.logger.info(f"  {class_name}: {f1:.4f}")
        
        # Confusion Matrix
        cm = confusion_matrix(true_labels, predictions)
        self.logger.info(f"\nConfusion Matrix:\n{cm}")
        
        # Class distribution
        self.logger.info(f"\nClass Distribution: {Counter(true_labels)}")
        
        # Detailed classification report
        self.logger.info("\nDetailed Classification Report:")
        self.logger.info(classification_report(true_labels, predictions, target_names=class_names))
        
        self.logger.info("=" * 60)
        
        return accuracy