import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch.utils.tensorboard import SummaryWriter
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib
from datetime import datetime
from dataset import get_data_loaders

class TitanicSVM:
    def __init__(self, kernel='rbf', random_state=42):
        self.model = SVC(
            kernel=kernel,
            random_state=random_state,
            class_weight="balanced"
        )
        self.model_name = 'svm'

    def train(self, train_loader, test_loader, batch_size=32):
        # Create TensorBoard writer
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = f'logs/{self.model_name}/batch_{batch_size}/{current_time}'
        writer = SummaryWriter(log_dir)

        print(f"\n[INFO] Training SVM with batch size {batch_size}")

        # Extract dataset from PyTorch DataLoaders (needed for sklearn)
        X_train = torch.cat([batch[0] for batch in train_loader]).numpy()
        y_train = torch.cat([batch[1] for batch in train_loader]).numpy()
        X_test = torch.cat([batch[0] for batch in test_loader]).numpy()
        y_test = torch.cat([batch[1] for batch in test_loader]).numpy()

        # Train SVM model
        self.model.fit(X_train, y_train)

        # Compute accuracy
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)

        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)

        # Log metrics to TensorBoard
        writer.add_scalar(f'Accuracy/train_batch_{batch_size}', train_acc, 0)
        writer.add_scalar(f'Accuracy/test_batch_{batch_size}', test_acc, 0)
        writer.close()

        print(f"[INFO] Training complete. Logs saved in {log_dir}")
        return train_acc, test_acc

    def evaluate(self, test_loader):
        X_test = torch.cat([batch[0] for batch in test_loader]).numpy()
        y_test = torch.cat([batch[1] for batch in test_loader]).numpy()
        predictions = self.model.predict(X_test)
        return classification_report(y_test, predictions)

    def save_model(self, filepath="models"):
        os.makedirs(filepath, exist_ok=True)
        joblib.dump(self.model, f"{filepath}/{self.model_name}.joblib")

# ================================
# 2. Training Loop with Different Batch Sizes
# ================================
def main():

    # Load dataset
    train_loader, test_loader, input_dim = get_data_loaders("../dataset/dataset.csv")

    # Initialize model
    svm_model = TitanicSVM()

    # Train the model with specific parameters
    train_acc, test_acc = svm_model.train(train_loader, test_loader)

    # Evaluate the model
    print("\nFinal Model Evaluation on Test Data:")
    print(svm_model.evaluate(test_loader))

    # Save the model for each batch size
    model_name = f"models/svm_model.joblib"
    svm_model.save_model(model_name)
    print(f"Model saved to {model_name}\n")

if __name__ == "__main__":
    main()
