import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import classification_report
from datetime import datetime
from dataset import get_data_loaders
from visualization.confusion_matrix import plot_confusion_matrix
import time

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)  # Single-layer logistic regression
        
    def forward(self, x):
        return torch.sigmoid(self.linear(x))  # Sigmoid activation

class LogisticRegressionTrainer:
    def __init__(self, input_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LogisticRegressionModel(input_dim).to(self.device)
        self.model_name = 'logistic_regression'
        self.criterion = nn.BCELoss()  # Binary classification loss
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.3)

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss, correct, total = 0, 0, 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Forward pass
            outputs = self.model(inputs).squeeze()
            loss = self.criterion(outputs, targets)
            
            # Backward pass & optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Metrics calculation
            total_loss += loss.item()
            predicted = (outputs > 0.5).float()
            correct += (predicted == targets).sum().item()
            total += targets.size(0)

        return total_loss / len(train_loader), correct / total

    def evaluate_epoch(self, test_loader):
        self.model.eval()
        total_loss, correct, total = 0, 0, 0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs).squeeze()
                loss = self.criterion(outputs, targets)
                
                # Metrics calculation
                total_loss += loss.item()
                predicted = (outputs > 0.5).float()
                correct += (predicted == targets).sum().item()
                total += targets.size(0)

        return total_loss / len(test_loader), correct / total
    
    def train(self, train_loader, test_loader, epochs=50, batch_size=32):
        # Create TensorBoard writer
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = f'logs/{self.model_name}/epochs_{epochs}_batch_{batch_size}/{current_time}'
        writer = SummaryWriter(log_dir)

        print(f"\n[INFO] Training logistic regression for {epochs} epochs with batch size {batch_size}")

        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            test_loss, test_acc = self.evaluate_epoch(test_loader)

            # Log metrics
            writer.add_scalar(f'Loss/train_epochs_{epochs}_batch_{batch_size}', train_loss, epoch)
            writer.add_scalar(f'Loss/test_epochs_{epochs}_batch_{batch_size}', test_loss, epoch)
            writer.add_scalar(f'Accuracy/train_epochs_{epochs}_batch_{batch_size}', train_acc, epoch)
            writer.add_scalar(f'Accuracy/test_epochs_{epochs}_batch_{batch_size}', test_acc, epoch)

            if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                print(f'Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}')

        writer.close()
        print(f"[INFO] Training complete. Logs saved in {log_dir}")
        return train_acc, test_acc

    def evaluate(self, test_loader):
        self.model.eval()
        all_predictions, all_targets = [], []

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs).squeeze()
                predicted = (outputs > 0.5).float().cpu()
                all_predictions.extend(predicted.numpy())
                all_targets.extend(targets.numpy())

        return classification_report(all_targets, all_predictions)

    def save_model(self, filepath="models"):
        os.makedirs(filepath, exist_ok=True)
        torch.save(self.model.state_dict(), f"{filepath}/{self.model_name}.pth")

    def nr_of_params(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Total Parameters: {total_params}")
    
    def compute_confusion_matrix(self, test_loader, name):
        all_predictions, all_targets = [], []

        for inputs, targets in test_loader:
            inputs = inputs.to(self.device)
            outputs = self.model(inputs).squeeze()
            predicted = (outputs > 0.5).float().cpu()
            all_predictions.extend(predicted.numpy())
            all_targets.extend(targets.numpy())

        #plot_confusion_matrix
        plot_confusion_matrix(all_targets, all_predictions, class_names=["Did Not Survive", "Survived"], name=name)

def main():
    epochs_list = [5, 10, 20, 30]  # Number of epochs to test
    batch_sizes = [8, 16, 32]  # Different batch sizes to test

    for epochs in epochs_list:
        for batch_size in batch_sizes:
            print(f"\nTraining with {epochs} epochs and batch size {batch_size}\n")

            # Load Data with different batch sizes
            train_loader, test_loader, input_dim = get_data_loaders("../dataset/dataset.csv", batch_size=batch_size)

            # Initialize model
            logreg_model = LogisticRegressionTrainer(input_dim)

            # Measure training time
            start_time = time.time()

            # Train the model with specific parameters
            train_acc, test_acc = logreg_model.train(train_loader, test_loader, epochs=epochs, batch_size=batch_size)

            end_time = time.time()

            # Evaluate the model
            print("\nFinal Model Evaluation on Test Data:")
            print(logreg_model.evaluate(test_loader))

            # Nr of parameters
            logreg_model.nr_of_params()

            # Save the model for each combination of parameters
            model_name = f"models/logreg_epochs_{epochs}_batch_{batch_size}.pth"
            logreg_model.save_model(model_name)
            print(f"Model saved to {model_name}\n")

            # Compute confusion matrix
            logreg_model.compute_confusion_matrix(test_loader, f"logistic_regression_epochs_{epochs}_batch_{batch_size}")

            # Print the training time
            print(f"Training time for {epochs} epochs and batch size {batch_size}: {end_time - start_time} seconds")

if __name__ == "__main__":
    main()
