import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import classification_report
from datetime import datetime
from dataset import get_data_loaders

class TitanicNet(nn.Module):
    def __init__(self, input_dim):
        super(TitanicNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)

class TitanicNeuralNetwork:
    def __init__(self, input_dim):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = TitanicNet(input_dim).to(self.device)
        self.model_name = 'neural_network'
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters())
        
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets.unsqueeze(1))
            
            # Backward pass and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            predicted = (outputs > 0.5).float()
            total += targets.size(0)
            correct += (predicted.squeeze() == targets).sum().item()
        
        return total_loss / len(train_loader), correct / total
    
    def evaluate_epoch(self, test_loader):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets.unsqueeze(1))
                
                total_loss += loss.item()
                predicted = (outputs > 0.5).float()
                total += targets.size(0)
                correct += (predicted.squeeze() == targets).sum().item()
        
        return total_loss / len(test_loader), correct / total
        
    def train(self, train_loader, test_loader, epochs = 50, batch_size = 32):
        # Create TensorBoard writer
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = f'logs/{self.model_name}/epochs_{epochs}_batch_{batch_size}/{current_time}'
        writer = SummaryWriter(log_dir)
        
        print(f"\n[INFO] Training model for {epochs} epochs with batch size {batch_size}")

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
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                predicted = (outputs > 0.5).float().cpu()
                all_predictions.extend(predicted.squeeze().numpy())
                all_targets.extend(targets.numpy())
        
        return classification_report(all_targets, all_predictions)
    
    def save_model(self, filepath='models'):
        os.makedirs(filepath, exist_ok=True)
        torch.save(self.model.state_dict(), f'{filepath}/{self.model_name}.pth')

def main():
    epochs = [10, 20, 30]
    batch_sizes = [8, 16, 32]

    for epochs in epochs:
        for batch_size in batch_sizes:
            print(f"\nTraining with {epochs} epochs and batch size {batch_size}\n")

            # Load Data with different batch sizes
            train_loader, test_loader, input_dim = get_data_loaders('../dataset/dataset.csv', batch_size=batch_size)

            # Initialize model
            nn_model = TitanicNeuralNetwork(input_dim)

            # Train the model with specific parameters
            train_acc, test_acc = nn_model.train(train_loader, test_loader, epochs=epochs, batch_size=batch_size)

            # Evaluate the model
            print("\nFinal Model Evaluation on Test Data:")
            print(nn_model.evaluate(test_loader))

            # Save the model for each combination of parameters
            model_name = f'models/nn_epochs_{epochs}_batch_{batch_size}.pth'
            nn_model.save_model(model_name)
            print(f"Model saved to {model_name}\n")

if __name__ == "__main__":
    main()