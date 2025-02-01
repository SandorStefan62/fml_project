import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from dataset import get_data_loaders
import torch

class TitanicRandomForest:
    def __init__(self, n_estimators=200, max_depth=10, min_samples_split=10, min_samples_leaf=5, class_weight="balanced", max_features="sqrt", random_state=42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            class_weight=class_weight,
            random_state=random_state
        )
        self.model_name = 'random_forest'
        
    def train(self, train_loader, test_loader):
        # Create TensorBoard writer
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = f'logs/{self.model_name}/{current_time}'
        writer = SummaryWriter(log_dir)
        
        # Get all data from loaders for sklearn model
        X_train = torch.cat([batch[0] for batch in train_loader]).numpy()
        y_train = torch.cat([batch[1] for batch in train_loader]).numpy()
        X_test = torch.cat([batch[0] for batch in test_loader]).numpy()
        y_test = torch.cat([batch[1] for batch in test_loader]).numpy()
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Calculate training and test metrics
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)
        
        # Log metrics to TensorBoard
        writer.add_scalar('Accuracy/train', train_acc, 0)
        writer.add_scalar('Accuracy/test', test_acc, 0)
        writer.close()
            
        return train_acc, test_acc
    
    def feature_importance(self, feature_names):
        importances = self.model.feature_importances_
        sorted_idx = np.argsort(importances)[::-1]

        print("\nFeature Importance Ranking:")
        for i in sorted_idx:
            print(f"{feature_names[i]}: {importances[i]:.4f}")
    
    def evaluate(self, test_loader):
        X_test = torch.cat([batch[0] for batch in test_loader]).numpy()
        y_test = torch.cat([batch[1] for batch in test_loader]).numpy()
        predictions = self.model.predict(X_test)
        return classification_report(y_test, predictions)
    
    def save_model(self, filepath='models'):
        os.makedirs(filepath, exist_ok=True)
        joblib.dump(self.model, f'{filepath}/{self.model_name}.joblib')

def main():
        # Load the data
        train_loader, test_loader, input_dim = get_data_loaders("../dataset/dataset.csv")

        # Initialize model
        rf_model = TitanicRandomForest()

        # Train the model with specific parameters
        train_acc, test_acc = rf_model.train(train_loader, test_loader)

        # Evaluate the model
        print("\nFinal Model Evaluation on Test Data:")
        print(rf_model.evaluate(test_loader))

        # Save the model for each batch size
        model_name = f"models/rf_model.joblib"
        rf_model.save_model(model_name)
        print(f"Model saved to {model_name}\n")

if __name__ == "__main__":
    main()