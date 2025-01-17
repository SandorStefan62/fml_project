import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ================================
# Load dataset
# ================================

X_train_df = pd.read_csv('../dataset/X_train.csv')
X_test_df = pd.read_csv('../dataset/X_test.csv')
y_train_df = pd.read_csv('../dataset/y_train.csv')
y_test_df = pd.read_csv('../dataset/y_test.csv')

print("NaNs in X_train:", X_train_df.isnull().sum())
print("NaNs in X_test:", X_test_df.isnull().sum())

# Convert to tensors
X_train = torch.tensor(X_train_df.values, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test_df.values, dtype=torch.float32).to(device)

y_train = torch.tensor(y_train_df.values, dtype=torch.float32).squeeze().to(device)
y_test = torch.tensor(y_test_df.values, dtype=torch.float32).squeeze().to(device)

num_features = X_train.shape[1]

# ================================
# Define the Logistic Regression Model
# ================================
class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegression, self).__init__()
        # A single linear layer from input_dim to 1 output
        self.linear = nn.Linear(input_dim, 1)
        
    def forward(self, x):
        # Forward pass: linear -> sigmoid
        logits = self.linear(x)           # shape: [batch_size, 1]
        probs = torch.sigmoid(logits)     # shape: [batch_size, 1]
        return probs

# Initialize the model
model = LogisticRegression(input_dim=num_features)

# ================================
# 3. Define Loss Function and Optimizer
# ================================
criterion = nn.BCELoss()               # For binary classification
optimizer = optim.SGD(model.parameters(), lr=0.01)

# ================================
# 4. Training Loop
# ================================
num_epochs = 150
for epoch in range(num_epochs):
    model.train()  # set model to training mode
    
    # Forward pass
    outputs = model(X_train).squeeze()  # shape: [batch_size]
    loss = criterion(outputs, y_train)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Update weights
    optimizer.step()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# ================================
# 5. Evaluate the Model
# ================================
model.eval()

def evaluate_model(X_tensor, y_tensor, dataset_name="Dataset"):
    with torch.no_grad():
        # Get predicted probabilities
        predicted_probs = model(X_tensor).squeeze()  # shape: [batch_size]
        # Convert probabilities to binary predictions (threshold = 0.5)
        predicted_labels = (predicted_probs >= 0.5).float()
        # Calculate accuracy
        accuracy = (predicted_labels == y_tensor).float().mean()
        print(f"{dataset_name} Accuracy: {accuracy.item() * 100:.2f}%")

evaluate_model(X_train, y_train, "Training")
evaluate_model(X_test, y_test, "Test")

# ================================
# 6. Save the Trained Model
# ================================
# We'll save the model's parameters (state_dict) in the same directory as this script.
model_save_path = os.path.join(os.path.dirname(__file__), "logistic_model.pt")
torch.save(model.state_dict(), model_save_path)

print(f"\nModel parameters saved to: {model_save_path}")