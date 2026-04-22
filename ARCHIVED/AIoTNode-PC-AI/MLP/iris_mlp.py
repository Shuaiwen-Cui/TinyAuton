"""
Iris Classification - MLP (MCU Deployment Optimized)
Simplified model structure for microcontroller deployment
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import onnx
import pickle


class SimpleMLP(nn.Module):
    """
    Simplified MLP model for MCU deployment
    Structure: 4 -> 16 -> 8 -> 3
    Uses ReLU activation, no Dropout
    """
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(4, 16)   # Input layer to hidden layer 1
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(16, 8)   # Hidden layer 1 to hidden layer 2
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(8, 3)    # Hidden layer 2 to output layer
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x


def load_and_preprocess_data(csv_path='iris_data.csv'):
    """
    Load and preprocess data from CSV file
    
    Args:
        csv_path: CSV file path with columns: sepal_length, sepal_width, petal_length, petal_width, species
    
    Returns:
        X_train, X_test, y_train, y_test, scaler
    """
    # Load data from CSV
    df = pd.read_csv(csv_path)
    
    # Separate features and labels
    feature_columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    X = df[feature_columns].values
    y = df['species'].values
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, scaler


def train_model(model, X_train, y_train, epochs=300, lr=0.01):
    """Train the model"""
    # Convert to Tensor
    X_tensor = torch.FloatTensor(X_train)
    y_tensor = torch.LongTensor(y_train)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        # Forward pass
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 50 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate the model"""
    model.eval()
    X_tensor = torch.FloatTensor(X_test)
    y_tensor = torch.LongTensor(y_test)
    
    with torch.no_grad():
        outputs = model(X_tensor)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y_tensor).float().mean().item()
    
    print(f'Test Accuracy: {accuracy*100:.2f}%')
    return accuracy


def export_to_onnx(model, onnx_path='iris_mlp.onnx'):
    """Export ONNX model - MCU optimized"""
    model.eval()
    dummy_input = torch.FloatTensor([[0.0, 0.0, 0.0, 0.0]])
    
    # Use opset 18 (PyTorch default) for better compatibility
    # Most MCU toolchains support opset 18 now
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
    )
    
    # Validate model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    
    # Check actual opset version
    actual_opset = onnx_model.opset_import[0].version
    print(f'ONNX model exported: {onnx_path} (opset {actual_opset})')
    
    return onnx_path


def main():
    """Main function"""
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load data
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data()
    print(f"Training set: {len(X_train)}, Test set: {len(X_test)}")
    
    # Create model
    model = SimpleMLP()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params}")
    
    # Train
    model = train_model(model, X_train, y_train, epochs=300, lr=0.01)
    
    # Evaluate
    evaluate_model(model, X_test, y_test)
    
    # Export ONNX
    export_to_onnx(model)
    
    # Save scaler
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print('Scaler saved: scaler.pkl')


if __name__ == '__main__':
    main()

