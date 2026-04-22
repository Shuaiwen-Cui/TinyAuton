"""
Gesture Recognition - 1D CNN (MCU Deployment Optimized)
Using accelerometer data to classify gestures
Simplified model structure for microcontroller deployment
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import onnx
import pickle


class Simple1DCNN(nn.Module):
    """
    Simplified 1D CNN model for MCU deployment
    Input: (batch, 3, seq_len) - 3-axis accelerometer data
    Structure: Conv1D -> ReLU -> Conv1D -> ReLU -> GlobalAvgPool -> FC
    """
    def __init__(self, num_classes=5, seq_len=64):
        super(Simple1DCNN, self).__init__()
        # First conv layer: 3 channels -> 8 channels
        self.conv1 = nn.Conv1d(in_channels=3, out_channels=8, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        # Second conv layer: 8 channels -> 16 channels
        self.conv2 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        # Fully connected layer
        self.fc = nn.Linear(16, num_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.global_pool(x)
        x = x.squeeze(-1)  # Remove sequence dimension
        x = self.fc(x)
        return x


def generate_gesture_data(num_samples=500, seq_len=64, num_classes=5):
    """
    Generate synthetic accelerometer data for gesture recognition
    Simulates 5 gestures: swipe_left, swipe_right, tap, circle, wave
    
    Args:
        num_samples: Number of samples per class
        seq_len: Sequence length (time steps)
        num_classes: Number of gesture classes
    
    Returns:
        X: (num_samples*num_classes, 3, seq_len) - accelerometer data
        y: (num_samples*num_classes,) - gesture labels
    """
    np.random.seed(42)
    X_list = []
    y_list = []
    
    for class_id in range(num_classes):
        for _ in range(num_samples):
            # Generate synthetic accelerometer pattern for each gesture
            if class_id == 0:  # swipe_left
                pattern = np.sin(np.linspace(0, 2*np.pi, seq_len)) * 0.5
                x_axis = pattern + np.random.normal(0, 0.1, seq_len)
                y_axis = np.random.normal(0, 0.1, seq_len)
                z_axis = np.random.normal(1.0, 0.1, seq_len)
            elif class_id == 1:  # swipe_right
                pattern = -np.sin(np.linspace(0, 2*np.pi, seq_len)) * 0.5
                x_axis = pattern + np.random.normal(0, 0.1, seq_len)
                y_axis = np.random.normal(0, 0.1, seq_len)
                z_axis = np.random.normal(1.0, 0.1, seq_len)
            elif class_id == 2:  # tap
                x_axis = np.random.normal(0, 0.1, seq_len)
                y_axis = np.random.normal(0, 0.1, seq_len)
                z_axis = np.concatenate([np.random.normal(1.0, 0.1, seq_len//2),
                                         np.random.normal(0.5, 0.2, seq_len//4),
                                         np.random.normal(1.0, 0.1, seq_len//4)])
            elif class_id == 3:  # circle
                t = np.linspace(0, 2*np.pi, seq_len)
                x_axis = np.cos(t) * 0.5 + np.random.normal(0, 0.1, seq_len)
                y_axis = np.sin(t) * 0.5 + np.random.normal(0, 0.1, seq_len)
                z_axis = np.random.normal(1.0, 0.1, seq_len)
            else:  # wave
                pattern = np.sin(np.linspace(0, 4*np.pi, seq_len)) * 0.3
                x_axis = pattern + np.random.normal(0, 0.1, seq_len)
                y_axis = np.random.normal(0, 0.1, seq_len)
                z_axis = np.random.normal(1.0, 0.1, seq_len)
            
            # Stack into (3, seq_len) format
            sample = np.stack([x_axis, y_axis, z_axis], axis=0)
            X_list.append(sample)
            y_list.append(class_id)
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    return X, y


def save_data_to_csv(X, y, csv_path='gesture_data.csv'):
    """Save gesture data to CSV file"""
    # Flatten the 3D array to 2D for CSV
    num_samples, num_channels, seq_len = X.shape
    data_list = []
    
    for i in range(num_samples):
        row = {'gesture': y[i]}
        for ch in range(num_channels):
            for t in range(seq_len):
                row[f'ch{ch}_t{t}'] = X[i, ch, t]
        data_list.append(row)
    
    df = pd.DataFrame(data_list)
    df.to_csv(csv_path, index=False)
    print(f'Data saved to {csv_path}')
    return csv_path


def load_and_preprocess_data(csv_path='gesture_data.csv', seq_len=64):
    """
    Load and preprocess data from CSV file
    
    Args:
        csv_path: CSV file path
        seq_len: Sequence length
    
    Returns:
        X_train, X_test, y_train, y_test, scaler
    """
    df = pd.read_csv(csv_path)
    
    # Extract labels
    y = df['gesture'].values
    
    # Extract features and reshape to (samples, channels, seq_len)
    feature_cols = [col for col in df.columns if col.startswith('ch')]
    X_flat = df[feature_cols].values
    num_samples = len(X_flat)
    X = X_flat.reshape(num_samples, 3, seq_len)
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Standardize features (per channel)
    scaler = StandardScaler()
    # Reshape for scaling: (samples*channels, seq_len)
    n_train, n_ch, n_seq = X_train.shape
    X_train_reshaped = X_train.transpose(0, 1, 2).reshape(-1, n_seq)
    X_train_scaled = scaler.fit_transform(X_train_reshaped)
    X_train_scaled = X_train_scaled.reshape(n_train, n_ch, n_seq).transpose(0, 1, 2)
    
    n_test, _, _ = X_test.shape
    X_test_reshaped = X_test.transpose(0, 1, 2).reshape(-1, n_seq)
    X_test_scaled = scaler.transform(X_test_reshaped)
    X_test_scaled = X_test_scaled.reshape(n_test, n_ch, n_seq).transpose(0, 1, 2)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def train_model(model, X_train, y_train, epochs=200, lr=0.001):
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
        
        if (epoch + 1) % 40 == 0:
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


def export_to_onnx(model, onnx_path='gesture_cnn_1d.onnx', seq_len=64):
    """Export ONNX model - MCU optimized"""
    model.eval()
    # Fixed input shape: (batch=1, channels=3, seq_len=64)
    dummy_input = torch.FloatTensor(np.zeros((1, 3, seq_len)))
    
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
    
    seq_len = 64
    num_classes = 5
    
    # Generate or load data
    print("Generating gesture data...")
    X, y = generate_gesture_data(num_samples=400, seq_len=seq_len, num_classes=num_classes)
    print(f"Generated {len(X)} samples")
    
    # Save to CSV
    save_data_to_csv(X, y, 'gesture_data.csv')
    
    # Load and preprocess
    print("\nLoading and preprocessing data...")
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data('gesture_data.csv', seq_len)
    print(f"Training set: {len(X_train)}, Test set: {len(X_test)}")
    
    # Create model
    model = Simple1DCNN(num_classes=num_classes, seq_len=seq_len)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params}")
    print(model)
    
    # Train
    print("\nTraining...")
    model = train_model(model, X_train, y_train, epochs=200, lr=0.001)
    
    # Evaluate
    print("\nEvaluating...")
    evaluate_model(model, X_test, y_test)
    
    # Export ONNX
    print("\nExporting ONNX...")
    export_to_onnx(model, 'gesture_cnn_1d.onnx', seq_len)
    
    # Save scaler
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print('Scaler saved: scaler.pkl')


if __name__ == '__main__':
    main()

