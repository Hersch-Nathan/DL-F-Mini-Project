import torch
from torch import nn

def train_model(model, train_loader, test_loader, epochs=100, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    model = model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    best_test_loss = float('inf')
    patience = 20
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_samples = 0
        correct_predictions = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            
            total_samples += targets.size(0)
            angle_diff = torch.abs(outputs - targets)
            correct_predictions += (angle_diff < 0.5).all(dim=1).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct_predictions / total_samples
        
        if (epoch + 1) % 10 == 0:
            model.eval()
            test_loss = 0
            test_samples = 0
            test_correct = 0
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    test_loss += criterion(outputs, targets).item()
                    test_samples += targets.size(0)
                    angle_diff = torch.abs(outputs - targets)
                    test_correct += (angle_diff < 0.5).all(dim=1).sum().item()
            test_avg_loss = test_loss / len(test_loader)
            test_accuracy = 100 * test_correct / test_samples
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.6f}, Train Acc: {accuracy:.2f}%, Test Loss: {test_avg_loss:.6f}, Test Acc: {test_accuracy:.2f}%")
            
            scheduler.step(test_avg_loss)
            
            if test_avg_loss < best_test_loss:
                best_test_loss = test_avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
    
    return model

def train_tejomurt_model(model, train_loader, test_loader, output_mean, output_std, epochs=100, lr=0.01):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    model = model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    
    output_mean_t = torch.tensor(output_mean, dtype=torch.float32).to(device)
    output_std_t = torch.tensor(output_std, dtype=torch.float32).to(device)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_samples = 0
        correct_predictions = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            total_samples += targets.size(0)
            outputs_denorm = outputs * output_std_t + output_mean_t
            targets_denorm = targets * output_std_t + output_mean_t
            angle_diff = torch.abs(outputs_denorm - targets_denorm)
            correct_predictions += (angle_diff < 0.5).all(dim=1).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct_predictions / total_samples
        
        if (epoch + 1) % 10 == 0:
            model.eval()
            test_loss = 0
            test_samples = 0
            test_correct = 0
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    test_loss += criterion(outputs, targets).item()
                    test_samples += targets.size(0)
                    outputs_denorm = outputs * output_std_t + output_mean_t
                    targets_denorm = targets * output_std_t + output_mean_t
                    angle_diff = torch.abs(outputs_denorm - targets_denorm)
                    test_correct += (angle_diff < 0.5).all(dim=1).sum().item()
            test_avg_loss = test_loss / len(test_loader)
            test_accuracy = 100 * test_correct / test_samples
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.6f}, Train Acc: {accuracy:.2f}%, Test Loss: {test_avg_loss:.6f}, Test Acc: {test_accuracy:.2f}%")
        
        scheduler.step()
    
    return model
