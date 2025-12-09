import torch
from torch import nn

def evaluate_model(model, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0
    total_samples = 0
    correct_predictions_strict = 0
    correct_predictions_relaxed = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            
            total_samples += targets.size(0)
            angle_diff = torch.abs(outputs - targets)
            correct_predictions_strict += (angle_diff < 0.1).all(dim=1).sum().item()
            correct_predictions_relaxed += (angle_diff < 0.5).all(dim=1).sum().item()
    
    avg_loss = total_loss / len(test_loader)
    accuracy_strict = 100 * correct_predictions_strict / total_samples
    accuracy_relaxed = 100 * correct_predictions_relaxed / total_samples
    print(f"Final Test Loss: {avg_loss:.6f}")
    print(f"Test Accuracy (0.1 rad / 5.7 deg): {accuracy_strict:.2f}%")
    print(f"Test Accuracy (0.5 rad / 28.6 deg): {accuracy_relaxed:.2f}%")
    return avg_loss
