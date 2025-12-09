import torch
from torch import nn
from ..config import ACCURACY_THRESHOLD

def evaluate_model(model, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0
    total_samples = 0
    correct_predictions_dls = 0
    correct_predictions_relaxed = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            
            total_samples += targets.size(0)
            angle_diff = torch.abs(outputs - targets)
            correct_predictions_dls += (angle_diff < ACCURACY_THRESHOLD).all(dim=1).sum().item()
            correct_predictions_relaxed += (angle_diff < 0.5).all(dim=1).sum().item()
    
    avg_loss = total_loss / len(test_loader)
    accuracy_dls = 100 * correct_predictions_dls / total_samples
    accuracy_relaxed = 100 * correct_predictions_relaxed / total_samples
    print(f"Final Test Loss: {avg_loss:.6f}")
    print(f"Test Accuracy (1e-6 rad / DLS precision): {accuracy_dls:.2f}%")
    print(f"Test Accuracy (0.5 rad / 28.6 deg): {accuracy_relaxed:.2f}%")
    return avg_loss
