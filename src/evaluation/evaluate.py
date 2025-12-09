import torch
from torch import nn

def evaluate_model(model, test_loader, accuracy_threshold=0.5):
    """Evaluate model on test set with specified accuracy threshold.
    
    Args:
        model: PyTorch model to evaluate
        test_loader: DataLoader for test data
        accuracy_threshold: Accuracy threshold in radians for correct predictions
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0
    total_samples = 0
    correct = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            
            total_samples += targets.size(0)
            angle_diff = torch.abs(outputs - targets)
            correct += (angle_diff < accuracy_threshold).all(dim=1).sum().item()
    
    avg_loss = total_loss / len(test_loader)
    accuracy = 100 * correct / total_samples
    
    print(f"  Test Loss: {avg_loss:.6f}")
    print(f"  Accuracy ({accuracy_threshold:.2e} rad): {accuracy:.2f}%")
    return avg_loss
