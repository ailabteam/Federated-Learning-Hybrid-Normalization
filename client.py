# client.py (Full Code)
import torch
import flwr as fl
from collections import OrderedDict
from models import get_model # This will be updated to get_model(norm_type)

# Check if CUDA is available and set the device
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(net, trainloader, epochs):
    """Train the model on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    net.train()
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()

def test(net, testloader):
    """Validate the model on the test set."""
    if not testloader:
        return 0.0, 0.0 # Handle case where client testloader is None

    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    if total == 0:
        return 0.0, 0.0
        
    accuracy = correct / total
    return loss / len(testloader.dataset), accuracy

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, net, trainloader, testloader):
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.testloader = testloader

    def get_parameters(self, config):
        """Get model parameters, optionally filtering for FedBN."""
        # If FedBN is enabled in the config, filter out BN parameters
        if config.get("fedbn", False):
            # Find all keys that are not part of a BatchNorm layer
            non_bn_keys = [key for key in self.net.state_dict().keys() if "bn" not in key]
            # Return the corresponding values
            return [self.net.state_dict()[key].cpu().numpy() for key in non_bn_keys]
        else:
            # Default behavior: return all parameters
            return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        """Set model parameters, handling both FedAvg and FedBN."""
        all_keys = self.net.state_dict().keys()
        
        # Check if the number of received parameters is less than the total number
        # This is a strong indicator of FedBN.
        if len(parameters) < len(all_keys):
            # Get the keys for non-BN layers
            non_bn_keys = [key for key in all_keys if "bn" not in key]
            params_dict = zip(non_bn_keys, parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            # Load the non-BN parameters, keeping the client's local BN params
            self.net.load_state_dict(state_dict, strict=False)
        else:
            # FedAvg: load all parameters
            params_dict = zip(all_keys, parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        epochs = config.get("local_epochs", 1)
        train(self.net, self.trainloader, epochs=epochs)
        # Pass the config to get_parameters to enable FedBN logic
        return self.get_parameters(config=config), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(self.net, self.testloader)
        return float(loss), len(self.testloader.dataset), {"accuracy": float(accuracy)}
