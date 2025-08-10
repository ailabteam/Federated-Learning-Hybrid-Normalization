# run_fedbn.py
import flwr as fl
from flwr.server.strategy import FedAvg
import torch
from collections import OrderedDict
import warnings
import os

# --- MUTE WARNINGS ---
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["RAY_DEDUP_LOGS"] = "0"
warnings.filterwarnings("ignore", category=DeprecationWarning, module='torchvision.datasets.utils')

from client import FlowerClient, test
from models import get_model
from dataset import load_datasets
from utils import plot_and_save_history

# --- Configuration ---
EXPERIMENT_NAME = "fedbn_non_iid"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

NUM_CLIENTS = 10
NUM_ROUNDS = 20
LOCAL_EPOCHS = 5
BATCH_SIZE = 32
NON_IID_ALPHA = 0.5

print(f"\nStarting experiment: {EXPERIMENT_NAME}")
print(f"Training on {DEVICE} using PyTorch {torch.__version__} and Flower {fl.__version__}")

# --- Centralized Evaluation Function ---
def get_evaluate_fn(test_loader):
    def evaluate(server_round, parameters, config):
        # FedBN server does not have BN layers, so we load into a model
        # with strict=False
        model = get_model(norm_type="bn").to(DEVICE) # Use BN model as the base
        
        non_bn_keys = [k for k in model.state_dict().keys() if "bn" not in k]
        params_dict = zip(non_bn_keys, parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        
        model.load_state_dict(state_dict, strict=False)
        
        loss, accuracy = test(model, test_loader)
        print(f"Round {server_round:02d} | FedBN Server-side eval: accuracy = {accuracy:.4f}, loss = {loss:.4f}")
        return loss, {"accuracy": accuracy}
    return evaluate

# --- Client Creation Function ---
def client_fn(cid: str) -> fl.client.Client:
    net = get_model(norm_type="bn").to(DEVICE) # Clients use BN model
    trainloader = trainloaders[int(cid)]
    client_testloader = testloaders[int(cid)] if testloaders else None
    
    numpy_client = FlowerClient(cid, net, trainloader, client_testloader)
    return numpy_client.to_client()

# --- Fit Configuration Function ---
def fit_config(server_round: int):
    """Return training configuration dict for each round."""
    config = {
        "local_epochs": LOCAL_EPOCHS,
        "fedbn": True, # <-- This is the key to activate FedBN on the client
    }
    return config

if __name__ == "__main__":
    trainloaders, testloader, testloaders = load_datasets(
        num_clients=NUM_CLIENTS,
        partition_type="non-iid-dirichlet",
        alpha=NON_IID_ALPHA,
        batch_size=BATCH_SIZE
    )

    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        evaluate_fn=get_evaluate_fn(testloader),
        on_fit_config_fn=fit_config,
        # We need to tell the server to NOT initialize parameters
        # because the server model (without BN) is different from client model
        # However, FedAvg handles this by just waiting for the first client.
        # So we don't need initial_parameters.
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        history = fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=NUM_CLIENTS,
            config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
            strategy=strategy,
            client_resources={"num_cpus": 2, "num_gpus": 0.5} if DEVICE.type == "cuda" else None
        )

    print(f"\n--- Experiment {EXPERIMENT_NAME} Finished ---")
    if history.metrics_centralized and "accuracy" in history.metrics_centralized:
        final_acc = history.metrics_centralized["accuracy"][-1][1]
        print(f"Final accuracy: {final_acc:.4f}")
        plot_and_save_history(history, EXPERIMENT_NAME, save_dir="results/step2_baselines")
