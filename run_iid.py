# run_iid.py
import flwr as fl
from flwr.server.strategy import FedAvg
import torch
from collections import OrderedDict
import warnings
import os

# --- MUTE WARNINGS AND INFO LOGS ---
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["RAY_DEDUP_LOGS"] = "0"
warnings.filterwarnings("ignore", category=DeprecationWarning, module='torchvision.datasets.utils')

from client import FlowerClient, test
from models import get_model
from dataset import load_datasets
from utils import plot_and_save_history

# --- Configuration ---
EXPERIMENT_NAME = "fedavg_bn_iid"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

NUM_CLIENTS = 10
NUM_ROUNDS = 20
LOCAL_EPOCHS = 5
BATCH_SIZE = 32

print(f"\nStarting experiment: {EXPERIMENT_NAME}")
print(f"Training on {DEVICE} using PyTorch {torch.__version__} and Flower {fl.__version__}")

# --- Centralized Evaluation Function ---
def get_evaluate_fn(test_loader):
    def evaluate(server_round, parameters, config):
        model = get_model().to(DEVICE)
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        
        loss, accuracy = test(model, test_loader)
        print(f"Round {server_round:02d} | Server-side evaluation: accuracy = {accuracy:.4f}, loss = {loss:.4f}")
        return loss, {"accuracy": accuracy}
    return evaluate

# --- Client Creation Function ---
def client_fn(cid: str) -> fl.client.Client:
    net = get_model().to(DEVICE)
    trainloader = trainloaders[int(cid)]
    # Use the client's specific testloader if available, otherwise None
    client_testloader = testloaders[int(cid)] if testloaders else None
    
    numpy_client = FlowerClient(cid, net, trainloader, client_testloader)
    return numpy_client.to_client()


if __name__ == "__main__":
    # 1. Load IID data (Corrected to receive 3 values)
    trainloaders, testloader, testloaders = load_datasets(
        num_clients=NUM_CLIENTS,
        partition_type="iid",
        batch_size=BATCH_SIZE
    )

    # 2. Define FedAvg strategy
    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        evaluate_fn=get_evaluate_fn(testloader),
        on_fit_config_fn=lambda server_round: {"local_epochs": LOCAL_EPOCHS},
    )

    # 3. Start simulation
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        history = fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=NUM_CLIENTS,
            config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
            strategy=strategy,
            client_resources={"num_cpus": 2, "num_gpus": 0.5} if DEVICE.type == "cuda" else None
        )

    # 4. Save results
    print(f"\n--- Experiment {EXPERIMENT_NAME} Finished ---")
    if history.metrics_centralized and "accuracy" in history.metrics_centralized:
        final_acc = history.metrics_centralized["accuracy"][-1][1]
        print(f"Final accuracy: {final_acc:.4f}")
        plot_and_save_history(history, EXPERIMENT_NAME, save_dir="results/step1_replication")
    else:
        print("Could not retrieve accuracy from history. Evaluation might have failed.")
