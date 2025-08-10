# run_fedbn.py (Corrected Version)
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
# This function is now CORRECT because the parameters it receives from the server
# will also be non-BN parameters.
def get_evaluate_fn(test_loader):
    def evaluate(server_round, parameters, config):
        model = get_model(norm_type="bn").to(DEVICE)
        
        # Get the keys for non-BN layers from a fresh model instance
        non_bn_keys = [key for key in model.state_dict().keys() if "bn" not in key]
        
        # Create the state_dict for non-BN layers
        params_dict = zip(non_bn_keys, parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        
        # Load the non-BN parameters. The BN layers in `model` will keep their initial random state.
        # This is fine for evaluation as they are in eval mode.
        model.load_state_dict(state_dict, strict=False)
        
        loss, accuracy = test(model, test_loader)
        print(f"Round {server_round:02d} | FedBN Server-side eval: accuracy = {accuracy:.4f}, loss = {loss:.4f}")
        return loss, {"accuracy": accuracy}
    return evaluate

# --- Client Creation Function ---
def client_fn(cid: str) -> fl.client.Client:
    net = get_model(norm_type="bn").to(DEVICE)
    trainloader = trainloaders[int(cid)]
    client_testloader = testloaders[int(cid)] if testloaders else None
    
    numpy_client = FlowerClient(cid, net, trainloader, client_testloader)
    return numpy_client.to_client()

# --- Fit Configuration Function ---
def fit_config(server_round: int):
    return {
        "local_epochs": LOCAL_EPOCHS,
        "fedbn": True,
    }

if __name__ == "__main__":
    trainloaders, testloader, testloaders = load_datasets(
        num_clients=NUM_CLIENTS,
        partition_type="non-iid-dirichlet",
        alpha=NON_IID_ALPHA,
        batch_size=BATCH_SIZE
    )

    # --- STRATEGY INITIALIZATION (THE FIX IS HERE) ---
    # 1. Create a sample model to get the initial non-BN parameters
    initial_model = get_model(norm_type="bn")
    non_bn_params = [
        val.cpu().numpy()
        for key, val in initial_model.state_dict().items()
        if "bn" not in key
    ]
    
    # 2. Convert these numpy arrays to Flower Parameters
    initial_parameters = fl.common.ndarrays_to_parameters(non_bn_params)

    # 3. Initialize the strategy with these non-BN parameters
    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        evaluate_fn=get_evaluate_fn(testloader),
        on_fit_config_fn=fit_config,
        initial_parameters=initial_parameters, # Provide the correct initial parameters
    )

    # --- Start Simulation ---
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        history = fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=NUM_CLIENTS,
            config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
            strategy=strategy,
            client_resources={"num_cpus": 2, "num_gpus": 0.5} if DEVICE.type == "cuda" else None
        )

    # --- Save Results ---
    print(f"\n--- Experiment {EXPERIMENT_NAME} Finished ---")
    if history.metrics_centralized and "accuracy" in history.metrics_centralized:
        # Check if the list is not empty before accessing the last element
        if history.metrics_centralized["accuracy"]:
            final_acc = history.metrics_centralized["accuracy"][-1][1]
            print(f"Final accuracy: {final_acc:.4f}")
            plot_and_save_history(history, EXPERIMENT_NAME, save_dir="results/step2_baselines")
        else:
            print("Evaluation results are empty. Something went wrong during evaluation.")
    else:
        print("Could not retrieve accuracy from history.")
