# utils.py
import matplotlib.pyplot as plt
import os

def plot_and_save_history(history, experiment_name, save_dir="results"):
    """
    Plots the centralized accuracy from the simulation history and saves it to a file.

    Args:
        history: The History object returned by fl.simulation.start_simulation.
        experiment_name (str): A unique name for the experiment (e.g., 'fedavg_iid').
        save_dir (str): The directory to save the plot.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Extract accuracy from history
    # The accuracy is stored in a list of tuples (round, value)
    rounds, accuracies = zip(*history.metrics_centralized["accuracy"])

    plt.figure(figsize=(10, 6))
    plt.plot(rounds, accuracies, marker='o', linestyle='-')
    plt.title(f"Federated Learning Performance: {experiment_name.replace('_', ' ').title()}")
    plt.xlabel("Communication Round")
    plt.ylabel("Global Model Accuracy")
    plt.grid(True)
    plt.xticks(rounds)
    plt.ylim(0, 1) # Accuracy is between 0 and 1

    # Save the figure
    save_path = os.path.join(save_dir, f"{experiment_name}_accuracy_plot.png")
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    
    # Close the plot to free memory
    plt.close()
