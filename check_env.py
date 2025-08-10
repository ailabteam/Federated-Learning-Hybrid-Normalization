import sys
import platform

# --- Helper function for printing ---
def print_status(package_name, status, version=""):
    """Prints a formatted status line."""
    p_name = f"{package_name:<15}"
    if status == "OK":
        stat_colored = "\033[92mOK\033[0m" # Green color
    else:
        stat_colored = "\033[91mFAILED\033[0m" # Red color
    
    version_str = f" (version: {version})" if version else ""
    print(f"[*] Checking {p_name}: {stat_colored}{version_str}")

def print_header(title):
    """Prints a section header."""
    print("\n" + "="*50)
    print(f"| {title:^46} |")
    print("="*50)

# --- Main Check Script ---
if __name__ == "__main__":
    print_header("Federated Learning Environment Check")

    # 1. Check Python Version
    print_header("Python Interpreter")
    py_version = platform.python_version()
    if sys.version_info.major == 3 and sys.version_info.minor >= 8:
        print_status("Python Version", "OK", py_version)
    else:
        print_status("Python Version", "FAILED", py_version)
        print("    -> Warning: Recommended Python 3.8 or newer.")

    # 2. Check Core Libraries
    print_header("Core Libraries")
    libraries_to_check = {
        "torch": None,
        "torchvision": None,
        "flwr": "flower", # package name for pip is "flower"
        "numpy": None,
        "matplotlib": None,
        "wandb": None,
    }

    for lib, package_name in libraries_to_check.items():
        try:
            module = __import__(lib)
            version = module.__version__
            print_status(lib, "OK", version)
        except ImportError:
            pip_install_name = package_name if package_name else lib
            if lib == "flwr":
                pip_install_name = "flwr[simulation]"
            print_status(lib, "FAILED")
            print(f"    -> Please install it by running: pip install {pip_install_name}")
        except AttributeError:
             # Some libraries might not have a __version__ attribute
             print_status(lib, "OK", "version not detected")


    # 3. Check PyTorch CUDA (GPU) availability
    print_header("PyTorch GPU (CUDA) Support")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print_status("CUDA", "OK", f"{gpu_count} device(s) found")
            print(f"    -> GPU 0: {gpu_name}")
        else:
            print_status("CUDA", "FAILED")
            print("    -> PyTorch was installed without CUDA support or no GPU was found.")
            print("    -> Training will run on CPU, which can be very slow.")
    except ImportError:
        print("[!] PyTorch not found, skipping CUDA check.")


    # --- Final Summary ---
    print_header("Check Complete")
    print("If all checks are 'OK', your environment is ready for the project!")
    print("If any check 'FAILED', please follow the instructions to install the missing packages.")
    print("="*50)
