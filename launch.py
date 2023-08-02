import importlib.util
import subprocess
import sys


def is_installed(package):
    try:
        spec = importlib.util.find_spec(package)
    except ModuleNotFoundError:
        return False

    return spec is not None


def run_pip(command, desc):
    python = sys.executable
    subprocess.check_call([python, "-m", "pip", *command.split(" ")])
    