import subprocess
import os, sys
import pkg_resources
from tqdm import tqdm
import urllib.request
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

req_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "requirements.txt")

models_dir = os.path.abspath("models/roop")

model_url = "https://huggingface.co/henryruhs/roop/resolve/main/inswapper_128.onnx"
model_name = os.path.basename(model_url)
model_path = os.path.join(models_dir, model_name)

def run_pip(*args):
    subprocess.run([sys.executable, "-m", "pip", "install", *args])

def is_installed (
        package: str, version: str | None = None
):
    has_package = None
    try:
        has_package = pkg_resources.get_distribution(package)
        if has_package is not None:
            installed_version = has_package.version
            if installed_version != version:
                return False
            else:
                return True
        else:
            return False
    except:
        return False
    
def download(url, path):
    request = urllib.request.urlopen(url)
    total = int(request.headers.get('Content-Length', 0))
    with tqdm(total=total, desc='Downloading', unit='B', unit_scale=True, unit_divisor=1024) as progress:
        urllib.request.urlretrieve(url, path, reporthook=lambda count, block_size, total_size: progress.update(block_size))

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(model_path):
    download(model_url, model_path)

print("Installing...")
with open(req_file) as file:
    for package in file:
        package_version = None
        try:
            package = package.strip()
            if "==" in package:
                package_version = package.split('==')[1]
            if not is_installed(package,package_version):
                run_pip(package)
        except Exception as e:
            print(e)
            print(f"Warning: Failed to install {package}, ReActor will not work.")
            raise e
print("Ok")
