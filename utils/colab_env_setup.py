#!/usr/bin/env python3
"""
Google Colab Environment Setup for Personalized Storytelling System
Upload this file to Colab and run: exec(open('colab_env_setup.py').read())
If Google Drive is already mounted with the Project, you can also run:
import os
if os.path.exists("colab_env_setup.py"):
    exec(open("colab_env_setup.py").read())
else:
    for root, dirs, files in os.walk("/content/drive"):
        if "colab_env_setup.py" in files:
            exec(open(os.path.join(root, "colab_env_setup.py")).read())
"""

import os
import subprocess
import importlib
from pathlib import Path

def check_package_installed(package_name):
    """Check if a package is already installed."""
    try:
        # For NVIDIA packages, check the actual package name with pip
        if package_name.startswith('nvidia_') or 'nvidia' in package_name:
            result = subprocess.run([
                "pip", "show", package_name.replace('_', '-')
            ], capture_output=True, text=True)
            return result.returncode == 0
        
        # For other packages, use importlib
        module_name = package_name.lower().replace('-', '_')
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False

def find_project_folder(start_path, target_folder="IE7374-Summer-2025-Group-5-Project"):
    """Recursively search for the project folder."""
    for root, dirs, files in os.walk(start_path):
        if target_folder in dirs:
            return os.path.join(root, target_folder)
    return None

def search_google_drive(target_folder):
    """Search Google Drive for the project folder."""
    search_paths = [
        "/content/drive/MyDrive",
        "/content/drive/Shared drives", 
        "/content/drive"
    ]
    
    for search_path in search_paths:
        if os.path.exists(search_path):
            project_path = find_project_folder(search_path, target_folder)
            if project_path:
                return project_path
    return None

def mount_google_drive():
    """Mount Google Drive in Colab."""
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        return True
    except ImportError:
        print("Not running in Google Colab environment")
        return False
    except Exception as e:
        print(f"Failed to mount Google Drive: {e}")
        return False

def setup_project_directory():
    """Setup working directory for Colab environment."""
    print("Setting up project directory...")
    
    target_folder = "IE7374-Summer-2025-Group-5-Project"
    gdrive_path = "/content/drive"
    
    # Check if Google Drive is already mounted
    if os.path.exists(gdrive_path):
        project_path = search_google_drive(target_folder)
    else:
        print("Google Drive not mounted")
        mount_response = input("Would you like to mount Google Drive to search for the project? (y/N): ").strip().lower()
        
        project_path = None
        if mount_response == 'y':
            if mount_google_drive():
                project_path = search_google_drive(target_folder)
            else:
                print("Failed to mount Google Drive")
        else:
            print("Skipping Google Drive mount")
    
    # If found in Google Drive, use it
    if project_path:
        os.chdir(project_path)
        print(f"Changed working directory to: {os.getcwd()}")
        return
    
    print("Project folder not found in Google Drive" if os.path.exists(gdrive_path) else "")
    
    # Fallback: Clone from GitHub
    
    # Check if folder already exists locally
    if os.path.exists(target_folder):
        os.chdir(target_folder)
        print(f"Changed working directory to: {os.getcwd()}")
        return
    
    # Clone the repository
    print("Cloning repository from GitHub...")
    clone_command = f"git clone https://github.com/davis-j11-msdae/IE7374-Summer-2025-Group-5-Project.git"
    result = subprocess.run(clone_command, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        os.chdir(target_folder)
        print(f"Changed working directory to: {os.getcwd()}")
    else:
        print(f"Failed to clone repository: {result.stderr}")
        print("Please manually clone or mount Google Drive with the project folder")
        
def create_admin_user():
    """Create users.txt with admin user for immediate access."""
    print("Setting up admin user...")
    
    users_dir = Path("data/users")
    users_dir.mkdir(parents=True, exist_ok=True)

    users_file = users_dir / "users.txt"

    # Check if users file exists
    if users_file.exists():
        try:
            with open(users_file, 'r') as f:
                lines = f.readlines()
            
            # Check if admin user already exists
            for line in lines:
                if line.strip().startswith('admin,'):
                    print("Admin user already exists")
                    return
            
            # Admin doesn't exist, append it
            with open(users_file, 'a') as f:
                f.write("admin,25,admin,1\n")
            print("Added admin user to existing file")
            
        except Exception:
            # If file exists but can't be read, don't modify it
            print("Users file exists but cannot be read")
            return
    else:
        # Create new users file with header and admin user
        with open(users_file, 'w') as f:
            f.write("username,age,password,admin\n")
            f.write("admin,25,admin,1\n")
        print("Created users file with admin user credentials: username=admin, password=admin")

def install_package_fallback(package_spec):
    """Install package using pip as fallback."""
    result = subprocess.run([
        "pip", "install", package_spec, "--quiet"
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        return True
    else:
        return False

def install_wheels():
    """Install all wheel files from directory in dependency order with fallback."""
    print("Check required packages not available on Colab by default...")
    
    # Source directory
    wheels_dir = Path("/content/drive/Othercomputers/DESKTOP_SLER/whls")
    
    # Installation order with wheel files and fallback package specs
    packages = [
        ("python_dotenv-1.1.1-py3-none-any.whl", "python-dotenv==1.1.1", "dotenv"),
        ("pyphen-0.17.2-py3-none-any.whl", "pyphen==0.17.2", "pyphen"),
        ("cmudict-1.0.33-py3-none-any.whl", "cmudict==1.0.33", "cmudict"),
        ("textstat-0.7.7-py3-none-any.whl", "textstat==0.7.7", "textstat"),
        ("hjson-3.1.0-py3-none-any.whl", "hjson==3.1.0", "hjson"),
        ("ninja-1.11.1.4-py3-none-manylinux_2_12_x86_64.manylinux2010_x86_64.whl", "ninja==1.11.1.4", "ninja"),
        ("nvidia_nvjitlink_cu12-12.4.127-py3-none-manylinux2014_x86_64.whl", "nvidia-nvjitlink-cu12==12.4.127", "nvidia-nvjitlink-cu12"),
        ("nvidia_curand_cu12-10.3.5.147-py3-none-manylinux2014_x86_64.whl", "nvidia-curand-cu12==10.3.5.147", "nvidia-curand-cu12"),
        ("nvidia_cufft_cu12-11.2.1.3-py3-none-manylinux2014_x86_64.whl", "nvidia-cufft-cu12==11.2.1.3", "nvidia-cufft-cu12"),
        ("nvidia_cuda_runtime_cu12-12.4.127-py3-none-manylinux2014_x86_64.whl", "nvidia-cuda-runtime-cu12==12.4.127", "nvidia-cuda-runtime-cu12"),
        ("nvidia_cuda_nvrtc_cu12-12.4.127-py3-none-manylinux2014_x86_64.whl", "nvidia-cuda-nvrtc-cu12==12.4.127", "nvidia-cuda-nvrtc-cu12"),
        ("nvidia_cuda_cupti_cu12-12.4.127-py3-none-manylinux2014_x86_64.whl", "nvidia-cuda-cupti-cu12==12.4.127", "nvidia-cuda-cupti-cu12"),
        ("nvidia_cublas_cu12-12.4.5.8-py3-none-manylinux2014_x86_64.whl", "nvidia-cublas-cu12==12.4.5.8", "nvidia-cublas-cu12"),
        ("nvidia_cusparse_cu12-12.3.1.170-py3-none-manylinux2014_x86_64.whl", "nvidia-cusparse-cu12==12.3.1.170", "nvidia-cusparse-cu12"),
        ("nvidia_cudnn_cu12-9.1.0.70-py3-none-manylinux2014_x86_64.whl", "nvidia-cudnn-cu12==9.1.0.70", "nvidia-cudnn-cu12"),
        ("nvidia_cusolver_cu12-11.6.1.9-py3-none-manylinux2014_x86_64.whl", "nvidia-cusolver-cu12==11.6.1.9", "nvidia-cusolver-cu12"),
        ("detoxify-0.5.2-py3-none-any.whl", "detoxify==0.5.2", "detoxify"),
        ("deepspeed-0.17.2-py3-none-any.whl", "deepspeed==0.17.2", "deepspeed"),
        ("bitsandbytes-0.46.1-py3-none-manylinux_2_24_x86_64.whl","bitsandbytes-0.46.1","bitsandbytes")
    ]
    
    for wheel_file, package_spec, check_name in packages:
        # Check if package is already installed
        if check_package_installed(check_name):
            continue
        
        wheel_path = wheels_dir / wheel_file
        install_success = False
        
        # Try wheel installation first if available
        if wheels_dir.exists() and wheel_path.exists():
            result = subprocess.run([
                "pip", "install", 
                "--force-reinstall",
                "--no-deps",
                "--quiet",
                str(wheel_path)
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                install_success = True
            else:
                print(f"Failed to install {check_name} from wheel: {result.stderr}")
        
        # Fallback to pip installation if wheel failed or not available
        if not install_success:
            if not install_package_fallback(package_spec):
                print(f"Failed to install {check_name} from PyPI")

def numpy_check():
  import numpy as np
  import subprocess
  numpy_version = np.__version__
  if numpy_version.startswith('2.'):
      # Uninstall current NumPy
      subprocess.check_call([sys.executable, '-m', 'pip', 'uninstall', 'numpy', '-y'])
      
      # Install NumPy 1.26.4
      subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'numpy==1.26.4'])
      print("NumPy 2.0 detected. Downgraded to 1.26.4 for compatibility. Restart Session.")
      sys.exit(0)

def reload_modules():
  import importlib
  import sys
  target_modules = [
      "helpers", 
      "data_loader", 
      "data_tokenizer", 
      "download_data", 
      "environment_check", 
      "eval", 
      "generate_users", 
      "history", 
      "model_runner", 
      "samples", 
      "train"
  ]

  for target in target_modules:
      for module_name in list(sys.modules.keys()):
          if sys.modules[module_name] is None:
              continue
          
          name_parts = module_name.split('.')
          
          if target in name_parts or module_name.endswith(f".{target}") or module_name == target:
              try:
                  importlib.reload(sys.modules[module_name])
              except:
                  pass
def main():
    """Main setup function."""
    print("Starting Google Colab Environment Setup...")
    print("This will set up the Environment for the Personalized Storytelling System")
    numpy_check()
    reload_modules()
    # Setup project directory
    setup_project_directory()

    # Install dependencies
    install_wheels()

    # Create admin user
    create_admin_user()
    
    print("Setup complete! You can now run the main application.")


if __name__ == "__main__":
    main()