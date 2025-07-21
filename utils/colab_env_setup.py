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
from pathlib import Path

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
        print("❌ Not running in Google Colab environment")
        return False
    except Exception as e:
        print(f"❌ Failed to mount Google Drive: {e}")
        return False

def setup_project_directory():
    """Setup working directory for Colab environment."""
    target_folder = "IE7374-Summer-2025-Group-5-Project"
    gdrive_path = "/content/drive"
    
    # Check if Google Drive is already mounted
    if os.path.exists(gdrive_path):
        project_path = search_google_drive(target_folder)
    else:
        print("❌ Google Drive not mounted")
        mount_response = input("Would you like to mount Google Drive to search for the project? (y/N): ").strip().lower()
        
        project_path = None
        if mount_response == 'y':
            if mount_google_drive():
                project_path = search_google_drive(target_folder)
            else:
                print("❌ Failed to mount Google Drive")
        else:
            print("Skipping Google Drive mount")
    
    # If found in Google Drive, use it
    if project_path:
        os.chdir(project_path)
        print(f"✅ Changed working directory to: {os.getcwd()}")
        return
    
    print("❌ Project folder not found in Google Drive" if os.path.exists(gdrive_path) else "")
    
    # Fallback: Clone from GitHub
    
    # Check if folder already exists locally
    if os.path.exists(target_folder):
        os.chdir(target_folder)
        print(f"✅ Changed working directory to: {os.getcwd()}")
        return
    
    # Clone the repository
    clone_command = f"git clone https://github.com/davis-j11-msdae/IE7374-Summer-2025-Group-5-Project.git"
    result = subprocess.run(clone_command, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        os.chdir(target_folder)
        print(f"✅ Changed working directory to: {os.getcwd()}")
    else:
        print(f"❌ Failed to clone repository: {result.stderr}")
        print("Please manually clone or mount Google Drive with the project folder")
        
def create_admin_user():
    """Create users.txt with admin user for immediate access."""
    users_dir = Path("data/users")
    users_dir.mkdir(parents=True, exist_ok=True)

    users_file = users_dir / "users.txt"

    with open(users_file, 'w') as f:
        f.write("username,age,password,admin\n")
        f.write("admin,25,admin,1\n")

    print("Development Admin credentials: username=admin, password=admin")

def install_package_fallback(package_spec):
    """Install package using pip as fallback."""
    
    result = subprocess.run([
        "pip", "install", package_spec
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        return True
    else:
        print(f"  ❌ "+package_spec+" Fallback installation failed: {result.stderr}")
        return False

def install_wheels():
    """Install all wheel files from directory in dependency order with fallback."""
    
    # Source directory
    wheels_dir = Path("/content/drive/Othercomputers/DESKTOP_SLER/whls")
    
    if not wheels_dir.exists():
        print(f"❌ Wheels directory not found: {wheels_dir}")
        return
    
    # Installation order with wheel files and fallback package specs
    packages = [
        ("dotenv-0.9.9-py2.py3-none-any.whl", "dotenv==0.9.9"),
        ("python_dotenv-1.1.1-py3-none-any.whl", "python-dotenv==1.1.1"),
        ("pyphen-0.17.2-py3-none-any.whl", "pyphen==0.17.2"),
        ("cmudict-1.0.33-py3-none-any.whl", "cmudict==1.0.33"),
        ("textstat-0.7.7-py3-none-any.whl", "textstat==0.7.7"),
        ("hjson-3.1.0-py3-none-any.whl", "hjson==3.1.0"),
        ("ninja-1.11.1.4-py3-none-manylinux_2_12_x86_64.manylinux2010_x86_64.whl", "ninja==1.11.1.4"),
        ("nvidia_nvjitlink_cu12-12.4.127-py3-none-manylinux2014_x86_64.whl", "nvidia-nvjitlink-cu12==12.4.127"),
        ("nvidia_curand_cu12-10.3.5.147-py3-none-manylinux2014_x86_64.whl", "nvidia-curand-cu12==10.3.5.147"),
        ("nvidia_cufft_cu12-11.2.1.3-py3-none-manylinux2014_x86_64.whl", "nvidia-cufft-cu12==11.2.1.3"),
        ("nvidia_cuda_runtime_cu12-12.4.127-py3-none-manylinux2014_x86_64.whl", "nvidia-cuda-runtime-cu12==12.4.127"),
        ("nvidia_cuda_nvrtc_cu12-12.4.127-py3-none-manylinux2014_x86_64.whl", "nvidia-cuda-nvrtc-cu12==12.4.127"),
        ("nvidia_cuda_cupti_cu12-12.4.127-py3-none-manylinux2014_x86_64.whl", "nvidia-cuda-cupti-cu12==12.4.127"),
        ("nvidia_cublas_cu12-12.4.5.8-py3-none-manylinux2014_x86_64.whl", "nvidia-cublas-cu12==12.4.5.8"),
        ("nvidia_cusparse_cu12-12.3.1.170-py3-none-manylinux2014_x86_64.whl", "nvidia-cusparse-cu12==12.3.1.170"),
        ("nvidia_cudnn_cu12-9.1.0.70-py3-none-manylinux2014_x86_64.whl", "nvidia-cudnn-cu12==9.1.0.70"),
        ("nvidia_cusolver_cu12-11.6.1.9-py3-none-manylinux2014_x86_64.whl", "nvidia-cusolver-cu12==11.6.1.9"),
        ("detoxify-0.5.2-py3-none-any.whl", "detoxify==0.5.2"),
        ("deepspeed-0.17.2-py3-none-any.whl", "deepspeed==0.17.2")  # DeepSpeed last
    ]
    
    print(f"Installing {len(packages)} packages from {wheels_dir}")
    
    for wheel_file, package_spec in packages:
        wheel_path = wheels_dir / wheel_file
        
        if wheel_path.exists():
            
            result = subprocess.run([
                "pip", "install", 
                "--force-reinstall",
                "--no-deps",
                str(wheel_path)
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                install_package_fallback(package_spec)
        else:
            install_package_fallback(package_spec)
    
    print("Installation complete!")

def main():
    """Main setup function."""
    print("🚀 Starting Google Colab Environment Setup...")
    print("This will set up the Environment for the Personalized Storytelling System")

    # Setup project directory
    setup_project_directory()

    # Install dependencies
    install_wheels()

    # Create admin user
    create_admin_user()


if __name__ == "__main__":
    main()