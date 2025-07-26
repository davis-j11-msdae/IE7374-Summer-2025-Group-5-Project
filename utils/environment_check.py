import os
import sys
import torch
import importlib
from typing import Dict, List, Any
from dotenv import load_dotenv
from helpers import load_config, log_operation_status, set_cwd

# Get current working directory for path operations
cwd = set_cwd()


def check_environment_variables() -> Dict[str, bool]:
    """Check required environment variables."""
    load_dotenv()

    required_vars = {
        'HF_TOKEN': os.getenv('HF_TOKEN')
    }

    return {var: value is not None for var, value in required_vars.items()}


def check_gpu_availability() -> Dict[str, Any]:
    """Check GPU availability and specifications."""
    gpu_info = {
        'cuda_available': torch.cuda.is_available(),
        'device_count': 0,
        'devices': []
    }

    if gpu_info['cuda_available']:
        gpu_info['device_count'] = torch.cuda.device_count()
        for i in range(gpu_info['device_count']):
            device_props = torch.cuda.get_device_properties(i)
            gpu_info['devices'].append({
                'name': device_props.name,
                'memory_gb': device_props.total_memory / (1024 ** 3),
                'compute_capability': f"{device_props.major}.{device_props.minor}"
            })

    return gpu_info


def check_required_packages() -> Dict[str, bool]:
    """Check if required packages are installed."""
    required_packages = [
        'torch', 'transformers', 'datasets', 'accelerate',
        'huggingface_hub', 'textstat', 'detoxify', 'pandas',
        'numpy', 'sklearn', 'yaml', 'tqdm', 'peft', 'bitsandbytes'
    ]

    package_status = {}
    for package in required_packages:
        try:
            importlib.import_module(package.replace('-', '_'))
            package_status[package] = True
        except ImportError:
            package_status[package] = False

    return package_status


def check_directory_structure() -> Dict[str, bool]:
    """Check if required directories exist."""
    config = load_config()
    paths = config['paths']

    required_dirs = [
        paths['data_root'],
        paths['data_raw'],
        paths['data_processed'],
        paths['data_tokenized'],
        paths['data_evaluated'],
        paths['models'],
        paths['outputs'],
        paths['user_history'],
        paths['samples'],
        paths['users']
    ]

    dir_status = {}
    for dir_path in required_dirs:
        dir_status[dir_path] = os.path.exists(dir_path)

    return dir_status


def check_data_files() -> Dict[str, bool]:
    """Check if required data files exist."""
    config = load_config()
    paths = config['paths']

    files_to_check = {
        'users_file': os.path.join(paths['users'], "users.txt"),
        'config_file': os.path.join(cwd, "configs", "model_config.yaml")
    }

    file_status = {}
    for file_name, file_path in files_to_check.items():
        file_status[file_name] = os.path.exists(file_path)

    return file_status


def run_full_environment_check() -> Dict[str, Any]:
    """Run complete environment check."""
    log_operation_status("Environment check")

    check_results = {
        'environment_variables': check_environment_variables(),
        'gpu_info': check_gpu_availability(),
        'packages': check_required_packages(),
        'directories': check_directory_structure(),
        'files': check_data_files()
    }

    return check_results


def print_environment_report(check_results: Dict[str, Any]) -> None:
    """Print formatted environment check report."""
    print("\n" + "=" * 60)
    print("ENVIRONMENT CHECK REPORT")
    print("=" * 60)

    # Environment Variables
    print("\nEnvironment Variables:")
    env_vars = check_results['environment_variables']
    for var, status in env_vars.items():
        status_icon = "✓" if status else "✗"
        print(f"  {status_icon} {var}")

    # GPU Information
    print("\nGPU Information:")
    gpu_info = check_results['gpu_info']
    if gpu_info['cuda_available']:
        print(f"  ✓ CUDA Available - {gpu_info['device_count']} device(s)")
        for i, device in enumerate(gpu_info['devices']):
            print(f"    GPU {i}: {device['name']} ({device['memory_gb']:.1f}GB)")
            if device['memory_gb'] >= 8:
                print(f"      ✓ Sufficient VRAM for Mistral 7B")
            else:
                print(f"      ⚠ May need additional optimization for Mistral 7B")
    else:
        print("  ✗ CUDA Not Available")

    # Packages
    print("\nPackage Status:")
    packages = check_results['packages']
    missing_packages = [pkg for pkg, status in packages.items() if not status]
    installed_count = sum(packages.values())
    total_count = len(packages)
    print(f"  Installed: {installed_count}/{total_count}")

    if missing_packages:
        print("  Missing packages:")
        for pkg in missing_packages:
            print(f"    ✗ {pkg}")

    # Directories
    print("\nDirectory Structure:")
    directories = check_results['directories']
    missing_dirs = [dir_path for dir_path, status in directories.items() if not status]
    if missing_dirs:
        print("  Missing directories:")
        for dir_path in missing_dirs:
            print(f"    ✗ {dir_path}")
    else:
        print("  ✓ All directories exist")

    # Files
    print("\nConfiguration Files:")
    files = check_results['files']
    for file_name, status in files.items():
        status_icon = "✓" if status else "✗"
        print(f"  {status_icon} {file_name}")

    # Overall Status
    all_env_vars_ok = all(check_results['environment_variables'].values())
    all_packages_ok = all(check_results['packages'].values())
    all_files_ok = all(check_results['files'].values())
    gpu_available = check_results['gpu_info']['cuda_available']

    print("\nOverall Status:")
    if all_env_vars_ok and all_packages_ok and all_files_ok and gpu_available:
        print("  ✓ System Ready for Mistral 7B")
    else:
        print("  ✗ System Needs Setup")
        if not all_env_vars_ok:
            print("    - Configure environment variables (.env file)")
        if not all_packages_ok:
            print("    - Install missing packages")
        if not all_files_ok:
            print("    - Generate missing configuration files")
        if not gpu_available:
            print("    - CUDA setup required for GPU acceleration")


def main():
    """Main function to run environment check."""
    check_results = run_full_environment_check()
    print_environment_report(check_results)

    log_operation_status("Environment check", "completed")


if __name__ == "__main__":
    main()