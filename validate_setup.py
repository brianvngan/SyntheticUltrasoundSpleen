#!/usr/bin/env python3
"""
Setup validation script for DoRA SDXL project.
Verifies all components are properly installed and configured.
"""

import sys
import subprocess
from pathlib import Path
import importlib.util
from typing import Tuple, List


class SetupValidator:
    """Validates project setup and environment."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.errors = []
        self.warnings = []
        self.success_count = 0
    
    def check_python_version(self) -> bool:
        """Check Python version >= 3.10."""
        version = sys.version_info
        if version.major >= 3 and version.minor >= 10:
            self.print_success(f"Python {version.major}.{version.minor}.{version.micro} ✓")
            return True
        else:
            msg = f"Python 3.10+ required, found {version.major}.{version.minor}"
            self.print_error(msg)
            return False
    
    def check_directory_structure(self) -> bool:
        """Check if all required directories exist."""
        required_dirs = [
            "config",
            "data",
            "utils",
            "api",
            "checkpoints",
            "outputs",
        ]
        
        all_exist = True
        for dir_name in required_dirs:
            dir_path = self.project_root / dir_name
            if dir_path.exists():
                self.print_success(f"Directory '{dir_name}/' exists ✓")
            else:
                self.print_error(f"Directory '{dir_name}/' missing")
                all_exist = False
        
        return all_exist
    
    def check_required_files(self) -> bool:
        """Check if all required files exist."""
        required_files = [
            "config/dora_sdxl.yaml",
            "data/__init__.py",
            "data/dataset.py",
            "utils/__init__.py",
            "utils/logging.py",
            "utils/checkpoint.py",
            "api/__init__.py",
            "api/core.py",
            "api/app.py",
            "api/cli.py",
            "train_dora_sdxl.py",
            "inference.py",
            "requirements.txt",
            "README.md",
        ]
        
        all_exist = True
        for file_name in required_files:
            file_path = self.project_root / file_name
            if file_path.exists():
                size_kb = file_path.stat().st_size / 1024
                self.print_success(f"File '{file_name}' exists ({size_kb:.1f}KB) ✓")
            else:
                self.print_error(f"File '{file_name}' missing")
                all_exist = False
        
        return all_exist
    
    def check_package_imports(self) -> bool:
        """Check if core packages can be imported."""
        packages = {
            "torch": "PyTorch",
            "transformers": "Transformers",
            "diffusers": "Diffusers",
            "peft": "PEFT",
            "accelerate": "Accelerate",
            "PIL": "Pillow",
            "yaml": "PyYAML",
            "numpy": "NumPy",
        }
        
        all_installed = True
        for package, name in packages.items():
            if self._can_import(package):
                self.print_success(f"{name} is installed ✓")
            else:
                self.print_warning(f"{name} not installed (will be needed for training)")
                all_installed = False
        
        return all_installed
    
    def check_optional_packages(self) -> bool:
        """Check optional packages."""
        optional = {
            "wandb": "Weights & Biases",
            "flask": "Flask",
            "xformers": "xformers (memory optimization)",
        }
        
        for package, name in optional.items():
            if self._can_import(package):
                self.print_success(f"{name} is installed ✓")
            else:
                self.print_warning(f"{name} not installed (optional, improves performance)")
        
        return True
    
    def check_cuda_availability(self) -> bool:
        """Check if CUDA is available."""
        if self._can_import("torch"):
            try:
                import torch
                if torch.cuda.is_available():
                    device_count = torch.cuda.device_count()
                    device_name = torch.cuda.get_device_name(0)
                    self.print_success(f"CUDA available ({device_count} GPU(s): {device_name}) ✓")
                    return True
                else:
                    self.print_warning("CUDA not available (will use CPU, training will be slow)")
                    return False
            except Exception as e:
                self.print_warning(f"Could not check CUDA: {e}")
                return False
        return False
    
    def check_config_file(self) -> bool:
        """Validate configuration file."""
        config_path = self.project_root / "config/dora_sdxl.yaml"
        if not config_path.exists():
            self.print_error("Configuration file missing")
            return False
        
        try:
            if self._can_import("yaml"):
                import yaml
                with open(config_path) as f:
                    config = yaml.safe_load(f)
                
                required_keys = ["training", "dora", "data", "model", "validation"]
                missing_keys = [k for k in required_keys if k not in config]
                
                if missing_keys:
                    self.print_error(f"Config missing keys: {missing_keys}")
                    return False
                
                self.print_success(f"Configuration file is valid ✓")
                return True
            else:
                self.print_warning("PyYAML not installed, cannot validate config content")
                return True
        except Exception as e:
            self.print_error(f"Config file is invalid: {e}")
            return False
    
    def check_api_imports(self) -> bool:
        """Check if API modules can be imported."""
        try:
            sys.path.insert(0, str(self.project_root))
            
            # Try importing API components
            from api.core import DoRAProject, DoRAConfig
            self.print_success("API modules can be imported ✓")
            return True
        except ImportError as e:
            self.print_warning(f"API import error (expected if dependencies not installed): {e}")
            return False
        except Exception as e:
            self.print_error(f"API import failed: {e}")
            return False
    
    def check_datasets_available(self) -> bool:
        """Check if processed datasets are available."""
        dataset_dir = self.project_root / "Processed Datasets" / "Processed_NPZ_Dataset"
        
        if not dataset_dir.exists():
            self.print_warning("Processed dataset directory not found (run process_datasets.py first)")
            return False
        
        # Count NPZ files
        npz_files = list(dataset_dir.rglob("*.npz"))
        if npz_files:
            self.print_success(f"Found {len(npz_files)} processed dataset files ✓")
            return True
        else:
            self.print_warning("Processed dataset directory exists but is empty")
            return False
    
    def run_all_checks(self) -> bool:
        """Run all validation checks."""
        print("\n" + "=" * 70)
        print("DoRA SDXL Project - Setup Validation")
        print("=" * 70 + "\n")
        
        checks = [
            ("Python Version", self.check_python_version),
            ("Directory Structure", self.check_directory_structure),
            ("Required Files", self.check_required_files),
            ("Core Packages", self.check_package_imports),
            ("Optional Packages", self.check_optional_packages),
            ("CUDA Availability", self.check_cuda_availability),
            ("Configuration File", self.check_config_file),
            ("API Modules", self.check_api_imports),
            ("Datasets", self.check_datasets_available),
        ]
        
        results = []
        for check_name, check_func in checks:
            print(f"\n[{check_name}]")
            try:
                result = check_func()
                results.append((check_name, result))
            except Exception as e:
                self.print_error(f"Unexpected error: {e}")
                results.append((check_name, False))
        
        self.print_summary(results)
        return all(result for _, result in results)
    
    def print_success(self, message: str):
        """Print success message."""
        print(f"  ✓ {message}")
        self.success_count += 1
    
    def print_error(self, message: str):
        """Print error message."""
        print(f"  ✗ {message}")
        self.errors.append(message)
    
    def print_warning(self, message: str):
        """Print warning message."""
        print(f"  ⚠ {message}")
        self.warnings.append(message)
    
    def print_summary(self, results: List[Tuple[str, bool]]):
        """Print validation summary."""
        print("\n" + "=" * 70)
        print("VALIDATION SUMMARY")
        print("=" * 70)
        
        passed = sum(1 for _, result in results if result)
        total = len(results)
        
        print(f"\nPassed: {passed}/{total}")
        
        if self.errors:
            print(f"\nErrors ({len(self.errors)}):")
            for error in self.errors:
                print(f"  ✗ {error}")
        
        if self.warnings:
            print(f"\nWarnings ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  ⚠ {warning}")
        
        print("\n" + "=" * 70)
        
        if passed == total:
            print("✅ Setup is complete and ready!")
            print("\nNext steps:")
            print("  1. pip install -r requirements.txt  (if not done)")
            print("  2. python -m api.cli dataset load")
            print("  3. python -m api.cli training prepare")
            print("  4. python -m api.cli training start --epochs 5")
        elif passed >= total * 0.8:
            print("⚠️  Setup is mostly ready, but some optional components are missing")
            print("    Install remaining packages: pip install -r requirements.txt")
        else:
            print("❌ Setup has issues. Please address the errors above.")
        
        print("=" * 70 + "\n")
    
    @staticmethod
    def _can_import(package_name: str) -> bool:
        """Check if a package can be imported."""
        try:
            importlib.import_module(package_name)
            return True
        except ImportError:
            return False


def main():
    """Main entry point."""
    validator = SetupValidator()
    success = validator.run_all_checks()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
