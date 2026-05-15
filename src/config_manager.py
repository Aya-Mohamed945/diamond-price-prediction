
"""Configuration management module"""

import yaml
from pathlib import Path
from typing import Any, Dict


class ConfigManager:
    """Manage all configuration settings"""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.configs = {}
        self.load_all_configs()
        
    def load_config(self, config_name: str) -> Dict[str, Any]:
        """Load a specific configuration file"""
        config_path = self.config_dir / f"{config_name}.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def load_all_configs(self):
        """Load all configuration files"""
        config_files = ['config', 'features', 'preprocessing']
        
        for config_file in config_files:
            self.configs[config_file] = self.load_config(config_file)
            
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation"""
        parts = key.split('.')
        current = self.configs.get(parts[0], {})
        
        for part in parts[1:]:
            if isinstance(current, dict):
                current = current.get(part, default)
            else:
                return default
                
        return current
    
    def update_config(self, config_name: str, updates: Dict[str, Any]):
        """Update configuration dynamically"""
        if config_name not in self.configs:
            self.configs[config_name] = {}
            
        self._deep_update(self.configs[config_name], updates)
        
    def _deep_update(self, base_dict: Dict, updates: Dict):
        """Deep update nested dictionary"""
        for key, value in updates.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
                
    def save_config(self, config_name: str):
        """Save configuration back to file"""
        config_path = self.config_dir / f"{config_name}.yaml"
        
        with open(config_path, 'w') as f:
            yaml.dump(self.configs[config_name], f, default_flow_style=False)
            
    def display_config(self, config_name: str = None):
        """Display configuration in a readable format"""
        if config_name:
            configs_to_display = {config_name: self.configs[config_name]}
        else:
            configs_to_display = self.configs
            
        for name, config in configs_to_display.items():
            print(f"\n{'='*50}")
            print(f"Configuration: {name}")
            print(f"{'='*50}")
            print(yaml.dump(config, default_flow_style=False))


# Example usage in main.py
if __name__ == "__main__":
    config = ConfigManager()
    
    # Get specific values
    test_size = config.get('config.data.test_size')
    model_enabled = config.get('config.models.knn.enabled')
    cv_folds = config.get('config.tuning.cv_folds')
    
    print(f"Test size: {test_size}")
    print(f"KNN Enabled: {model_enabled}")
    print(f"CV Folds: {cv_folds}")
    
    # Display all configurations
    config.display_config()