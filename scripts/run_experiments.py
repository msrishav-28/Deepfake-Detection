#!/usr/bin/env python3
"""
Experiment runner script for deepfake detection experiments
"""

import os
import sys
import yaml
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

def load_experiment_config(config_path):
    """Load experiment configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_single_experiment(experiment, base_config):
    """Run a single experiment"""
    print(f"\n🚀 Starting experiment: {experiment['name']}")
    print(f"   Type: {experiment['type']}")
    print(f"   Config: {experiment['config']}")
    
    # Create output directory
    output_dir = experiment.get('overrides', {}).get('experiment.output_dir', 
                                                    f"experiments/{experiment['name']}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Run experiment based on type
    try:
        if experiment['type'] == 'training':
            cmd = [
                sys.executable, "training/train.py",
                "--config", experiment['config'],
                "--output", output_dir
            ]
            
        elif experiment['type'] == 'evaluation':
            cmd = [
                sys.executable, "evaluation/evaluate.py", 
                "--config", experiment['config'],
                "--output", output_dir
            ]
            
        elif experiment['type'] == 'fusion':
            cmd = [
                sys.executable, "fusion/train_fusion.py",
                "--config", experiment['config'],
                "--output", output_dir
            ]
            
        else:
            raise ValueError(f"Unknown experiment type: {experiment['type']}")
        
        # Add overrides as command line arguments
        overrides = experiment.get('overrides', {})
        for key, value in overrides.items():
            cmd.extend([f"--{key.replace('.', '_')}", str(value)])
        
        # Run the command
        print(f"   Command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ Experiment {experiment['name']} completed successfully")
            return True
        else:
            print(f"❌ Experiment {experiment['name']} failed")
            print(f"   Error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Experiment {experiment['name']} failed with exception: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run deepfake detection experiments")
    parser.add_argument("--config", type=str, required=True, 
                       help="Path to experiment configuration file")
    parser.add_argument("--experiment", type=str, 
                       help="Specific experiment to run (optional)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be run without executing")
    args = parser.parse_args()
    
    # Load experiment configuration
    config = load_experiment_config(args.config)
    
    print(f"🔬 Experiment Suite: {config['experiment']['name']}")
    print(f"   Description: {config['experiment']['description']}")
    print(f"   Output Dir: {config['experiment']['output_dir']}")
    
    # Filter experiments if specific one requested
    experiments = config['experiments']
    if args.experiment:
        experiments = [exp for exp in experiments if exp['name'] == args.experiment]
        if not experiments:
            print(f"❌ Experiment '{args.experiment}' not found")
            return 1
    
    print(f"\n📋 Found {len(experiments)} experiments to run:")
    for exp in experiments:
        print(f"   - {exp['name']} ({exp['type']})")
    
    if args.dry_run:
        print("\n🔍 Dry run mode - no experiments will be executed")
        return 0
    
    # Create main experiment directory
    os.makedirs(config['experiment']['output_dir'], exist_ok=True)
    
    # Run experiments
    results = []
    for i, experiment in enumerate(experiments, 1):
        print(f"\n📊 Progress: [{i}/{len(experiments)}]")
        success = run_single_experiment(experiment, config)
        results.append({
            'name': experiment['name'],
            'success': success,
            'timestamp': datetime.now().isoformat()
        })
        
        # Check if we should continue on error
        if not success and not config.get('execution', {}).get('continue_on_error', True):
            print("❌ Stopping due to error (continue_on_error=False)")
            break
    
    # Print summary
    print(f"\n📈 Experiment Summary:")
    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful
    
    print(f"   ✅ Successful: {successful}")
    print(f"   ❌ Failed: {failed}")
    print(f"   📊 Total: {len(results)}")
    
    # Save results
    results_file = os.path.join(config['experiment']['output_dir'], 'experiment_results.yaml')
    with open(results_file, 'w') as f:
        yaml.dump({
            'experiment': config['experiment'],
            'results': results,
            'summary': {
                'successful': successful,
                'failed': failed,
                'total': len(results)
            }
        }, f)
    
    print(f"\n💾 Results saved to: {results_file}")
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())