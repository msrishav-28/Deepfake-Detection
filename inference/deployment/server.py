# inference/deployment/server.py
import os
import argparse
from .api import create_api, run_server
from .utils import load_deployment_config


def main():
    """Main function for running the deployment server"""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Deepfake Detection Deployment Server")
    parser.add_argument("--config", type=str, default="inference/deployment/config/deployment_config.yaml",
                       help="Path to deployment configuration file")
    parser.add_argument("--host", type=str, help="Host to bind (overrides config)")
    parser.add_argument("--port", type=int, help="Port to bind (overrides config)")
    args = parser.parse_args()
    
    # Load configuration
    config = load_deployment_config(args.config)
    
    # Get server settings
    server_config = config.get("server", {})
    host = args.host or server_config.get("host", "0.0.0.0")
    port = args.port or server_config.get("port", 8000)
    
    # Create API
    app = create_api(args.config)
    
    # Print startup message
    print(f"Starting Deepfake Detection API server at http://{host}:{port}")
    print(f"API documentation available at http://{host}:{port}/docs")
    
    # Run server
    run_server(app, host=host, port=port)


if __name__ == "__main__":
    main()