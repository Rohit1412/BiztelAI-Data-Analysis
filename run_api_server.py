<<<<<<< HEAD
#!/usr/bin/env python
"""
BiztelAI Dataset API Server
---------------------------

This script starts the API server for the BiztelAI Dataset project.
It includes a setup wizard to help configure the environment and start the server.
"""

import os
import sys
import subprocess
import platform
import argparse
from pathlib import Path
import webbrowser
import time

def check_dependencies():
    """Check if all required dependencies are installed."""
    try:
        import flask
        import pandas
        import numpy
        import dotenv
        print("✓ All required packages are installed")
        return True
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        return False

def setup_environment():
    """Setup the environment for the API server."""
    # Check if .env file exists
    if not os.path.exists('.env'):
        print("Setting up environment...")
        
        # Copy the example environment file if it exists
        if os.path.exists('.env.example'):
            with open('.env.example', 'r') as example_file:
                example_content = example_file.read()
            
            with open('.env', 'w') as env_file:
                env_file.write(example_content)
            
            print("✓ Created .env file from example")
        else:
            print("✗ Could not find .env.example file")
            return False
    else:
        print("✓ Environment file (.env) already exists")
    
    return True

def check_data_files():
    """Check if required data files exist."""
    required_files = [
        "processed_data.csv",
        "BiztelAI_DS_Dataset_Mar'25.json"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"✗ Missing data files: {', '.join(missing_files)}")
        
        # Check if data pipeline exists
        if os.path.exists('data_pipeline.py'):
            print("You can generate the processed data by running the data pipeline:")
            print("  python data_pipeline.py")
        
        return False
    else:
        print("✓ All required data files exist")
        return True

def create_logs_directory():
    """Create logs directory if it doesn't exist."""
    if not os.path.exists('logs'):
        os.makedirs('logs')
        print("✓ Created logs directory")
    else:
        print("✓ Logs directory already exists")
    
    return True

def run_server(port, open_browser=True):
    """Run the API server."""
    print(f"Starting API server on port {port}...")
    
    # Set environment variable for port
    os.environ['PORT'] = str(port)
    
    # Run the server
    try:
        if open_browser:
            print("Opening API documentation in browser...")
            webbrowser.open(f"http://localhost:{port}/api", new=2)
        
        # Run the server using the Python executable
        python_exe = sys.executable
        subprocess.run([python_exe, "api_server.py"])
    except KeyboardInterrupt:
        print("\nServer stopped")
    except Exception as e:
        print(f"Error starting server: {e}")
        return False
    
    return True

def main():
    """Main function to run the API server."""
    parser = argparse.ArgumentParser(description='BiztelAI Dataset API Server')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the server on (default: 5000)')
    parser.add_argument('--no-browser', action='store_true', help='Do not open browser')
    parser.add_argument('--skip-checks', action='store_true', help='Skip dependency and data checks')
    
    args = parser.parse_args()
    
    print("=" * 40)
    print("BiztelAI Dataset API Server Setup")
    print("=" * 40)
    
    # Perform checks if not skipped
    if not args.skip_checks:
        # Check dependencies
        if not check_dependencies():
            print("\nPlease install missing dependencies:")
            print("  pip install -r requirements.txt")
            return 1
        
        # Check data files
        if not check_data_files():
            return 1
        
        # Setup environment
        if not setup_environment():
            return 1
        
        # Create logs directory
        if not create_logs_directory():
            return 1
    else:
        print("Skipping dependency and data checks")
    
    print("\n" + "=" * 40)
    print(f"Starting API server on port {args.port}")
    print("=" * 40)
    print("Press Ctrl+C to stop the server")
    print()
    
    # Run server
    return 0 if run_server(args.port, not args.no_browser) else 1

if __name__ == "__main__":
=======
#!/usr/bin/env python
"""
BiztelAI Dataset API Server
---------------------------

This script starts the API server for the BiztelAI Dataset project.
It includes a setup wizard to help configure the environment and start the server.
"""

import os
import sys
import subprocess
import platform
import argparse
from pathlib import Path
import webbrowser
import time

def check_dependencies():
    """Check if all required dependencies are installed."""
    try:
        import flask
        import pandas
        import numpy
        import dotenv
        print("✓ All required packages are installed")
        return True
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        return False

def setup_environment():
    """Setup the environment for the API server."""
    # Check if .env file exists
    if not os.path.exists('.env'):
        print("Setting up environment...")
        
        # Copy the example environment file if it exists
        if os.path.exists('.env.example'):
            with open('.env.example', 'r') as example_file:
                example_content = example_file.read()
            
            with open('.env', 'w') as env_file:
                env_file.write(example_content)
            
            print("✓ Created .env file from example")
        else:
            print("✗ Could not find .env.example file")
            return False
    else:
        print("✓ Environment file (.env) already exists")
    
    return True

def check_data_files():
    """Check if required data files exist."""
    required_files = [
        "processed_data.csv",
        "BiztelAI_DS_Dataset_Mar'25.json"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"✗ Missing data files: {', '.join(missing_files)}")
        
        # Check if data pipeline exists
        if os.path.exists('data_pipeline.py'):
            print("You can generate the processed data by running the data pipeline:")
            print("  python data_pipeline.py")
        
        return False
    else:
        print("✓ All required data files exist")
        return True

def create_logs_directory():
    """Create logs directory if it doesn't exist."""
    if not os.path.exists('logs'):
        os.makedirs('logs')
        print("✓ Created logs directory")
    else:
        print("✓ Logs directory already exists")
    
    return True

def run_server(port, open_browser=True):
    """Run the API server."""
    print(f"Starting API server on port {port}...")
    
    # Set environment variable for port
    os.environ['PORT'] = str(port)
    
    # Run the server
    try:
        if open_browser:
            print("Opening API documentation in browser...")
            webbrowser.open(f"http://localhost:{port}/api", new=2)
        
        # Run the server using the Python executable
        python_exe = sys.executable
        subprocess.run([python_exe, "api_server.py"])
    except KeyboardInterrupt:
        print("\nServer stopped")
    except Exception as e:
        print(f"Error starting server: {e}")
        return False
    
    return True

def main():
    """Main function to run the API server."""
    parser = argparse.ArgumentParser(description='BiztelAI Dataset API Server')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the server on (default: 5000)')
    parser.add_argument('--no-browser', action='store_true', help='Do not open browser')
    parser.add_argument('--skip-checks', action='store_true', help='Skip dependency and data checks')
    
    args = parser.parse_args()
    
    print("=" * 40)
    print("BiztelAI Dataset API Server Setup")
    print("=" * 40)
    
    # Perform checks if not skipped
    if not args.skip_checks:
        # Check dependencies
        if not check_dependencies():
            print("\nPlease install missing dependencies:")
            print("  pip install -r requirements.txt")
            return 1
        
        # Check data files
        if not check_data_files():
            return 1
        
        # Setup environment
        if not setup_environment():
            return 1
        
        # Create logs directory
        if not create_logs_directory():
            return 1
    else:
        print("Skipping dependency and data checks")
    
    print("\n" + "=" * 40)
    print(f"Starting API server on port {args.port}")
    print("=" * 40)
    print("Press Ctrl+C to stop the server")
    print()
    
    # Run server
    return 0 if run_server(args.port, not args.no_browser) else 1

if __name__ == "__main__":
>>>>>>> 5d96285edcdaab24a8fcf7efb1d7ec4f7b04bb22
    sys.exit(main()) 