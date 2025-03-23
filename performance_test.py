<<<<<<< HEAD
#!/usr/bin/env python
"""
Performance testing script for BiztelAI API

This script tests the performance of the API by sending parallel requests
and measuring response times.
"""
import time
import json
import asyncio
import argparse
import logging
import statistics
import aiohttp
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from tabulate import tabulate

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("performance_test")

# Default test configuration
DEFAULT_BASE_URL = "http://localhost:5000"
DEFAULT_ENDPOINT = "/api/health"
DEFAULT_CONCURRENCY = 10
DEFAULT_REQUESTS = 100
DEFAULT_AUTH_ENDPOINT = "/api/login"
DEFAULT_USERNAME = "user"
DEFAULT_PASSWORD = "user123"

class PerformanceTester:
    """Class for testing API performance."""
    
    def __init__(self, base_url, auth_endpoint=None, username=None, password=None):
        """Initialize with API base URL and optional authentication."""
        self.base_url = base_url
        self.auth_endpoint = auth_endpoint
        self.username = username
        self.password = password
        self.token = None
        self.results = {}
    
    async def authenticate(self):
        """Authenticate with the API and get a JWT token."""
        if not all([self.auth_endpoint, self.username, self.password]):
            logger.info("Authentication not configured, skipping.")
            return
        
        async with aiohttp.ClientSession() as session:
            auth_url = f"{self.base_url}{self.auth_endpoint}"
            payload = {
                "username": self.username,
                "password": self.password
            }
            
            try:
                logger.info(f"Authenticating as {self.username}...")
                start_time = time.time()
                async with session.post(auth_url, json=payload) as response:
                    duration = time.time() - start_time
                    if response.status == 200:
                        data = await response.json()
                        self.token = data.get("access_token")
                        logger.info(f"Authentication successful in {duration:.2f}s")
                    else:
                        text = await response.text()
                        logger.error(f"Authentication failed: {response.status} - {text}")
            except Exception as e:
                logger.error(f"Error during authentication: {str(e)}")
    
    async def _make_request(self, session, endpoint, request_id):
        """Make a single request to the API."""
        url = f"{self.base_url}{endpoint}"
        headers = {}
        
        # Add authentication if available
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        
        start_time = time.time()
        try:
            async with session.get(url, headers=headers) as response:
                duration = time.time() - start_time
                status = response.status
                try:
                    data = await response.json()
                except:
                    data = await response.text()
                
                return {
                    "id": request_id,
                    "status": status,
                    "duration": duration,
                    "success": 200 <= status < 300
                }
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Error in request {request_id}: {str(e)}")
            return {
                "id": request_id,
                "status": 0,
                "duration": duration,
                "success": False,
                "error": str(e)
            }
    
    async def test_endpoint(self, endpoint, num_requests, concurrency):
        """Test an API endpoint with parallel requests."""
        logger.info(f"Testing endpoint {endpoint} with {num_requests} requests ({concurrency} concurrent)...")
        
        # Authenticate if credentials provided
        if self.auth_endpoint and not self.token:
            await self.authenticate()
        
        results = []
        
        # Create async tasks for requests
        async with aiohttp.ClientSession() as session:
            tasks = []
            for i in range(num_requests):
                task = asyncio.create_task(self._make_request(session, endpoint, i+1))
                tasks.append(task)
                
                # Limit concurrency
                if len(tasks) >= concurrency:
                    # Wait for some tasks to complete before adding more
                    completed, tasks = await asyncio.wait(
                        tasks, return_when=asyncio.FIRST_COMPLETED
                    )
                    results.extend([task.result() for task in completed])
            
            # Wait for remaining tasks
            if tasks:
                completed, _ = await asyncio.wait(tasks)
                results.extend([task.result() for task in completed])
        
        # Sort results by ID
        results.sort(key=lambda x: x["id"])
        
        # Calculate statistics
        durations = [r["duration"] for r in results]
        success_count = sum(1 for r in results if r["success"])
        
        stats = {
            "endpoint": endpoint,
            "requests": num_requests,
            "concurrency": concurrency,
            "success_rate": success_count / num_requests if num_requests > 0 else 0,
            "min_time": min(durations) if durations else 0,
            "max_time": max(durations) if durations else 0,
            "avg_time": statistics.mean(durations) if durations else 0,
            "median_time": statistics.median(durations) if durations else 0,
            "p95_time": np.percentile(durations, 95) if durations else 0,
            "total_time": sum(durations)
        }
        
        # Store results
        self.results[endpoint] = {
            "stats": stats,
            "requests": results
        }
        
        logger.info(f"Test completed for {endpoint}")
        logger.info(f"Success rate: {stats['success_rate']:.2%}")
        logger.info(f"Average response time: {stats['avg_time']:.4f}s")
        logger.info(f"95th percentile: {stats['p95_time']:.4f}s")
        
        return stats
    
    def print_results(self):
        """Print test results in tabular format."""
        if not self.results:
            logger.info("No test results to display.")
            return
        
        headers = ["Endpoint", "Requests", "Concurrency", "Success", "Avg (s)", "Median (s)", "P95 (s)", "Min (s)", "Max (s)"]
        rows = []
        
        for endpoint, result in self.results.items():
            stats = result["stats"]
            rows.append([
                endpoint,
                stats["requests"],
                stats["concurrency"],
                f"{stats['success_rate']:.2%}",
                f"{stats['avg_time']:.4f}",
                f"{stats['median_time']:.4f}",
                f"{stats['p95_time']:.4f}",
                f"{stats['min_time']:.4f}",
                f"{stats['max_time']:.4f}"
            ])
        
        print("\n" + tabulate(rows, headers=headers, tablefmt="grid"))
    
    def visualize_results(self, output_file=None):
        """Generate visualization of test results."""
        if not self.results:
            logger.info("No test results to visualize.")
            return
        
        # Prepare data for visualization
        endpoints = list(self.results.keys())
        avg_times = [self.results[endpoint]["stats"]["avg_time"] for endpoint in endpoints]
        p95_times = [self.results[endpoint]["stats"]["p95_time"] for endpoint in endpoints]
        success_rates = [self.results[endpoint]["stats"]["success_rate"] * 100 for endpoint in endpoints]
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Response time plot
        x = np.arange(len(endpoints))
        width = 0.35
        
        ax1.bar(x - width/2, avg_times, width, label='Average Time (s)')
        ax1.bar(x + width/2, p95_times, width, label='95th Percentile (s)')
        
        ax1.set_ylabel('Response Time (seconds)')
        ax1.set_title('API Response Time by Endpoint')
        ax1.set_xticks(x)
        ax1.set_xticklabels(endpoints)
        ax1.legend()
        
        # Success rate plot
        ax2.bar(endpoints, success_rates, color='green')
        ax2.set_ylim([0, 105])  # 0-105% range
        ax2.set_ylabel('Success Rate (%)')
        ax2.set_title('API Success Rate by Endpoint')
        
        # Add labels
        for i, v in enumerate(success_rates):
            ax2.text(i, v + 2, f"{v:.1f}%", ha='center')
        
        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fig.text(0.5, 0.01, f"Generated: {timestamp}", ha='center')
        
        plt.tight_layout()
        
        # Save to file if specified
        if output_file:
            plt.savefig(output_file)
            logger.info(f"Results visualization saved to {output_file}")
        
        # Show plot
        plt.show()
    
    def save_results(self, output_file):
        """Save test results to a JSON file."""
        if not self.results:
            logger.info("No test results to save.")
            return
        
        try:
            with open(output_file, 'w') as f:
                json.dump(self.results, f, indent=2)
            logger.info(f"Results saved to {output_file}")
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")

async def main():
    """Main function to run performance tests."""
    parser = argparse.ArgumentParser(description='BiztelAI API Performance Tester')
    parser.add_argument('--url', default=DEFAULT_BASE_URL, help='Base URL of the API')
    parser.add_argument('--endpoint', default=DEFAULT_ENDPOINT, help='API endpoint to test')
    parser.add_argument('--requests', type=int, default=DEFAULT_REQUESTS, help='Number of requests to send')
    parser.add_argument('--concurrency', type=int, default=DEFAULT_CONCURRENCY, help='Number of concurrent requests')
    parser.add_argument('--auth', action='store_true', help='Authenticate before testing')
    parser.add_argument('--username', default=DEFAULT_USERNAME, help='Username for authentication')
    parser.add_argument('--password', default=DEFAULT_PASSWORD, help='Password for authentication')
    parser.add_argument('--save', type=str, help='Save results to JSON file')
    parser.add_argument('--visualize', action='store_true', help='Generate visualization of results')
    parser.add_argument('--output-image', type=str, help='Save visualization to image file')
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = PerformanceTester(
        base_url=args.url,
        auth_endpoint=DEFAULT_AUTH_ENDPOINT if args.auth else None,
        username=args.username if args.auth else None,
        password=args.password if args.auth else None
    )
    
    # Run test
    await tester.test_endpoint(args.endpoint, args.requests, args.concurrency)
    
    # Print results
    tester.print_results()
    
    # Save results if specified
    if args.save:
        tester.save_results(args.save)
    
    # Visualize results if specified
    if args.visualize:
        tester.visualize_results(args.output_image)

if __name__ == "__main__":
=======
#!/usr/bin/env python
"""
Performance testing script for BiztelAI API

This script tests the performance of the API by sending parallel requests
and measuring response times.
"""
import time
import json
import asyncio
import argparse
import logging
import statistics
import aiohttp
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from tabulate import tabulate

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("performance_test")

# Default test configuration
DEFAULT_BASE_URL = "http://localhost:5000"
DEFAULT_ENDPOINT = "/api/health"
DEFAULT_CONCURRENCY = 10
DEFAULT_REQUESTS = 100
DEFAULT_AUTH_ENDPOINT = "/api/login"
DEFAULT_USERNAME = "user"
DEFAULT_PASSWORD = "user123"

class PerformanceTester:
    """Class for testing API performance."""
    
    def __init__(self, base_url, auth_endpoint=None, username=None, password=None):
        """Initialize with API base URL and optional authentication."""
        self.base_url = base_url
        self.auth_endpoint = auth_endpoint
        self.username = username
        self.password = password
        self.token = None
        self.results = {}
    
    async def authenticate(self):
        """Authenticate with the API and get a JWT token."""
        if not all([self.auth_endpoint, self.username, self.password]):
            logger.info("Authentication not configured, skipping.")
            return
        
        async with aiohttp.ClientSession() as session:
            auth_url = f"{self.base_url}{self.auth_endpoint}"
            payload = {
                "username": self.username,
                "password": self.password
            }
            
            try:
                logger.info(f"Authenticating as {self.username}...")
                start_time = time.time()
                async with session.post(auth_url, json=payload) as response:
                    duration = time.time() - start_time
                    if response.status == 200:
                        data = await response.json()
                        self.token = data.get("access_token")
                        logger.info(f"Authentication successful in {duration:.2f}s")
                    else:
                        text = await response.text()
                        logger.error(f"Authentication failed: {response.status} - {text}")
            except Exception as e:
                logger.error(f"Error during authentication: {str(e)}")
    
    async def _make_request(self, session, endpoint, request_id):
        """Make a single request to the API."""
        url = f"{self.base_url}{endpoint}"
        headers = {}
        
        # Add authentication if available
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        
        start_time = time.time()
        try:
            async with session.get(url, headers=headers) as response:
                duration = time.time() - start_time
                status = response.status
                try:
                    data = await response.json()
                except:
                    data = await response.text()
                
                return {
                    "id": request_id,
                    "status": status,
                    "duration": duration,
                    "success": 200 <= status < 300
                }
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Error in request {request_id}: {str(e)}")
            return {
                "id": request_id,
                "status": 0,
                "duration": duration,
                "success": False,
                "error": str(e)
            }
    
    async def test_endpoint(self, endpoint, num_requests, concurrency):
        """Test an API endpoint with parallel requests."""
        logger.info(f"Testing endpoint {endpoint} with {num_requests} requests ({concurrency} concurrent)...")
        
        # Authenticate if credentials provided
        if self.auth_endpoint and not self.token:
            await self.authenticate()
        
        results = []
        
        # Create async tasks for requests
        async with aiohttp.ClientSession() as session:
            tasks = []
            for i in range(num_requests):
                task = asyncio.create_task(self._make_request(session, endpoint, i+1))
                tasks.append(task)
                
                # Limit concurrency
                if len(tasks) >= concurrency:
                    # Wait for some tasks to complete before adding more
                    completed, tasks = await asyncio.wait(
                        tasks, return_when=asyncio.FIRST_COMPLETED
                    )
                    results.extend([task.result() for task in completed])
            
            # Wait for remaining tasks
            if tasks:
                completed, _ = await asyncio.wait(tasks)
                results.extend([task.result() for task in completed])
        
        # Sort results by ID
        results.sort(key=lambda x: x["id"])
        
        # Calculate statistics
        durations = [r["duration"] for r in results]
        success_count = sum(1 for r in results if r["success"])
        
        stats = {
            "endpoint": endpoint,
            "requests": num_requests,
            "concurrency": concurrency,
            "success_rate": success_count / num_requests if num_requests > 0 else 0,
            "min_time": min(durations) if durations else 0,
            "max_time": max(durations) if durations else 0,
            "avg_time": statistics.mean(durations) if durations else 0,
            "median_time": statistics.median(durations) if durations else 0,
            "p95_time": np.percentile(durations, 95) if durations else 0,
            "total_time": sum(durations)
        }
        
        # Store results
        self.results[endpoint] = {
            "stats": stats,
            "requests": results
        }
        
        logger.info(f"Test completed for {endpoint}")
        logger.info(f"Success rate: {stats['success_rate']:.2%}")
        logger.info(f"Average response time: {stats['avg_time']:.4f}s")
        logger.info(f"95th percentile: {stats['p95_time']:.4f}s")
        
        return stats
    
    def print_results(self):
        """Print test results in tabular format."""
        if not self.results:
            logger.info("No test results to display.")
            return
        
        headers = ["Endpoint", "Requests", "Concurrency", "Success", "Avg (s)", "Median (s)", "P95 (s)", "Min (s)", "Max (s)"]
        rows = []
        
        for endpoint, result in self.results.items():
            stats = result["stats"]
            rows.append([
                endpoint,
                stats["requests"],
                stats["concurrency"],
                f"{stats['success_rate']:.2%}",
                f"{stats['avg_time']:.4f}",
                f"{stats['median_time']:.4f}",
                f"{stats['p95_time']:.4f}",
                f"{stats['min_time']:.4f}",
                f"{stats['max_time']:.4f}"
            ])
        
        print("\n" + tabulate(rows, headers=headers, tablefmt="grid"))
    
    def visualize_results(self, output_file=None):
        """Generate visualization of test results."""
        if not self.results:
            logger.info("No test results to visualize.")
            return
        
        # Prepare data for visualization
        endpoints = list(self.results.keys())
        avg_times = [self.results[endpoint]["stats"]["avg_time"] for endpoint in endpoints]
        p95_times = [self.results[endpoint]["stats"]["p95_time"] for endpoint in endpoints]
        success_rates = [self.results[endpoint]["stats"]["success_rate"] * 100 for endpoint in endpoints]
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Response time plot
        x = np.arange(len(endpoints))
        width = 0.35
        
        ax1.bar(x - width/2, avg_times, width, label='Average Time (s)')
        ax1.bar(x + width/2, p95_times, width, label='95th Percentile (s)')
        
        ax1.set_ylabel('Response Time (seconds)')
        ax1.set_title('API Response Time by Endpoint')
        ax1.set_xticks(x)
        ax1.set_xticklabels(endpoints)
        ax1.legend()
        
        # Success rate plot
        ax2.bar(endpoints, success_rates, color='green')
        ax2.set_ylim([0, 105])  # 0-105% range
        ax2.set_ylabel('Success Rate (%)')
        ax2.set_title('API Success Rate by Endpoint')
        
        # Add labels
        for i, v in enumerate(success_rates):
            ax2.text(i, v + 2, f"{v:.1f}%", ha='center')
        
        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fig.text(0.5, 0.01, f"Generated: {timestamp}", ha='center')
        
        plt.tight_layout()
        
        # Save to file if specified
        if output_file:
            plt.savefig(output_file)
            logger.info(f"Results visualization saved to {output_file}")
        
        # Show plot
        plt.show()
    
    def save_results(self, output_file):
        """Save test results to a JSON file."""
        if not self.results:
            logger.info("No test results to save.")
            return
        
        try:
            with open(output_file, 'w') as f:
                json.dump(self.results, f, indent=2)
            logger.info(f"Results saved to {output_file}")
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")

async def main():
    """Main function to run performance tests."""
    parser = argparse.ArgumentParser(description='BiztelAI API Performance Tester')
    parser.add_argument('--url', default=DEFAULT_BASE_URL, help='Base URL of the API')
    parser.add_argument('--endpoint', default=DEFAULT_ENDPOINT, help='API endpoint to test')
    parser.add_argument('--requests', type=int, default=DEFAULT_REQUESTS, help='Number of requests to send')
    parser.add_argument('--concurrency', type=int, default=DEFAULT_CONCURRENCY, help='Number of concurrent requests')
    parser.add_argument('--auth', action='store_true', help='Authenticate before testing')
    parser.add_argument('--username', default=DEFAULT_USERNAME, help='Username for authentication')
    parser.add_argument('--password', default=DEFAULT_PASSWORD, help='Password for authentication')
    parser.add_argument('--save', type=str, help='Save results to JSON file')
    parser.add_argument('--visualize', action='store_true', help='Generate visualization of results')
    parser.add_argument('--output-image', type=str, help='Save visualization to image file')
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = PerformanceTester(
        base_url=args.url,
        auth_endpoint=DEFAULT_AUTH_ENDPOINT if args.auth else None,
        username=args.username if args.auth else None,
        password=args.password if args.auth else None
    )
    
    # Run test
    await tester.test_endpoint(args.endpoint, args.requests, args.concurrency)
    
    # Print results
    tester.print_results()
    
    # Save results if specified
    if args.save:
        tester.save_results(args.save)
    
    # Visualize results if specified
    if args.visualize:
        tester.visualize_results(args.output_image)

if __name__ == "__main__":
>>>>>>> 5d96285edcdaab24a8fcf7efb1d7ec4f7b04bb22
    asyncio.run(main()) 