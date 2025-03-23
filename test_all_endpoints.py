<<<<<<< HEAD
#!/usr/bin/env python
"""
Script to test performance of all BiztelAI API endpoints.
"""
import os
import asyncio
import argparse
import logging
import json
from datetime import datetime
from performance_test import PerformanceTester

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_all_endpoints")

# Default API endpoints to test
DEFAULT_ENDPOINTS = [
    "/api/health",
    "/api/dataset/summary",
    "/api/dataset/transform",
    "/api/analysis/transcript"
]

# Default test configuration
DEFAULT_BASE_URL = "http://localhost:5000"
DEFAULT_CONCURRENCY = 10
DEFAULT_REQUESTS = 50
DEFAULT_AUTH_ENDPOINT = "/api/login"
DEFAULT_USERNAME = "user"
DEFAULT_PASSWORD = "user123"

async def main():
    """Run performance tests on all specified endpoints."""
    parser = argparse.ArgumentParser(description='BiztelAI API Performance Test Suite')
    parser.add_argument('--url', default=DEFAULT_BASE_URL, help='Base URL of the API')
    parser.add_argument('--endpoints', nargs='+', default=None, help='List of endpoints to test')
    parser.add_argument('--endpoints-file', type=str, help='JSON file containing endpoints to test')
    parser.add_argument('--requests', type=int, default=DEFAULT_REQUESTS, help='Number of requests per endpoint')
    parser.add_argument('--concurrency', type=int, default=DEFAULT_CONCURRENCY, help='Number of concurrent requests')
    parser.add_argument('--auth', action='store_true', help='Authenticate before testing')
    parser.add_argument('--username', default=DEFAULT_USERNAME, help='Username for authentication')
    parser.add_argument('--password', default=DEFAULT_PASSWORD, help='Password for authentication')
    parser.add_argument('--output-dir', default='performance_results', help='Directory to save results')
    parser.add_argument('--visualize', action='store_true', help='Generate visualization of results')
    
    args = parser.parse_args()
    
    # Determine which endpoints to test
    endpoints_to_test = DEFAULT_ENDPOINTS
    
    if args.endpoints:
        endpoints_to_test = args.endpoints
    elif args.endpoints_file:
        try:
            with open(args.endpoints_file, 'r') as f:
                endpoints_data = json.load(f)
                if isinstance(endpoints_data, list):
                    endpoints_to_test = endpoints_data
                elif isinstance(endpoints_data, dict) and 'endpoints' in endpoints_data:
                    endpoints_to_test = endpoints_data['endpoints']
        except Exception as e:
            logger.error(f"Failed to load endpoints from file: {str(e)}")
    
    logger.info(f"Starting performance tests for {len(endpoints_to_test)} endpoints")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Create timestamp for this test run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(results_dir)
    
    # Initialize tester
    tester = PerformanceTester(
        base_url=args.url,
        auth_endpoint=DEFAULT_AUTH_ENDPOINT if args.auth else None,
        username=args.username if args.auth else None,
        password=args.password if args.auth else None
    )
    
    # Run tests for each endpoint
    all_stats = []
    for endpoint in endpoints_to_test:
        logger.info(f"Testing endpoint: {endpoint}")
        stats = await tester.test_endpoint(endpoint, args.requests, args.concurrency)
        all_stats.append(stats)
        
        # Save individual endpoint results
        endpoint_name = endpoint.replace('/', '_').strip('_')
        results_file = os.path.join(results_dir, f"{endpoint_name}.json")
        with open(results_file, 'w') as f:
            json.dump(tester.results[endpoint], f, indent=2)
    
    # Print summary of all results
    tester.print_results()
    
    # Save combined results
    combined_results_file = os.path.join(results_dir, "all_results.json")
    with open(combined_results_file, 'w') as f:
        json.dump(tester.results, f, indent=2)
    
    # Generate summary report
    summary_file = os.path.join(results_dir, "summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"BiztelAI API Performance Test Summary\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Base URL: {args.url}\n")
        f.write(f"Requests per endpoint: {args.requests}\n")
        f.write(f"Concurrency level: {args.concurrency}\n")
        f.write(f"Authentication: {'Enabled' if args.auth else 'Disabled'}\n\n")
        
        f.write("Results Summary:\n")
        for stats in all_stats:
            f.write(f"\nEndpoint: {stats['endpoint']}\n")
            f.write(f"  Success Rate: {stats['success_rate']:.2%}\n")
            f.write(f"  Average Response Time: {stats['avg_time']:.4f}s\n")
            f.write(f"  Median Response Time: {stats['median_time']:.4f}s\n")
            f.write(f"  95th Percentile: {stats['p95_time']:.4f}s\n")
            f.write(f"  Min/Max Time: {stats['min_time']:.4f}s / {stats['max_time']:.4f}s\n")
    
    # Visualize results if requested
    if args.visualize:
        visualization_file = os.path.join(results_dir, "performance_visualization.png")
        tester.visualize_results(visualization_file)
    
    logger.info(f"All tests completed. Results saved to {results_dir}")
    logger.info(f"Summary report: {summary_file}")

if __name__ == "__main__":
=======
#!/usr/bin/env python
"""
Script to test performance of all BiztelAI API endpoints.
"""
import os
import asyncio
import argparse
import logging
import json
from datetime import datetime
from performance_test import PerformanceTester

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_all_endpoints")

# Default API endpoints to test
DEFAULT_ENDPOINTS = [
    "/api/health",
    "/api/dataset/summary",
    "/api/dataset/transform",
    "/api/analysis/transcript"
]

# Default test configuration
DEFAULT_BASE_URL = "http://localhost:5000"
DEFAULT_CONCURRENCY = 10
DEFAULT_REQUESTS = 50
DEFAULT_AUTH_ENDPOINT = "/api/login"
DEFAULT_USERNAME = "user"
DEFAULT_PASSWORD = "user123"

async def main():
    """Run performance tests on all specified endpoints."""
    parser = argparse.ArgumentParser(description='BiztelAI API Performance Test Suite')
    parser.add_argument('--url', default=DEFAULT_BASE_URL, help='Base URL of the API')
    parser.add_argument('--endpoints', nargs='+', default=None, help='List of endpoints to test')
    parser.add_argument('--endpoints-file', type=str, help='JSON file containing endpoints to test')
    parser.add_argument('--requests', type=int, default=DEFAULT_REQUESTS, help='Number of requests per endpoint')
    parser.add_argument('--concurrency', type=int, default=DEFAULT_CONCURRENCY, help='Number of concurrent requests')
    parser.add_argument('--auth', action='store_true', help='Authenticate before testing')
    parser.add_argument('--username', default=DEFAULT_USERNAME, help='Username for authentication')
    parser.add_argument('--password', default=DEFAULT_PASSWORD, help='Password for authentication')
    parser.add_argument('--output-dir', default='performance_results', help='Directory to save results')
    parser.add_argument('--visualize', action='store_true', help='Generate visualization of results')
    
    args = parser.parse_args()
    
    # Determine which endpoints to test
    endpoints_to_test = DEFAULT_ENDPOINTS
    
    if args.endpoints:
        endpoints_to_test = args.endpoints
    elif args.endpoints_file:
        try:
            with open(args.endpoints_file, 'r') as f:
                endpoints_data = json.load(f)
                if isinstance(endpoints_data, list):
                    endpoints_to_test = endpoints_data
                elif isinstance(endpoints_data, dict) and 'endpoints' in endpoints_data:
                    endpoints_to_test = endpoints_data['endpoints']
        except Exception as e:
            logger.error(f"Failed to load endpoints from file: {str(e)}")
    
    logger.info(f"Starting performance tests for {len(endpoints_to_test)} endpoints")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Create timestamp for this test run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(results_dir)
    
    # Initialize tester
    tester = PerformanceTester(
        base_url=args.url,
        auth_endpoint=DEFAULT_AUTH_ENDPOINT if args.auth else None,
        username=args.username if args.auth else None,
        password=args.password if args.auth else None
    )
    
    # Run tests for each endpoint
    all_stats = []
    for endpoint in endpoints_to_test:
        logger.info(f"Testing endpoint: {endpoint}")
        stats = await tester.test_endpoint(endpoint, args.requests, args.concurrency)
        all_stats.append(stats)
        
        # Save individual endpoint results
        endpoint_name = endpoint.replace('/', '_').strip('_')
        results_file = os.path.join(results_dir, f"{endpoint_name}.json")
        with open(results_file, 'w') as f:
            json.dump(tester.results[endpoint], f, indent=2)
    
    # Print summary of all results
    tester.print_results()
    
    # Save combined results
    combined_results_file = os.path.join(results_dir, "all_results.json")
    with open(combined_results_file, 'w') as f:
        json.dump(tester.results, f, indent=2)
    
    # Generate summary report
    summary_file = os.path.join(results_dir, "summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"BiztelAI API Performance Test Summary\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Base URL: {args.url}\n")
        f.write(f"Requests per endpoint: {args.requests}\n")
        f.write(f"Concurrency level: {args.concurrency}\n")
        f.write(f"Authentication: {'Enabled' if args.auth else 'Disabled'}\n\n")
        
        f.write("Results Summary:\n")
        for stats in all_stats:
            f.write(f"\nEndpoint: {stats['endpoint']}\n")
            f.write(f"  Success Rate: {stats['success_rate']:.2%}\n")
            f.write(f"  Average Response Time: {stats['avg_time']:.4f}s\n")
            f.write(f"  Median Response Time: {stats['median_time']:.4f}s\n")
            f.write(f"  95th Percentile: {stats['p95_time']:.4f}s\n")
            f.write(f"  Min/Max Time: {stats['min_time']:.4f}s / {stats['max_time']:.4f}s\n")
    
    # Visualize results if requested
    if args.visualize:
        visualization_file = os.path.join(results_dir, "performance_visualization.png")
        tester.visualize_results(visualization_file)
    
    logger.info(f"All tests completed. Results saved to {results_dir}")
    logger.info(f"Summary report: {summary_file}")

if __name__ == "__main__":
>>>>>>> 5d96285edcdaab24a8fcf7efb1d7ec4f7b04bb22
    asyncio.run(main()) 