# Performance Testing for BiztelAI API

This document describes how to use the performance testing script to evaluate the performance of the BiztelAI API under various load conditions.

## Overview

The `performance_test.py` script allows you to:

- Send multiple concurrent requests to any API endpoint
- Measure response times and success rates
- Test authenticated endpoints using JWT authentication
- Visualize performance metrics
- Save test results for future analysis

## Requirements

Make sure you have installed all required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

The script can be executed with various command-line options to configure the test:

```bash
python performance_test.py [options]
```

### Basic Usage

Test the health endpoint with default settings (100 requests, 10 concurrent):

```bash
python performance_test.py
```

Test a specific endpoint:

```bash
python performance_test.py --endpoint /api/dataset/summary
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--url` | Base URL of the API | http://localhost:5000 |
| `--endpoint` | API endpoint to test | /api/health |
| `--requests` | Number of requests to send | 100 |
| `--concurrency` | Number of concurrent requests | 10 |
| `--auth` | Enable authentication | False |
| `--username` | Username for authentication | user |
| `--password` | Password for authentication | user123 |
| `--save` | Save results to JSON file | None |
| `--visualize` | Generate visualization of results | False |
| `--output-image` | Save visualization to image file | None |

### Examples

#### Testing with authentication

```bash
python performance_test.py --endpoint /api/analysis/transcript --auth --username admin --password admin123
```

#### Testing with higher concurrency

```bash
python performance_test.py --endpoint /api/dataset/summary --requests 1000 --concurrency 50
```

#### Saving and visualizing results

```bash
python performance_test.py --endpoint /api/dataset/transform --save results.json --visualize --output-image performance.png
```

## Interpreting Results

After running a test, the script will display statistics in a table format:

- **Endpoint**: The API endpoint tested
- **Requests**: Total number of requests sent
- **Concurrency**: Number of concurrent requests
- **Success**: Percentage of successful requests (HTTP 200-299)
- **Avg (s)**: Average response time in seconds
- **Median (s)**: Median response time in seconds
- **P95 (s)**: 95th percentile response time in seconds
- **Min (s)**: Minimum response time in seconds
- **Max (s)**: Maximum response time in seconds

If visualization is enabled, the script will generate:
1. A bar chart showing average and 95th percentile response times
2. A bar chart showing success rates for each endpoint

## Best Practices

- Start with a small number of requests and gradually increase to find performance limits
- Test different endpoints to identify potential bottlenecks
- Compare performance with and without authentication
- Monitor server resources during testing to identify potential issues
- Run tests at different times of day to account for system variability

## Troubleshooting

If you encounter issues:

- Ensure the API server is running and accessible
- Check that authentication credentials are correct
- Verify the endpoint path is correct
- Look for error messages in the logs
- Check server logs for potential backend issues

## Advanced Usage

For more complex testing scenarios, you can modify the script directly:

- Add support for POST, PUT, DELETE requests
- Include custom headers or payload data
- Test specific error conditions
- Implement more complex authentication flows
- Add additional performance metrics 