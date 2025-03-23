<<<<<<< HEAD
import requests
import json
import os
import time
import sys
from colorama import init, Fore, Style

# Initialize colorama for colored output
init()

# Base URL for API
BASE_URL = "http://localhost:5000/api"

def print_success(message):
    """Print success message in green."""
    print(f"{Fore.GREEN}✓ {message}{Style.RESET_ALL}")

def print_error(message):
    """Print error message in red."""
    print(f"{Fore.RED}✗ {message}{Style.RESET_ALL}")

def print_info(message):
    """Print info message in blue."""
    print(f"{Fore.BLUE}ℹ {message}{Style.RESET_ALL}")

def print_response(response, detailed=False):
    """Print response status and content."""
    status_code = response.status_code
    if 200 <= status_code < 300:
        status_color = Fore.GREEN
    elif 400 <= status_code < 500:
        status_color = Fore.YELLOW
    else:
        status_color = Fore.RED
    
    print(f"Status: {status_color}{status_code}{Style.RESET_ALL}")
    
    if detailed:
        try:
            json_response = response.json()
            print("Response:")
            print(json.dumps(json_response, indent=2))
        except Exception as e:
            print(f"Failed to parse response as JSON: {e}")
            print(response.text)
    else:
        print(f"Response size: {len(response.content)} bytes")

def test_health_endpoint():
    """Test the health check endpoint."""
    print_info("Testing health check endpoint...")
    
    try:
        response = requests.get(f"{BASE_URL}/health")
        
        if response.status_code == 200:
            print_success("Health check endpoint is working")
        else:
            print_error(f"Health check failed with status code {response.status_code}")
        
        print_response(response, detailed=True)
        
        return response.status_code == 200
    
    except Exception as e:
        print_error(f"Error connecting to health endpoint: {e}")
        return False

def test_documentation_endpoint():
    """Test the API documentation endpoint."""
    print_info("Testing API documentation endpoint...")
    
    try:
        response = requests.get(BASE_URL)
        
        if response.status_code == 200:
            print_success("Documentation endpoint is working")
        else:
            print_error(f"Documentation endpoint failed with status code {response.status_code}")
        
        print_response(response, detailed=True)
        
        return response.status_code == 200
    
    except Exception as e:
        print_error(f"Error connecting to documentation endpoint: {e}")
        return False

def test_dataset_summary_endpoint():
    """Test the dataset summary endpoint."""
    print_info("Testing dataset summary endpoint...")
    
    try:
        # Test basic summary
        print_info("Getting basic summary...")
        response_basic = requests.get(f"{BASE_URL}/dataset/summary")
        
        if response_basic.status_code == 200:
            print_success("Basic summary endpoint is working")
        else:
            print_error(f"Basic summary failed with status code {response_basic.status_code}")
        
        print_response(response_basic)
        
        # Test detailed summary
        print_info("Getting detailed summary...")
        response_detailed = requests.get(f"{BASE_URL}/dataset/summary?level=detailed")
        
        if response_detailed.status_code == 200:
            print_success("Detailed summary endpoint is working")
        else:
            print_error(f"Detailed summary failed with status code {response_detailed.status_code}")
        
        print_response(response_detailed)
        
        return response_basic.status_code == 200 and response_detailed.status_code == 200
    
    except Exception as e:
        print_error(f"Error connecting to dataset summary endpoint: {e}")
        return False

def test_transform_endpoint():
    """Test the data transformation endpoint."""
    print_info("Testing data transformation endpoint...")
    
    # Create a sample conversation for testing
    sample_data = {
        "test_conversation": {
            "article_url": "https://www.washingtonpost.com/sports/test-article",
            "config": "A",
            "content": [
                {
                    "message": "Did you watch the game last night? It was really exciting!",
                    "agent": "agent_1",
                    "sentiment": "Happy",
                    "knowledge_source": ["FS1"],
                    "turn_rating": "Good"
                },
                {
                    "message": "Yes, I did! The final score was so close. I couldn't believe that last-minute play.",
                    "agent": "agent_2",
                    "sentiment": "Surprised",
                    "knowledge_source": ["FS2"],
                    "turn_rating": "Excellent"
                }
            ]
        }
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/transform",
            json=sample_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            print_success("Transform endpoint is working")
            
            # Check if all expected transformations were applied
            json_response = response.json()
            transformations = json_response.get("transformations_applied", [])
            expected_transformations = [
                "missing_value_handling",
                "duplicate_removal",
                "text_cleaning",
                "tokenization",
                "stopword_removal",
                "lemmatization",
                "categorical_encoding",
                "feature_creation"
            ]
            
            missing_transformations = [t for t in expected_transformations if t not in transformations]
            
            if missing_transformations:
                print_error(f"Missing transformations: {', '.join(missing_transformations)}")
            else:
                print_success("All expected transformations were applied")
        else:
            print_error(f"Transform endpoint failed with status code {response.status_code}")
        
        print_response(response)
        
        return response.status_code == 200
    
    except Exception as e:
        print_error(f"Error connecting to transform endpoint: {e}")
        return False

def test_analyze_transcript_endpoint():
    """Test the transcript analysis endpoint."""
    print_info("Testing transcript analysis endpoint...")
    
    # Create a sample conversation for testing
    sample_data = {
        "test_conversation": {
            "article_url": "https://www.washingtonpost.com/sports/test-article",
            "config": "A",
            "content": [
                {
                    "message": "Did you watch the game last night? It was really exciting!",
                    "agent": "agent_1",
                    "sentiment": "Happy",
                    "knowledge_source": ["FS1"],
                    "turn_rating": "Good"
                },
                {
                    "message": "Yes, I did! The final score was so close. I couldn't believe that last-minute play.",
                    "agent": "agent_2",
                    "sentiment": "Surprised",
                    "knowledge_source": ["FS2"],
                    "turn_rating": "Excellent"
                },
                {
                    "message": "What did you think about the referee's decision in the third quarter?",
                    "agent": "agent_1",
                    "sentiment": "Curious to dive deeper",
                    "knowledge_source": ["FS1"],
                    "turn_rating": "Good"
                },
                {
                    "message": "I thought it was a bad call. The replay clearly showed it wasn't a foul.",
                    "agent": "agent_2",
                    "sentiment": "Angry",
                    "knowledge_source": ["FS2"],
                    "turn_rating": "Good"
                }
            ]
        }
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/analyze/transcript",
            json=sample_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            print_success("Transcript analysis endpoint is working")
            
            # Check for expected response structure
            json_response = response.json()
            analysis_results = json_response.get("analysis_results", {})
            
            if "article_link" in analysis_results and "message_counts" in analysis_results and "agent_sentiment" in analysis_results:
                print_success("Analysis results contain all expected fields")
                print_info(f"Article link: {analysis_results['article_link']}")
                print_info(f"Message counts: {analysis_results['message_counts']}")
                print_info("Agent sentiment:")
                for agent, sentiment in analysis_results.get("agent_sentiment", {}).items():
                    print_info(f"  {agent}: {sentiment.get('dominant_sentiment', 'Unknown')}")
            else:
                print_error("Analysis results are missing expected fields")
        else:
            print_error(f"Transcript analysis endpoint failed with status code {response.status_code}")
        
        print_response(response)
        
        return response.status_code == 200
    
    except Exception as e:
        print_error(f"Error connecting to transcript analysis endpoint: {e}")
        return False

def test_metrics_endpoint():
    """Test the API metrics endpoint."""
    print_info("Testing API metrics endpoint...")
    
    try:
        response = requests.get(f"{BASE_URL}/metrics")
        
        if response.status_code == 200:
            print_success("Metrics endpoint is working")
        else:
            print_error(f"Metrics endpoint failed with status code {response.status_code}")
        
        print_response(response, detailed=True)
        
        return response.status_code == 200
    
    except Exception as e:
        print_error(f"Error connecting to metrics endpoint: {e}")
        return False

def run_all_tests():
    """Run all API endpoint tests."""
    print_info("Starting API tests...")
    
    # First check if the API is running
    try:
        requests.get(f"{BASE_URL}/health", timeout=5)
    except requests.exceptions.ConnectionError:
        print_error(f"Cannot connect to the API at {BASE_URL}")
        print_info("Make sure the API server is running with: python api_server.py")
        return False
    
    # Run all tests
    tests = [
        ("Health check", test_health_endpoint),
        ("API documentation", test_documentation_endpoint),
        ("Dataset summary", test_dataset_summary_endpoint),
        ("Data transformation", test_transform_endpoint),
        ("Transcript analysis", test_analyze_transcript_endpoint),
        ("API metrics", test_metrics_endpoint)
    ]
    
    results = {}
    
    print("\n" + "="*50)
    print("RUNNING API TESTS")
    print("="*50 + "\n")
    
    for name, test_func in tests:
        print("\n" + "-"*50)
        print(f"TEST: {name}")
        print("-"*50)
        
        start_time = time.time()
        success = test_func()
        end_time = time.time()
        
        results[name] = {
            "success": success,
            "duration": end_time - start_time
        }
        
        print(f"Duration: {results[name]['duration']:.2f} seconds")
    
    # Print summary
    print("\n" + "="*50)
    print("TEST RESULTS SUMMARY")
    print("="*50)
    
    all_passed = True
    for name, result in results.items():
        if result["success"]:
            status = f"{Fore.GREEN}PASSED{Style.RESET_ALL}"
        else:
            status = f"{Fore.RED}FAILED{Style.RESET_ALL}"
            all_passed = False
        
        print(f"{name}: {status} ({result['duration']:.2f}s)")
    
    if all_passed:
        print(f"\n{Fore.GREEN}All tests passed!{Style.RESET_ALL}")
    else:
        print(f"\n{Fore.RED}Some tests failed. Please check the output above for details.{Style.RESET_ALL}")
    
    return all_passed

if __name__ == "__main__":
    # Run all tests by default
=======
import requests
import json
import os
import time
import sys
from colorama import init, Fore, Style

# Initialize colorama for colored output
init()

# Base URL for API
BASE_URL = "http://localhost:5000/api"

def print_success(message):
    """Print success message in green."""
    print(f"{Fore.GREEN}✓ {message}{Style.RESET_ALL}")

def print_error(message):
    """Print error message in red."""
    print(f"{Fore.RED}✗ {message}{Style.RESET_ALL}")

def print_info(message):
    """Print info message in blue."""
    print(f"{Fore.BLUE}ℹ {message}{Style.RESET_ALL}")

def print_response(response, detailed=False):
    """Print response status and content."""
    status_code = response.status_code
    if 200 <= status_code < 300:
        status_color = Fore.GREEN
    elif 400 <= status_code < 500:
        status_color = Fore.YELLOW
    else:
        status_color = Fore.RED
    
    print(f"Status: {status_color}{status_code}{Style.RESET_ALL}")
    
    if detailed:
        try:
            json_response = response.json()
            print("Response:")
            print(json.dumps(json_response, indent=2))
        except Exception as e:
            print(f"Failed to parse response as JSON: {e}")
            print(response.text)
    else:
        print(f"Response size: {len(response.content)} bytes")

def test_health_endpoint():
    """Test the health check endpoint."""
    print_info("Testing health check endpoint...")
    
    try:
        response = requests.get(f"{BASE_URL}/health")
        
        if response.status_code == 200:
            print_success("Health check endpoint is working")
        else:
            print_error(f"Health check failed with status code {response.status_code}")
        
        print_response(response, detailed=True)
        
        return response.status_code == 200
    
    except Exception as e:
        print_error(f"Error connecting to health endpoint: {e}")
        return False

def test_documentation_endpoint():
    """Test the API documentation endpoint."""
    print_info("Testing API documentation endpoint...")
    
    try:
        response = requests.get(BASE_URL)
        
        if response.status_code == 200:
            print_success("Documentation endpoint is working")
        else:
            print_error(f"Documentation endpoint failed with status code {response.status_code}")
        
        print_response(response, detailed=True)
        
        return response.status_code == 200
    
    except Exception as e:
        print_error(f"Error connecting to documentation endpoint: {e}")
        return False

def test_dataset_summary_endpoint():
    """Test the dataset summary endpoint."""
    print_info("Testing dataset summary endpoint...")
    
    try:
        # Test basic summary
        print_info("Getting basic summary...")
        response_basic = requests.get(f"{BASE_URL}/dataset/summary")
        
        if response_basic.status_code == 200:
            print_success("Basic summary endpoint is working")
        else:
            print_error(f"Basic summary failed with status code {response_basic.status_code}")
        
        print_response(response_basic)
        
        # Test detailed summary
        print_info("Getting detailed summary...")
        response_detailed = requests.get(f"{BASE_URL}/dataset/summary?level=detailed")
        
        if response_detailed.status_code == 200:
            print_success("Detailed summary endpoint is working")
        else:
            print_error(f"Detailed summary failed with status code {response_detailed.status_code}")
        
        print_response(response_detailed)
        
        return response_basic.status_code == 200 and response_detailed.status_code == 200
    
    except Exception as e:
        print_error(f"Error connecting to dataset summary endpoint: {e}")
        return False

def test_transform_endpoint():
    """Test the data transformation endpoint."""
    print_info("Testing data transformation endpoint...")
    
    # Create a sample conversation for testing
    sample_data = {
        "test_conversation": {
            "article_url": "https://www.washingtonpost.com/sports/test-article",
            "config": "A",
            "content": [
                {
                    "message": "Did you watch the game last night? It was really exciting!",
                    "agent": "agent_1",
                    "sentiment": "Happy",
                    "knowledge_source": ["FS1"],
                    "turn_rating": "Good"
                },
                {
                    "message": "Yes, I did! The final score was so close. I couldn't believe that last-minute play.",
                    "agent": "agent_2",
                    "sentiment": "Surprised",
                    "knowledge_source": ["FS2"],
                    "turn_rating": "Excellent"
                }
            ]
        }
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/transform",
            json=sample_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            print_success("Transform endpoint is working")
            
            # Check if all expected transformations were applied
            json_response = response.json()
            transformations = json_response.get("transformations_applied", [])
            expected_transformations = [
                "missing_value_handling",
                "duplicate_removal",
                "text_cleaning",
                "tokenization",
                "stopword_removal",
                "lemmatization",
                "categorical_encoding",
                "feature_creation"
            ]
            
            missing_transformations = [t for t in expected_transformations if t not in transformations]
            
            if missing_transformations:
                print_error(f"Missing transformations: {', '.join(missing_transformations)}")
            else:
                print_success("All expected transformations were applied")
        else:
            print_error(f"Transform endpoint failed with status code {response.status_code}")
        
        print_response(response)
        
        return response.status_code == 200
    
    except Exception as e:
        print_error(f"Error connecting to transform endpoint: {e}")
        return False

def test_analyze_transcript_endpoint():
    """Test the transcript analysis endpoint."""
    print_info("Testing transcript analysis endpoint...")
    
    # Create a sample conversation for testing
    sample_data = {
        "test_conversation": {
            "article_url": "https://www.washingtonpost.com/sports/test-article",
            "config": "A",
            "content": [
                {
                    "message": "Did you watch the game last night? It was really exciting!",
                    "agent": "agent_1",
                    "sentiment": "Happy",
                    "knowledge_source": ["FS1"],
                    "turn_rating": "Good"
                },
                {
                    "message": "Yes, I did! The final score was so close. I couldn't believe that last-minute play.",
                    "agent": "agent_2",
                    "sentiment": "Surprised",
                    "knowledge_source": ["FS2"],
                    "turn_rating": "Excellent"
                },
                {
                    "message": "What did you think about the referee's decision in the third quarter?",
                    "agent": "agent_1",
                    "sentiment": "Curious to dive deeper",
                    "knowledge_source": ["FS1"],
                    "turn_rating": "Good"
                },
                {
                    "message": "I thought it was a bad call. The replay clearly showed it wasn't a foul.",
                    "agent": "agent_2",
                    "sentiment": "Angry",
                    "knowledge_source": ["FS2"],
                    "turn_rating": "Good"
                }
            ]
        }
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/analyze/transcript",
            json=sample_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            print_success("Transcript analysis endpoint is working")
            
            # Check for expected response structure
            json_response = response.json()
            analysis_results = json_response.get("analysis_results", {})
            
            if "article_link" in analysis_results and "message_counts" in analysis_results and "agent_sentiment" in analysis_results:
                print_success("Analysis results contain all expected fields")
                print_info(f"Article link: {analysis_results['article_link']}")
                print_info(f"Message counts: {analysis_results['message_counts']}")
                print_info("Agent sentiment:")
                for agent, sentiment in analysis_results.get("agent_sentiment", {}).items():
                    print_info(f"  {agent}: {sentiment.get('dominant_sentiment', 'Unknown')}")
            else:
                print_error("Analysis results are missing expected fields")
        else:
            print_error(f"Transcript analysis endpoint failed with status code {response.status_code}")
        
        print_response(response)
        
        return response.status_code == 200
    
    except Exception as e:
        print_error(f"Error connecting to transcript analysis endpoint: {e}")
        return False

def test_metrics_endpoint():
    """Test the API metrics endpoint."""
    print_info("Testing API metrics endpoint...")
    
    try:
        response = requests.get(f"{BASE_URL}/metrics")
        
        if response.status_code == 200:
            print_success("Metrics endpoint is working")
        else:
            print_error(f"Metrics endpoint failed with status code {response.status_code}")
        
        print_response(response, detailed=True)
        
        return response.status_code == 200
    
    except Exception as e:
        print_error(f"Error connecting to metrics endpoint: {e}")
        return False

def run_all_tests():
    """Run all API endpoint tests."""
    print_info("Starting API tests...")
    
    # First check if the API is running
    try:
        requests.get(f"{BASE_URL}/health", timeout=5)
    except requests.exceptions.ConnectionError:
        print_error(f"Cannot connect to the API at {BASE_URL}")
        print_info("Make sure the API server is running with: python api_server.py")
        return False
    
    # Run all tests
    tests = [
        ("Health check", test_health_endpoint),
        ("API documentation", test_documentation_endpoint),
        ("Dataset summary", test_dataset_summary_endpoint),
        ("Data transformation", test_transform_endpoint),
        ("Transcript analysis", test_analyze_transcript_endpoint),
        ("API metrics", test_metrics_endpoint)
    ]
    
    results = {}
    
    print("\n" + "="*50)
    print("RUNNING API TESTS")
    print("="*50 + "\n")
    
    for name, test_func in tests:
        print("\n" + "-"*50)
        print(f"TEST: {name}")
        print("-"*50)
        
        start_time = time.time()
        success = test_func()
        end_time = time.time()
        
        results[name] = {
            "success": success,
            "duration": end_time - start_time
        }
        
        print(f"Duration: {results[name]['duration']:.2f} seconds")
    
    # Print summary
    print("\n" + "="*50)
    print("TEST RESULTS SUMMARY")
    print("="*50)
    
    all_passed = True
    for name, result in results.items():
        if result["success"]:
            status = f"{Fore.GREEN}PASSED{Style.RESET_ALL}"
        else:
            status = f"{Fore.RED}FAILED{Style.RESET_ALL}"
            all_passed = False
        
        print(f"{name}: {status} ({result['duration']:.2f}s)")
    
    if all_passed:
        print(f"\n{Fore.GREEN}All tests passed!{Style.RESET_ALL}")
    else:
        print(f"\n{Fore.RED}Some tests failed. Please check the output above for details.{Style.RESET_ALL}")
    
    return all_passed

if __name__ == "__main__":
    # Run all tests by default
>>>>>>> 5d96285edcdaab24a8fcf7efb1d7ec4f7b04bb22
    run_all_tests() 