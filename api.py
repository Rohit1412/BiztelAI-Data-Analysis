<<<<<<< HEAD
from flask import Flask, request, jsonify
import json
from transcript_analyzer import analyze_chat_transcript
import os
import traceback

app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint to verify the API is running."""
    return jsonify({"status": "healthy"})

@app.route('/analyze', methods=['POST'])
def analyze_transcript():
    """
    Analyze a chat transcript and return insights.
    
    Expected input:
    - JSON data containing a chat transcript in the request body
    
    Returns:
    - JSON with analysis results including:
      - Possible article link/topic
      - Number of messages by agent
      - Overall sentiment for each agent
    """
    try:
        # Get JSON data from request
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
        
        transcript_data = request.get_json()
        
        # Validate input
        if not transcript_data:
            return jsonify({"error": "No transcript data provided"}), 400
        
        # Analyze transcript
        results = analyze_chat_transcript(transcript_data)
        
        # Check for errors
        if 'error' in results:
            return jsonify({"error": results['error']}), 400
        
        # Return results
        return jsonify(results), 200
    
    except Exception as e:
        # Log the full exception
        print(f"Error processing request: {str(e)}")
        print(traceback.format_exc())
        
        # Return error message
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

@app.route('/analyze_file', methods=['POST'])
def analyze_transcript_file():
    """
    Analyze a chat transcript from an uploaded JSON file.
    
    Expected input:
    - A file upload with name 'transcript' containing JSON data
    
    Returns:
    - JSON with analysis results including:
      - Possible article link/topic
      - Number of messages by agent
      - Overall sentiment for each agent
    """
    try:
        # Check if file was uploaded
        if 'transcript' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files['transcript']
        
        # Check if file has a name
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Check if file is JSON
        if not file.filename.endswith('.json'):
            return jsonify({"error": "Only JSON files are supported"}), 400
        
        # Save file temporarily
        temp_file_path = os.path.join('temp', file.filename)
        os.makedirs('temp', exist_ok=True)
        file.save(temp_file_path)
        
        # Analyze transcript
        results = analyze_chat_transcript(temp_file_path)
        
        # Clean up temporary file
        os.remove(temp_file_path)
        
        # Check for errors
        if 'error' in results:
            return jsonify({"error": results['error']}), 400
        
        # Return results
        return jsonify(results), 200
    
    except Exception as e:
        # Log the full exception
        print(f"Error processing file: {str(e)}")
        print(traceback.format_exc())
        
        # Return error message
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

@app.route('/analyze_by_id', methods=['GET'])
def analyze_by_conversation_id():
    """
    Analyze a chat transcript by conversation ID.
    
    Expected input:
    - conversation_id: ID of the conversation to analyze (query parameter)
    
    Returns:
    - JSON with analysis results including:
      - Possible article link/topic
      - Number of messages by agent
      - Overall sentiment for each agent
    """
    try:
        # Get conversation ID from query parameters
        conversation_id = request.args.get('conversation_id')
        
        if not conversation_id:
            return jsonify({"error": "No conversation ID provided"}), 400
        
        # Load full dataset
        try:
            with open("BiztelAI_DS_Dataset_Mar'25.json", 'r', encoding='utf-8') as f:
                all_data = json.load(f)
        except Exception as e:
            return jsonify({"error": f"Failed to load dataset: {str(e)}"}), 500
        
        # Check if conversation exists
        if conversation_id not in all_data:
            return jsonify({"error": f"Conversation ID '{conversation_id}' not found"}), 404
        
        # Extract conversation data
        conversation_data = {conversation_id: all_data[conversation_id]}
        
        # Analyze transcript
        results = analyze_chat_transcript(conversation_data)
        
        # Check for errors
        if 'error' in results:
            return jsonify({"error": results['error']}), 400
        
        # Return results
        return jsonify(results), 200
    
    except Exception as e:
        # Log the full exception
        print(f"Error processing request: {str(e)}")
        print(traceback.format_exc())
        
        # Return error message
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    print("Starting BiztelAI Transcript Analysis API...")
    print("Available endpoints:")
    print("  - /health: Health check endpoint (GET)")
    print("  - /analyze: Analyze a transcript provided in request body (POST)")
    print("  - /analyze_file: Analyze a transcript from uploaded JSON file (POST)")
    print("  - /analyze_by_id: Analyze a transcript by conversation ID (GET)")
    
=======
from flask import Flask, request, jsonify
import json
from transcript_analyzer import analyze_chat_transcript
import os
import traceback

app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint to verify the API is running."""
    return jsonify({"status": "healthy"})

@app.route('/analyze', methods=['POST'])
def analyze_transcript():
    """
    Analyze a chat transcript and return insights.
    
    Expected input:
    - JSON data containing a chat transcript in the request body
    
    Returns:
    - JSON with analysis results including:
      - Possible article link/topic
      - Number of messages by agent
      - Overall sentiment for each agent
    """
    try:
        # Get JSON data from request
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
        
        transcript_data = request.get_json()
        
        # Validate input
        if not transcript_data:
            return jsonify({"error": "No transcript data provided"}), 400
        
        # Analyze transcript
        results = analyze_chat_transcript(transcript_data)
        
        # Check for errors
        if 'error' in results:
            return jsonify({"error": results['error']}), 400
        
        # Return results
        return jsonify(results), 200
    
    except Exception as e:
        # Log the full exception
        print(f"Error processing request: {str(e)}")
        print(traceback.format_exc())
        
        # Return error message
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

@app.route('/analyze_file', methods=['POST'])
def analyze_transcript_file():
    """
    Analyze a chat transcript from an uploaded JSON file.
    
    Expected input:
    - A file upload with name 'transcript' containing JSON data
    
    Returns:
    - JSON with analysis results including:
      - Possible article link/topic
      - Number of messages by agent
      - Overall sentiment for each agent
    """
    try:
        # Check if file was uploaded
        if 'transcript' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files['transcript']
        
        # Check if file has a name
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Check if file is JSON
        if not file.filename.endswith('.json'):
            return jsonify({"error": "Only JSON files are supported"}), 400
        
        # Save file temporarily
        temp_file_path = os.path.join('temp', file.filename)
        os.makedirs('temp', exist_ok=True)
        file.save(temp_file_path)
        
        # Analyze transcript
        results = analyze_chat_transcript(temp_file_path)
        
        # Clean up temporary file
        os.remove(temp_file_path)
        
        # Check for errors
        if 'error' in results:
            return jsonify({"error": results['error']}), 400
        
        # Return results
        return jsonify(results), 200
    
    except Exception as e:
        # Log the full exception
        print(f"Error processing file: {str(e)}")
        print(traceback.format_exc())
        
        # Return error message
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

@app.route('/analyze_by_id', methods=['GET'])
def analyze_by_conversation_id():
    """
    Analyze a chat transcript by conversation ID.
    
    Expected input:
    - conversation_id: ID of the conversation to analyze (query parameter)
    
    Returns:
    - JSON with analysis results including:
      - Possible article link/topic
      - Number of messages by agent
      - Overall sentiment for each agent
    """
    try:
        # Get conversation ID from query parameters
        conversation_id = request.args.get('conversation_id')
        
        if not conversation_id:
            return jsonify({"error": "No conversation ID provided"}), 400
        
        # Load full dataset
        try:
            with open("BiztelAI_DS_Dataset_Mar'25.json", 'r', encoding='utf-8') as f:
                all_data = json.load(f)
        except Exception as e:
            return jsonify({"error": f"Failed to load dataset: {str(e)}"}), 500
        
        # Check if conversation exists
        if conversation_id not in all_data:
            return jsonify({"error": f"Conversation ID '{conversation_id}' not found"}), 404
        
        # Extract conversation data
        conversation_data = {conversation_id: all_data[conversation_id]}
        
        # Analyze transcript
        results = analyze_chat_transcript(conversation_data)
        
        # Check for errors
        if 'error' in results:
            return jsonify({"error": results['error']}), 400
        
        # Return results
        return jsonify(results), 200
    
    except Exception as e:
        # Log the full exception
        print(f"Error processing request: {str(e)}")
        print(traceback.format_exc())
        
        # Return error message
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    print("Starting BiztelAI Transcript Analysis API...")
    print("Available endpoints:")
    print("  - /health: Health check endpoint (GET)")
    print("  - /analyze: Analyze a transcript provided in request body (POST)")
    print("  - /analyze_file: Analyze a transcript from uploaded JSON file (POST)")
    print("  - /analyze_by_id: Analyze a transcript by conversation ID (GET)")
    
>>>>>>> 5d96285edcdaab24a8fcf7efb1d7ec4f7b04bb22
    app.run(debug=True, host='0.0.0.0', port=5000) 