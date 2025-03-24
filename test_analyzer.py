import json
from transcript_analyzer import TranscriptAnalyzer
import os

def test_analyzer():
    """Test the transcript analyzer with a simple example."""
    
    # Create a test transcript
    test_transcript = {
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
    
    # Create output directory
    os.makedirs('test_output', exist_ok=True)
    
    # Save test transcript to file
    with open('test_output/test_transcript.json', 'w') as f:
        json.dump(test_transcript, f)
    
    # Initialize analyzer
    analyzer = TranscriptAnalyzer()
    
    print("Testing transcript analysis from variable...")
    results_from_var = analyzer.analyze_transcript(test_transcript)
    
    print("\nTesting transcript analysis from file...")
    results_from_file = analyzer.analyze_transcript('test_output/test_transcript.json')
    
    # Print results
    print("\n=== Analysis Results ===")
    print(f"Article: {results_from_var['article_link']}")
    print("\nMessage Counts:")
    for agent, count in results_from_var['message_counts'].items():
        print(f"  {agent}: {count} messages")
    
    print("\nAgent Sentiment:")
    for agent, sentiment in results_from_var['agent_sentiment'].items():
        print(f"  {agent}: {sentiment['dominant_sentiment']} (score: {sentiment['avg_score']:.2f})")
        if 'sentiment_distribution' in sentiment:
            print("    Distribution:")
            for label, pct in sentiment['sentiment_distribution'].items():
                print(f"      {label}: {pct:.1f}%")
    
    # Visualize results
    analyzer.visualize_results(results_from_var, 'test_output/test_results.png')
    
    print("\nTest completed. Results visualization saved to 'test_output/test_results.png'")
    
    return results_from_var

if __name__ == "__main__":
    test_analyzer() 