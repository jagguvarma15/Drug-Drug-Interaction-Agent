#!/usr/bin/env python3
"""
Test script for Drug-Drug Interaction Agent with RAG functionality
"""

from drug_drug_interaction_agent import drug_interaction_analysis

def test_rag_interactions():
    """Test the RAG functionality with known interactions"""
    
    print("=== Testing Drug-Drug Interaction Agent with RAG ===\n")
    
    # Test cases with interactions that should be in the RAG database
    test_cases = [
        "warfarin and aspirin",
        "What happens when I take simvastatin with grapefruit?",
        "Can I take metformin with alcohol?",
        "Is it safe to combine lisinopril and ibuprofen?",
        "Tell me about digoxin and furosemide interaction",
        "unknown drug A and unknown drug B"  # This should fall back to LLM
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"Test {i}: {test_case}")
        print("-" * 60)
        
        try:
            result = drug_interaction_analysis(test_case)
            print(f"Result:\n{result}")
            
        except Exception as e:
            print(f"Error: {e}")
        
        print("\n" + "=" * 80 + "\n")

if __name__ == "__main__":
    test_rag_interactions() 