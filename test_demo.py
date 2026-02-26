#!/usr/bin/env python3
"""
Test the Streamlit demo functionality
"""
import requests
import time
import sys

def test_demo():
    """Test if the Streamlit demo is accessible"""
    print("🧪 Testing Streamlit Demo")
    print("=" * 30)
    
    # Wait a moment for the app to start
    time.sleep(2)
    
    try:
        # Test if the demo is accessible
        response = requests.get("http://localhost:8501", timeout=10)
        
        if response.status_code == 200:
            print("✅ Streamlit demo is accessible")
            print(f"   Status: {response.status_code}")
            print(f"   Content length: {len(response.content)} bytes")
            
            # Check if it's actually the Streamlit app
            if "streamlit" in response.text.lower():
                print("✅ Streamlit app is running")
            else:
                print("⚠️ Response doesn't look like Streamlit")
                
        else:
            print(f"❌ Demo not accessible: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to demo - is it running?")
        return False
    except Exception as e:
        print(f"❌ Error testing demo: {e}")
        return False
    
    print("\n🌐 Demo should be available at: http://localhost:8501")
    print("✅ You can now:")
    print("   - Upload SAR images")
    print("   - Run denoising with different methods")
    print("   - Compare results")
    print("   - Adjust parameters")
    
    return True

if __name__ == "__main__":
    test_demo()


