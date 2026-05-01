#!/usr/bin/env python
"""Quick API test script for Truth backend."""

import requests
import json
import time

def test_api():
    print('Testing Truth API...')
    print('=' * 60)

    # Test 1: Health check
    try:
        resp = requests.get('http://127.0.0.1:8000/', timeout=5)
        print(f'✓ Health check: {resp.status_code}')
        print(f'  Response: {resp.json()}')
    except Exception as e:
        print(f'✗ Health check failed: {e}')
        return

    print()

    # Test 2: Simple text analysis
    try:
        print('Testing text analysis endpoint...')
        test_text = 'Breaking news: Scientists discover new species of frog in Amazon rainforest.'
        resp = requests.post(
            'http://127.0.0.1:8000/api/analyze',
            data={'text': test_text},
            timeout=15
        )
        if resp.status_code == 200:
            data = resp.json()
            print(f'✓ Analysis successful')
            print(f'  Prediction: {data["prediction"]}')
            print(f'  Confidence: {data["confidence"]}')
            print(f'  Processing Time: {data["processing_time"]}s')
            print(f'  Factual Checks: {len(data["factual_analysis"])} performed')
            print(f'  Analyzed text length: {len(data["analyzed_text"])} chars')
        else:
            print(f'✗ Request failed: {resp.status_code}')
            print(f'  Response: {resp.text}')
    except Exception as e:
        print(f'✗ Analysis failed: {e}')
        import traceback
        traceback.print_exc()

    print()
    print('=' * 60)
    print('Testing frontend build...')
    
    import os
    frontend_dist = './Truth/frontend/dist/index.html'
    if os.path.exists(frontend_dist):
        print(f'✓ Frontend built at {frontend_dist}')
        with open(frontend_dist, 'r') as f:
            content = f.read()
            print(f'  HTML size: {len(content)} chars')
    else:
        print(f'✗ Frontend dist not found')

if __name__ == '__main__':
    test_api()
