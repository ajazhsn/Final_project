# tester.py — Load balancing proof
import requests
from collections import Counter
import concurrent.futures

URL = "http://127.0.0.1:8000/health"

def call_api(_):
    try:
        r = requests.get(URL, timeout=5)
        return r.json().get("container_id", "Failed")
    except Exception:
        return "Failed"

print("Sending 20 requests to prove load distribution...")
with concurrent.futures.ThreadPoolExecutor(
        max_workers=5) as executor:
    results = list(executor.map(call_api, range(20)))

print("\nContainer hit distribution:")
for container, count in Counter(results).items():
    print(f"  {container}: {count} requests")
print(f"\nTotal: {len(results)} requests")
