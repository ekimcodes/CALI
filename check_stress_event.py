import json
import os

# Load data.js content
# format is: const DEMO_DATA = [...]
path = "dashboard/data.js"

with open(path, "r") as f:
    content = f.read()
    
# Strip variable decl
json_str = content.replace("const DEMO_DATA = ", "").strip().rstrip(";")
data = json.loads(json_str)

# Find first non-zero pred
first_stress_time = None
for pt in data:
    if pt['pred'] > 0.1: # threshold above baseline noise
        first_stress_time = pt['t']
        print(f"Stress Event starts at t = {pt['t']} seconds. Value: {pt['pred']}")
        break

if not first_stress_time:
    print("No stress events found in data (pred always < 0.1).")
else:
    print(f"Total duration: {data[-1]['t']} seconds")
