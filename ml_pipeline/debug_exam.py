import os
import glob
import pandas as pd

base_path = r"c:/Users/edwin/Desktop/CALI/a-wearable-exam-stress-dataset-for-predicting-cognitive-performance-in-real-world-settings-1.0.0/Data/Data"
print(f"Base: {base_path}")
student_dirs = glob.glob(os.path.join(base_path, "S*"))
print(f"Dirs found: {student_dirs}")

if student_dirs:
    s_dir = student_dirs[0]
    print(f"Checking {s_dir}")
    files = os.listdir(s_dir)
    print(f"Files: {files}")
    
    ibi_path = os.path.join(s_dir, 'IBI.csv')
    print(f"IBI Exists: {os.path.exists(ibi_path)}")
    
    if os.path.exists(ibi_path):
        try:
            df = pd.read_csv(ibi_path, nrows=5)
            print("IBI Head:")
            print(df)
        except Exception as e:
            print(f"Read Error: {e}")
