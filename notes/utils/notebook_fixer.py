import json
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
notebook_path = os.path.join(os.path.dirname(script_dir), 'm13_OOD_KNN_Detection.ipynb')
fixed_path = os.path.join(os.path.dirname(script_dir), 'm13_OOD_KNN_Detection.ipynb')

try:
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    if 'metadata' in notebook and 'widgets' in notebook['metadata']:
        print("Found and removing 'metadata.widgets'...")
        del notebook['metadata']['widgets']

        with open(fixed_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=2)
        print(f"Successfully fixed notebook! Saved as: {fixed_path}")

    else:
        print("No 'metadata.widgets' key found. File may already be ok.")

except Exception as e:
    print(f"An error occurred: {e}")