import torch

# 1. Change this to the exact name of your saved file
FILE_NAME = "caltech_train_features.pt" 

print(f"⏳ Loading the massive {FILE_NAME} into memory... (This might take a minute!)")

# Load the file locally
try:
    data = torch.load(FILE_NAME, map_location="cpu")
except Exception as e:
    print(f"Error loading file: {e}")
    exit()

# 2. Check if it's a list or a PyTorch tensor (which it should be based on our earlier code)
if isinstance(data, list) or torch.is_tensor(data):
    total_items = len(data)
    midpoint = total_items // 2
    
    # Slice it in half
    part1 = data[:midpoint]
    part2 = data[midpoint:]
    
    print(f"🔪 Successfully sliced {total_items} items into two halves: {len(part1)} and {len(part2)}.")
    
    # 3. Save the two halves
    print("💾 Saving Part 1... (features_part1.pt)")
    torch.save(part1, "features_part1.pt")
    
    print("💾 Saving Part 2... (features_part2.pt)")
    torch.save(part2, "features_part2.pt")
    
    print("✅ Done! You have successfully halved the file.")

elif isinstance(data, dict):
    print("Oh! This is a dictionary. I am splitting it by keys.")
    # If you saved train and test in the same dictionary, this separates them
    for key, value in data.items():
        save_name = f"features_{key}.pt"
        print(f"💾 Saving {key} data to {save_name}...")
        torch.save(value, save_name)
    print("✅ Done! Dictionary separated into individual files.")
else:
    print("I don't recognize this data format. Are you sure it's a list or tensor?")