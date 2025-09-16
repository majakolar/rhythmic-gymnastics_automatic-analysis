import pickle
import os

folder_name = "data/extracted/keypoints/2025-07-22_17-00-45"

start_names = ("train", "test", "val")
problematic_name = "problematic_videos"

all_kps_data = []
all_problematic_data = []
for file in os.listdir(folder_name):
    if file.startswith(start_names) and file.endswith(".pkl"):
        with open(os.path.join(folder_name, file), "rb") as f:
            kps_data = pickle.load(f)
        all_kps_data.append(kps_data)

    elif file.startswith(problematic_name) and file.endswith(".pkl"):
        with open(os.path.join(folder_name, file), "rb") as f:
            problematic_data = pickle.load(f)
        all_problematic_data.extend(problematic_data)


merged_kps_data = {}

for item_dict in all_kps_data:
    for category, video_data_dict in item_dict.items():
        if category not in merged_kps_data:
            merged_kps_data[category] = {}
        
        merged_kps_data[category].update(video_data_dict)

print(f"Keys in the merged dictionary: {list(merged_kps_data.keys())}")
print(f"Number of 'Balance' videos: {len(merged_kps_data.get('Balance', {}))}")


with open(os.path.join(folder_name, "all_kps_data.pkl"), "wb") as f:
    pickle.dump(merged_kps_data, f)

with open(os.path.join(folder_name, "all_problematic_data.pkl"), "wb") as f:
    pickle.dump(all_problematic_data, f)