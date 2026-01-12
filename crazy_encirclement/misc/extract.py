import os
import pandas as pd
from pathlib import Path
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
from tqdm import tqdm


# Define the root directory containing the bag files
rootdir = os.path.join('/home/paulo/Documents/k_03/')

# Flag to force extraction even if extracted files already exist
force_extraction = True

# Find all .db3 files in the rootdir
bag_files = []
for subdir, _, files in os.walk(rootdir):
    for file in files:
        if file.endswith('.db3'):
            bag_files.append(os.path.join(subdir, file))

# Topics to extract
topics_to_extract = [
    # Common topics
    ['/encircle', 'std_msgs/msg/Bool'],
    ['/landing', 'std_msgs/msg/Bool'],
    ['/C04/vicon_position', 'geometry_msgs/msg/PoseStamped'],
    ['/C05/vicon_position', 'geometry_msgs/msg/PoseStamped'],
    ['/C14/vicon_position', 'geometry_msgs/msg/PoseStamped'],
    # ['/C04/gps_position', 'geometry_msgs/msg/PoseStamped'],
    # ['/C05/gps_position', 'geometry_msgs/msg/PoseStamped'],
    # ['/C14/gps_position', 'geometry_msgs/msg/PoseStamped'],
    # ['/C04/gps_scanner_position', 'geometry_msgs/msg/PoseStamped'],
    # ['/C05/gps_scanner_position', 'geometry_msgs/msg/PoseStamped'],
    # ['/C14/gps_scanner_position', 'geometry_msgs/msg/PoseStamped'],
    # Baseline topics
    ['/C04/baseline/radius', 'std_msgs/msg/Float32'],
    ['/C05/baseline/radius', 'std_msgs/msg/Float32'],
    ['/C14/baseline/radius', 'std_msgs/msg/Float32'],
    ['/C04/baseline/omega', 'std_msgs/msg/Float32'],
    ['/C05/baseline/omega', 'std_msgs/msg/Float32'],
    ['/C14/baseline/omega', 'std_msgs/msg/Float32'],
    ['/C04/baseline/phase', 'std_msgs/msg/Float32'],
    ['/C05/baseline/phase', 'std_msgs/msg/Float32'],
    ['/C14/baseline/phase', 'std_msgs/msg/Float32'],
    ['/C04/baseline/pose', 'geometry_msgs/msg/PoseStamped'],
    ['/C05/baseline/pose', 'geometry_msgs/msg/PoseStamped'],
    ['/C14/baseline/pose', 'geometry_msgs/msg/PoseStamped'],
    ['/C04/baseline/phase_diff/leader', 'std_msgs/msg/Float32'],
    ['/C04/baseline/phase_diff/follower', 'std_msgs/msg/Float32'],
    ['/C05/baseline/phase_diff/leader', 'std_msgs/msg/Float32'],
    ['/C05/baseline/phase_diff/follower', 'std_msgs/msg/Float32'],
    ['/C14/baseline/phase_diff/leader', 'std_msgs/msg/Float32'],
    ['/C14/baseline/phase_diff/follower', 'std_msgs/msg/Float32'],
    # Filter topics
    ['/C04/filtered/radius', 'std_msgs/msg/Float32'],
    ['/C05/filtered/radius', 'std_msgs/msg/Float32'],
    ['/C14/filtered/radius', 'std_msgs/msg/Float32'],
    ['/C04/filtered/omega', 'std_msgs/msg/Float32'],
    ['/C05/filtered/omega', 'std_msgs/msg/Float32'],
    ['/C14/filtered/omega', 'std_msgs/msg/Float32'],
    ['/C04/filtered/phase', 'std_msgs/msg/Float32'],
    ['/C05/filtered/phase', 'std_msgs/msg/Float32'],
    ['/C14/filtered/phase', 'std_msgs/msg/Float32'],
    ['/C04/filtered/pose', 'geometry_msgs/msg/PoseStamped'],
    ['/C05/filtered/pose', 'geometry_msgs/msg/PoseStamped'],
    ['/C14/filtered/pose', 'geometry_msgs/msg/PoseStamped'],
    ['/C04/filtered/phase_diff/leader', 'std_msgs/msg/Float32'],
    ['/C04/filtered/phase_diff/follower', 'std_msgs/msg/Float32'],
    ['/C05/filtered/phase_diff/leader', 'std_msgs/msg/Float32'],
    ['/C05/filtered/phase_diff/follower', 'std_msgs/msg/Float32'],
    ['/C14/filtered/phase_diff/leader', 'std_msgs/msg/Float32'],
    ['/C14/filtered/phase_diff/follower', 'std_msgs/msg/Float32'],
]

typestore = get_typestore(Stores.ROS2_HUMBLE)

# Loop through each bag file and extract data to CSV
pbar = tqdm(bag_files, desc="Processing bag files")
for bag_file in bag_files:
    output_csv_file = bag_file.replace('.db3', '_extracted.csv')
    
    # Check if extracted file already exists
    if Path(output_csv_file).exists() and not force_extraction:
        print(f"Skipping {bag_file}: extracted file already exists. Use force_extraction=True to override.")
        pbar.update(1)
        continue

    # Extract data from bag file to a master CSV file
    with AnyReader([Path(bag_file)], default_typestore=typestore) as reader:
        # Filter connections to only those we want to extract
        connections = [x for x in reader.connections if [x.topic, x.msgtype] in topics_to_extract]
        # output_files = [open(bag_file.replace('.db3', f'_{conn.topic.replace("/", "_")}.csv'), 'w') for conn in connections]

        # Writing headers to the output CSV file considering the msg types
        header_parts = []
        for conn in connections:
            if 'pos' in conn.topic:
                header_parts.append(f"{conn.topic.replace('/', '_')}_pos_x,{conn.topic.replace('/', '_')}_pos_y,{conn.topic.replace('/', '_')}_pos_z")
            else:
                header_parts.append(f"{conn.topic.replace('/', '_')}_data")
        header = 'timestamp,' + ','.join(header_parts) + '\n'

        # Create a dict to hold all the data to be used for the DataFrame
        all_data = {col: [] for col in header.strip().split(',')}

        # Iterate through messages and write data to the pandas DataFrame
        for connection, timestamp, rawdata in reader.messages(connections=connections):
            # Process msg and write to csvfile in the correct columns
            msg = reader.deserialize(rawdata, connection.msgtype)

            # Add data to the all_data dict
            if 'pos' in connection.topic:
                all_data[f"{connection.topic.replace('/', '_')}_pos_x"].append(msg.pose.position.x)
                all_data[f"{connection.topic.replace('/', '_')}_pos_y"].append(msg.pose.position.y)
                all_data[f"{connection.topic.replace('/', '_')}_pos_z"].append(msg.pose.position.z)
            else:
                all_data[f"{connection.topic.replace('/', '_')}_data"].append(msg.data)
            all_data['timestamp'].append(timestamp * 1e-9)  # Convert to seconds

            # Add NaN for other columns not in this message
            for conn in connections:
                if conn.topic != connection.topic:
                    if 'pos' in conn.topic:
                        all_data[f"{conn.topic.replace('/', '_')}_pos_x"].append(float('nan'))
                        all_data[f"{conn.topic.replace('/', '_')}_pos_y"].append(float('nan'))
                        all_data[f"{conn.topic.replace('/', '_')}_pos_z"].append(float('nan'))
                    else:
                        all_data[f"{conn.topic.replace('/', '_')}_data"].append(float('nan'))

        # Save the DataFrame to CSV
        all_data_df = pd.DataFrame(all_data)
        all_data_df = all_data_df.sort_values(by='timestamp').reset_index(drop=True)
        all_data_df.to_csv(output_csv_file, index=False)
        pbar.update(1)

pbar.close()