import pandas as pd
import glob
import os

# Print current working directory for debugging
current_dir = os.getcwd()
print(f"Current working directory: {current_dir}")

absolute_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "csv"))

if os.path.exists(absolute_path):
    print(f"Directory exists: {absolute_path}")
    # List all files in the directory
    all_files = os.listdir(absolute_path)
    print(f"Files in directory: {all_files}")
else:
    print(f"Directory does not exist: {absolute_path}")
    exit()

csv_folder = absolute_path

# Get a list of all analysis CSV files
analysis_files = glob.glob(os.path.join(csv_folder, "*_analysis.csv"))
print(f"Found {len(analysis_files)} analysis files: {analysis_files}")

# Initialize an empty DataFrame for the merged training data
training_data = pd.DataFrame()

for analysis_file in analysis_files:
    # Example: if analysis file is "behind_the_neck_1_analysis.csv", 
    # the corresponding regular file is "behind_the_neck_1.csv"
    base_name = os.path.basename(analysis_file)
    base_name_without_analysis = base_name.replace("_analysis", "")
    regular_file = os.path.join(csv_folder, base_name_without_analysis)
    
    print(f"Looking for regular file: {regular_file}")
    
    if os.path.exists(regular_file):
        print(f"Merging features from {regular_file} with annotations from {analysis_file}")
        df_features = pd.read_csv(regular_file)
        df_labels = pd.read_csv(analysis_file)
        
        # Group the frame-level features by rep.
        df_features_grouped = df_features.groupby("rep").mean().reset_index()
        
        # Merge the grouped features with the analysis labels on the rep column.
        df_merged = pd.merge(df_features_grouped, df_labels, on="rep", how="inner")
        
        # Append to the overall training data
        training_data = pd.concat([training_data, df_merged], ignore_index=True)
    else:
        print(f"Regular file not found for {analysis_file}")

# Save the merged training data to a CSV for further use
output_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(os.path.dirname(output_dir), "merged_training_data.csv")
print(f"Saving merged data to {output_path}")
training_data.to_csv(output_path, index=False)
print(f"Saved {len(training_data)} rows of merged data")

# Display the head of the merged training data
print(training_data.head())
