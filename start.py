# Import necessary libraries
from pyopenms import MSExperiment, MzMLFile
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Function to load mzML file
def load_mzml_file(file_path):
    experiment = MSExperiment()
    MzMLFile().load(file_path, experiment)
    data = experiment.get_df()
    return experiment, data

def plot_spectrum(df, index):
    plt.figure(figsize=(10, 6))
    plt.stem(df.iloc[index]['mzarray'], df.iloc[index]['intarray'], linefmt="grey", markerfmt="D", basefmt="black")
    plt.title(f'Spectrum at Retention Time = {df.iloc[index]["RT"]}')
    plt.xlabel('m/z')
    plt.ylabel('Intensity')
    plt.show()


def plot_data_3d(df):
    plt.scatter(df["mzarray"], df["intarray"])
    plt.show()


# Load the mzML file
file_path = "C:\\Users\\victo\\VSCode Folder\\UMCG Mass Spec\\3_2_extract_overlap_1.mzML"
experiment, data = load_mzml_file(file_path)

# Get longest peak lines
x = 10
sorted_df = data.assign(list_length=data["intarray"].apply(len)).sort_values(by='list_length', ascending=False)
x_longest_lists = sorted_df.head(x)
print(x, "entries with the longest lists:")
for index, row in x_longest_lists.iterrows():
    print(f"\nLength: {row['list_length']}\nList: {row['intarray']}")
