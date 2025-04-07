import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Function to generate synthetic MS1 data with clear peaks
def generate_ms1_with_peaks(num_points=100, num_peaks=5, mz_range=(500, 600), rt_range=(2980, 3000)):
    """
    Generate synthetic MS1 data with Gaussian peaks.
    
    Parameters:
    - num_points: Number of data points (retention times)
    - num_peaks: Number of peaks to generate
    - mz_range: Tuple specifying the range of m/z values
    - rt_range: Tuple specifying the range of retention times
    
    Returns:
    - DataFrame with columns 'RT', 'mzarray', and 'intarray'
    """
    rt_values = np.linspace(rt_range[0], rt_range[1], num_points)  # Retention times
    mz_values = np.linspace(mz_range[0], mz_range[1], 500)  # m/z values (fine grid for better resolution)
    
    # Gaussian parameters for peaks (mean, stddev, max intensity)
    peak_centers = np.random.uniform(mz_range[0], mz_range[1], num_peaks)  # m/z positions of the peaks
    peak_widths = np.random.uniform(0.005, 0.05, num_peaks)  # Standard deviations for the peaks
    peak_intensities = np.random.uniform(50000, 500000, num_peaks)  # Maximum intensity for each peak
    
    # Prepare the dataframe to store the results
    mzarray_list = []
    intarray_list = []
    
    # Generate intensities based on Gaussian peaks for each retention time
    for rt in rt_values:
        intensities = np.zeros_like(mz_values)  # Initialize intensity array
        for i in range(num_peaks):
            # Create a Gaussian peak at the center
            gaussian_peak = peak_intensities[i] * np.exp(-0.5 * ((mz_values - peak_centers[i]) / peak_widths[i])**2)
            intensities += gaussian_peak  # Sum the Gaussian peak intensities
        
        # Add some noise to the intensity data to simulate real-world data
        noise = np.random.normal(0, 0.1, len(intensities))  # Small random noise
        intensities += noise
        
        # Save the m/z values and corresponding intensities
        mzarray_list.append(mz_values)
        intarray_list.append(intensities)
    
    # Convert lists to numpy arrays for easier handling
    mzarray_np = np.array(mzarray_list)
    intarray_np = np.array(intarray_list)
    
    # Create a DataFrame with the data
    ms1_data = pd.DataFrame({
        'RT': rt_values,
        'mzarray': list(mzarray_np),
        'intarray': list(intarray_np)
    })
    
    return ms1_data

# Example usage: Generate synthetic MS1 data with Gaussian peaks
num_points = 100
num_peaks = 5
ms1_data_with_peaks = generate_ms1_with_peaks(num_points=num_points, num_peaks=num_peaks)

# Plot the MS1 data as a heatmap with consistent RT and m/z ticks
plot_ms1_heatmap(ms1_data_with_peaks, rt_step=5, mz_step=10)
