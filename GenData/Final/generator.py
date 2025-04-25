import numpy as np
import pandas as pd
import plotly.express as px
import os
# Gaussian Generation Imports
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


class Generator:
    """
    A class to generate retention time (RT) and mass-to-charge (MZ) sampling grids
    with controlled random variations.

    Parameters:
        rt_start (float): Start of the RT range.
        rt_end (float): End of the RT range.
        rt_steps (int): Number of RT sampling points.
        rt_variation (float): Max variation applied to RT step sizes (relative).
        mz_start (float): Start of the MZ range.
        mz_end (float): End of the MZ range.
        mz_min_steps (int): Minimum number of MZ steps per RT.
        mz_max_steps (int): Maximum number of MZ steps per RT.
        mz_variation (float): Max variation applied to MZ step sizes (relative).
    """
    def __init__(self, rt_start, rt_end, rt_steps, rt_variation,
                 mz_start, mz_end, mz_min_steps, mz_max_steps, mz_variation):

        # Retention time configuration
        self.rt_start = rt_start
        self.rt_end = rt_end
        self.rt_steps = rt_steps
        self.rt_variation = rt_variation

        # Mass-to-charge ratio configuration
        self.mz_start = mz_start
        self.mz_end = mz_end
        self.mz_min_steps = mz_min_steps
        self.mz_max_steps = mz_max_steps
        self.mz_variation = mz_variation

    # Grid Generation
    def generate_grid(self) -> pd.DataFrame:
        """
        Public method. Generates a synthetic RT-MZ grid with irregular spacing.
        """
        rt_points = self._rt_variable_linspace()
        mz_column = [self._mz_variable_linspace() for _ in rt_points]
        return pd.DataFrame({"rt": rt_points, "mz": mz_column})
    
    def _rt_variable_linspace(self) -> list:
        """
        Generates a 1D array of RT values with controlled irregular spacing.

        Returns:
            list[float]: Irregularly spaced RT values from rt_start to rt_end.
        """
        if self.rt_steps < 2:
            raise ValueError("rt_steps must be at least 2")

        # Generate evenly spaced RT values and compute steps
        regular = np.linspace(self.rt_start, self.rt_end, self.rt_steps)
        steps = np.diff(regular)

        # Apply relative variation to internal steps
        variation = np.mean(steps) * self.rt_variation
        noise = np.random.uniform(-variation, variation, self.rt_steps - 2)
        noise -= np.mean(noise)  # zero-sum to maintain total span

        # Modify steps with noise
        modified_steps = steps.copy()
        modified_steps[:-1] += noise

        # Rebuild irregular RT points from modified steps
        irregular = [self.rt_start]
        for step in modified_steps:
            irregular.append(irregular[-1] + step)

        return irregular

    def _mz_variable_linspace(self) -> list:
        """
        Generates an irregularly spaced MZ array with random step count and controlled variation.

        Returns:
            list[float]: Irregularly spaced MZ values from mz_start to mz_end.
        """
        if self.mz_min_steps < 2:
            raise ValueError("mz_min_steps must be at least 2")

        # Randomly choose number of MZ steps within the user-defined range
        num_steps = np.random.randint(self.mz_min_steps, self.mz_max_steps)

        # Generate evenly spaced MZ values and compute steps
        regular = np.linspace(self.mz_start, self.mz_end, num_steps)
        steps = np.diff(regular)

        # Apply relative variation to internal steps
        variation = np.mean(steps) * self.mz_variation
        noise = np.random.uniform(-variation, variation, num_steps - 2)
        noise -= np.mean(noise)  # zero-sum constraint

        # Modify steps with noise
        modified_steps = steps.copy()
        modified_steps[:-1] += noise

        # Rebuild irregular MZ points from modified steps
        irregular = [self.mz_start]
        for step in modified_steps:
            irregular.append(irregular[-1] + step)

        return irregular

    # Gaussian Generation CHange noise to %
    def generate_gaussians(self, grid: pd.DataFrame, peak_params: list, noise_std=0.01) -> pd.DataFrame:
        """
        Applies Gaussian peaks and noise to a given RT/MZ grid.

        Parameters:
            grid (pd.DataFrame): A DataFrame containing 'rt' and 'mz' columns.
            peak_params (list): List of dictionaries defining Gaussian peaks.
            noise_std (float): Standard deviation of Gaussian noise added to intensities.

        Returns:
            pd.DataFrame: The input grid with an added 'intensities' column.
        """
        intensities = []

        for rt_val, mz_array in zip(grid["rt"], grid["mz"]):
            mz_array = np.array(mz_array)
            raw_intensity = np.zeros_like(mz_array)

            for peak in peak_params:
                peak_intensity = self._gaussian_pdf(
                    rt=rt_val,
                    mz_array=mz_array,
                    center_rt=peak["rt_center"],
                    center_mz=peak["mz_center"],
                    sigma_rt=peak["rt_sigma"],
                    sigma_mz=peak["mz_sigma"],
                    amplitude=peak["amplitude"]
                )
                raw_intensity += peak_intensity

            noise = np.random.normal(0, noise_std, size=raw_intensity.shape)
            noisy_intensity = np.clip(raw_intensity + noise, 0, None)
            intensities.append(noisy_intensity.tolist())

        # Return new DataFrame with intensity column
        result = grid.copy()
        result["intensities"] = intensities
        return result
    
    def _gaussian_pdf(self, rt, mz_array, center_rt, center_mz, sigma_rt, sigma_mz, amplitude=1) -> float:
        """
        Computes 2D Gaussian intensity for a single RT value and MZ array.
        """
        rt_gauss = np.exp(-0.5 * ((rt - center_rt) / sigma_rt) ** 2)
        mz_gauss = np.exp(-0.5 * ((mz_array - center_mz) / sigma_mz) ** 2)
        intensity = amplitude * (1 / (2 * np.pi * sigma_mz * sigma_rt)) * rt_gauss * mz_gauss
        return intensity

    # Plotting Functions
    def plot_grid(self, df):
        """
        Plots a scatterplot of RT vs MZ from the generated grid.

        Parameters:
            df (pd.DataFrame): The grid DataFrame to visualize.
        """
        scatter_data = df.explode('mz') # Expand MZ lists into separate rows

        fig = px.scatter(scatter_data, x='rt', y='mz', 
                         title='RT vs MZ Scatterplot', 
                         labels={'rt': 'RT', 'mz': 'MZ'})
        
        # Customize plot appearance
        fig.update_layout(
            width=1200,
            height=800,
            template="plotly_white",
            yaxis=dict(range=[self.mz_start, self.mz_start + 0.5]),
            title_x=0.5,
            title_font=dict(size=20),
            font=dict(size=12)
        )
        fig.update_traces(marker=dict(size=2, opacity=0.8))
        fig.update_xaxes(title_text='RT', title_font=dict(size=14))
        fig.update_yaxes(title_text='MZ', title_font=dict(size=14))
        fig.show()

    def plot_gaussians_grid(self, df, title="Interpolated mz Heatmap", zoom=False, mz_points=1000):
        """
        Plots an interpolated heatmap of intensity data across RT and MZ.
        """
        rt_values = df["rt"].values
        all_mz = np.concatenate(df["mz"].values)
        mz_min, mz_max = all_mz.min(), all_mz.max()
        common_mz = np.linspace(mz_min, mz_max, mz_points)

        interpolated_matrix = []
        for mz, intensity in zip(df["mz"], df["intensities"]):
            interp = interp1d(mz, intensity, kind='linear', bounds_error=False, fill_value=0)
            interpolated_matrix.append(interp(common_mz))

        intensity_matrix = np.array(interpolated_matrix).T

        # Matplotlib heatmap
        plt.figure(figsize=(12, 5))
        plt.imshow(intensity_matrix, extent=[rt_values[0], rt_values[-1], common_mz[-1], common_mz[0]],
                   aspect='auto', cmap='viridis')
        plt.colorbar(label='Intensity')
        plt.xlabel('Retention Time (min)')
        plt.ylabel('mz (Da)')
        plt.title(title)

        if zoom:
            plt.xlim(zoom.get('xlim', (rt_values[0], rt_values[-1])))
            plt.ylim(zoom.get('ylim', (common_mz[0], common_mz[-1])))

        plt.show()

        # Plotly version
        fig = px.imshow(
            intensity_matrix,
            origin='lower',
            aspect='auto',
            labels=dict(x="Retention Time", y="m/z", color="Intensity"),
            x=rt_values,
            y=common_mz,
            color_continuous_scale="Viridis"
        )
        fig.update_layout(
            title="2D Image Plot of LC-MS Data",
            width=800,
            height=600
        )
        fig.show()

    # Export Functions
    def grid_to_json(self, df, base_filename="v2_grid"):
        """
        Saves the grid DataFrame to a JSON file with an auto-incremented name to avoid overwriting.

        Parameters:
            df (pd.DataFrame): The grid to save.
            base_filename (str): Base name for the output file (without extension).
        """
        filename = f"{base_filename}.json"
        counter = 1

        # Ensure unique filename by appending a counter if needed
        while os.path.exists(filename):
            filename = f"{base_filename}_{counter}.json"
            counter += 1

        df.to_json(filename, index=False)
        print(f"File saved as: {filename}")

    def gaussians_grid_to_json(self, df, base_filename="gaussians_grid", output_folder="gaussian_grids"):
        """
        Saves the Gaussian intensity DataFrame to a uniquely named JSON file.
        """
        os.makedirs(output_folder, exist_ok=True)
        filename = f"{base_filename}.json"
        filepath = os.path.join(output_folder, filename)
        counter = 1

        while os.path.exists(filepath):
            filename = f"{base_filename}_{counter}.json"
            filepath = os.path.join(output_folder, filename)
            counter += 1

        df.to_json(filepath, index=False)
        print(f"File saved as: {filepath}")