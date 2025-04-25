# Generator Class Documentation

The `Generator` class simulates LC-MS data by generating irregular retention time (RT) and mass-to-charge (MZ) grids, applying synthetic Gaussian peaks, and supporting plotting and export functions. Ideal for testing algorithms and models in mass spectrometry workflows.

---

## Initialization

```python
Generator(
    rt_start, rt_end, rt_steps, rt_variation,
    mz_start, mz_end, mz_min_steps, mz_max_steps, mz_variation
)

```
**Parameters**:

rt_start / rt_end: Range of retention times.

rt_steps: Number of RT sampling points.

rt_variation: Max variation in RT step size (relative).

mz_start / mz_end: Range of m/z values.

mz_min_steps / mz_max_steps: Range of number of m/z steps per RT.

mz_variation: Max variation in m/z step size (relative).

## Grid Generation

generate_grid() -> pd.DataFrame

Generates an irregular RT/MZ sampling grid.

```python
grid = generator.generate_grid()
```

**Returns:**

A DataFrame with:

rt: Retention time values.

mz: A list of m/z values per RT.

## Gaussian Intensity Simulation

generate_gaussians(grid: pd.DataFrame, peak_params: list, noise_std=0.01) -> pd.DataFrame

Simulates signal intensity using 2D Gaussians and adds optional noise.

```python
intensity_df = generator.generate_gaussians(grid, peak_params)
```

**Parameters:**

grid: Output from generate_grid()

peak_params: List of peak dictionaries

noise_std: Gaussian noise standard deviation

**Returns:**

Original grid with an additional intensities column.

## Plotting Functions

plot_grid(df)

Plots RT vs. MZ as a scatterplot using Plotly.

```python
generator.plot_grid(grid)
plot_gaussians_grid(df, title="Interpolated mz Heatmap", zoom=False, mz_points=1000)
```
Generates heatmaps (matplotlib + Plotly) from interpolated intensity data.

```python
generator.plot_gaussians_grid(intensity_df)
```

**Parameters**:

- df: DataFrame with rt, mz, and intensities.
- title: Title of the plot.
- zoom: Optional dict with xlim and ylim ranges.
- mz_points: Number of interpolated m/z bins.

## Saving Data
save_grid(df, base_filename="v2_grid")
Saves a raw RT/MZ grid to a JSON file with unique naming.

save_gaussians_grid(df, base_filename="gaussians_grid", output_folder="gaussian_grids")
Saves an intensity-enhanced DataFrame to JSON in a specified folder.

## Example Peak Parameters

```python
peak_params = [
    {"rt_center": 12.0, "mz_center": 153.0, "rt_sigma": 0.1, "mz_sigma": 0.02, "amplitude": 20000},
    {"rt_center": 13.2, "mz_center": 154.3, "rt_sigma": 0.15, "mz_sigma": 0.03, "amplitude": 15000}
]
```

## Full Example Usage

```python
from generator import Generator

# Initialize generator
gen = Generator(
    rt_start=10, rt_end=15, rt_steps=100, rt_variation=0.1,
    mz_start=150, mz_end=160, mz_min_steps=990, mz_max_steps=1010, mz_variation=0.1
)

# Generate RT/MZ grid
grid = gen.generate_grid()

# Apply Gaussian peaks
intensity_df = gen.generate_gaussians(grid, peak_params)

# Plot data
gen.plot_grid(grid)
gen.plot_gaussians_grid(intensity_df)

# Save data
gen.save_grid(grid, base_filename="my_grid")
gen.save_gaussians_grid(intensity_df, base_filename="my_gaussians")

```