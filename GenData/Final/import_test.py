from generator import Generator


gg = Generator(rt_start=10, rt_end=15, rt_steps=100, rt_variation=0.1,
    mz_start=150, mz_end=160, mz_min_steps=990, mz_max_steps=1010, mz_variation=0.1
)

df = gg.generate_grid()

# Define peak
peak_params = [
    {"rt_center": 11.5, "mz_center": 152.1, "rt_sigma": 0.1, "mz_sigma": 0.03, "amplitude": 10000},
    {"rt_center": 12.0, "mz_center": 153.0, "rt_sigma": 0.15, "mz_sigma": 0.025, "amplitude": 25000},
    {"rt_center": 13.0, "mz_center": 154.5, "rt_sigma": 0.2, "mz_sigma": 0.04, "amplitude": 18000},
    {"rt_center": 12.25, "mz_center": 153.12, "rt_sigma": 0.15, "mz_sigma": 0.025, "amplitude": 20000},
    {"rt_center": 13.35, "mz_center": 154.425, "rt_sigma": 0.2, "mz_sigma": 0.04, "amplitude": 45000},
    # Adding more is possible
]

df2 = gg.generate_gaussians(grid=df, peak_params=peak_params)

gg.plot_gaussians_grid(df2)