import numpy as np
import pandas as pd

class TelemetryProcessor:
    def __init__(self, window_size_min: int = 15):
        self.window_size_min = window_size_min

    def compute_physiological_strain_index(self, telemetry_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates a real-time strain index based on heart rate reserve (HRR), 
        accelerometer-derived activity counts, and autonomic stress markers.
        """
        df = telemetry_df.copy()
        
        # Example feature extraction per DBDP standards
        df['rolling_hr_mean'] = df['heart_rate'].rolling(window=f'{self.window_size_min}min').mean()
        df['rmssd_metric'] = df['rr_intervals'].rolling(window=f'{self.window_size_min}min').apply(
            lambda x: np.sqrt(np.mean(np.diff(x) ** 2)) if len(x) > 1 else 0
        )
        
        # Composite physiological load score
        df['strain_index'] = (
            0.6 * (df['rolling_hr_mean'] / df['hr_max']) + 
            0.4 * (1.0 / (df['rmssd_metric'] + 1e-5))
        )
        
        return df