"""
cwb_loader.py
════════════════════════════════════════════════════════════════
[REMOVED_ZH:5] (CWB) [REMOVED_ZH:6]

[REMOVED_ZH:2]:
  [REMOVED_ZH:2] cwb_data.csv (MH [REMOVED_ZH:2])
  [REMOVED_ZH:6]：Ta (TX01), RH (RH01), WS (WD01), WD (WD02)
  [REMOVED_ZH:8] [8, 9, ..., 18]
"""

from __future__ import annotations

import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict

class CWBWeatherLoader:
    """[REMOVED_ZH:12]"""

    def __init__(self, csv_path: str):
        self.csv_path = Path(csv_path)
        self.df = None
        self._load()

    def _load(self):
        """[REMOVED_ZH:2] CWB CSV"""
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CWB data not found: {self.csv_path}")

        # CWB CSV [REMOVED_ZH:2]：[REMOVED_ZH:1] 72 [REMOVED_ZH:4]，[REMOVED_ZH:1] 73 [REMOVED_ZH:8]
        # [REMOVED_ZH:8]: stno, yyyymmddhh, PS01, TX01, RH01, WD01, WD02, ...
        self.df = pd.read_csv(
            self.csv_path,
            encoding='utf-8-sig',
            skiprows=73,  # [REMOVED_ZH:6] + [REMOVED_ZH:5]
            header=None,
            names=['stno', 'yyyymmddhh', 'PS01', 'TX01', 'RH01', 'WD01', 'WD02', 'WD07', 'WD08', 'PP01'],
            skipinitialspace=True,  # [REMOVED_ZH:9]
        )
        print(f"  [OK] Loaded {len(self.df)} CWB records")
        print(f"  Columns: {list(self.df.columns)}")

    def get_hourly_data(self, year: int, month: int, day: int,
                        hours: List[int] = None) -> Dict:
        """
        [REMOVED_ZH:10]

        Args:
            year, month, day: [REMOVED_ZH:2]
            hours: [REMOVED_ZH:4] (e.g., [8, 9, ..., 18])

        Returns:
            {
                'timestamp': [...],  # [REMOVED_ZH:4]
                'ta': [...],        # [REMOVED_ZH:2] (°C)
                'rh': [...],        # Relative Humidity (%)
                'ws': [...],        # Wind Speed (m/s)
                'wd': [...],        # [REMOVED_ZH:2] (°)
            }
        """
        if hours is None:
            hours = list(range(24))

        result = {
            'timestamp': [],
            'ta': [],
            'rh': [],
            'ws': [],
            'wd': [],
        }

        for hour in hours:
            # [REMOVED_ZH:2] yyyymmddhh
            yyyymmddhh = f"{year:04d}{month:02d}{day:02d}{hour:02d}"

            # [REMOVED_ZH:8]
            mask = self.df['yyyymmddhh'].astype(str) == yyyymmddhh

            if mask.any():
                rec = self.df[mask].iloc[0]  # [REMOVED_ZH:4]
                ta = float(rec.get('TX01', -999.9))
                rh = float(rec.get('RH01', -999.9))
                ws = float(rec.get('WD01', -999.9))
                wd = float(rec.get('WD02', -999.9))

                # [REMOVED_ZH:6] (< -100 [REMOVED_ZH:4])
                if ta < -100:
                    ta = -9.9
                if rh < -100:
                    rh = -9.9
                if ws < -100:
                    ws = -9.9
                if wd < -100:
                    wd = -9.9

                result['timestamp'].append(f"{year}-{month:02d}-{day:02d}T{hour:02d}:00:00")
                result['ta'].append(ta)
                result['rh'].append(rh)
                result['ws'].append(ws)
                result['wd'].append(wd)
            else:
                # [REMOVED_ZH:4]，[REMOVED_ZH:1] -9.9 [REMOVED_ZH:2]
                result['timestamp'].append(f"{year}-{month:02d}-{day:02d}T{hour:02d}:00:00")
                result['ta'].append(-9.9)
                result['rh'].append(-9.9)
                result['ws'].append(-9.9)
                result['wd'].append(-9.9)

        return result

    def get_monthly_statistics(self, year: int, month: int,
                               hours: List[int] = None) -> Dict:
        """
        [REMOVED_ZH:5]（[REMOVED_ZH:3]）

        Returns:
            {
                Ta_mean: float,
                RH_mean: float,
                WS_mean: float,
            }
        """
        if hours is None:
            hours = list(range(24))

        all_ta = []
        all_rh = []
        all_ws = []

        # [REMOVED_ZH:8]
        import calendar
        days_in_month = calendar.monthrange(year, month)[1]

        for day in range(1, days_in_month + 1):
            data = self.get_hourly_data(year, month, day, hours)
            all_ta.extend([x for x in data['ta'] if x > -100])
            all_rh.extend([x for x in data['rh'] if x > -100])
            all_ws.extend([x for x in data['ws'] if x > -100])

        return {
            'Ta_mean': float(sum(all_ta) / len(all_ta)) if all_ta else -9.9,
            'RH_mean': float(sum(all_rh) / len(all_rh)) if all_rh else -9.9,
            'WS_mean': float(sum(all_ws) / len(all_ws)) if all_ws else -9.9,
            'n_records': len(all_ta),
        }


if __name__ == "__main__":
    cwb_path = "inputs/cwb_data.csv"
    loader = CWBWeatherLoader(cwb_path)

    # [REMOVED_ZH:2]
    print("\n[Test] CWB Hourly Data (2025-07-15):")
    data = loader.get_hourly_data(2025, 7, 15, hours=[8, 9, 10, 14, 18])
    for i, ts in enumerate(data['timestamp']):
        print(f"  {ts}: Ta={data['ta'][i]:6.1f}°C, RH={data['rh'][i]:5.1f}%, "
              f"WS={data['ws'][i]:4.1f}m/s")

    print("\n[Test] CWB Monthly Statistics (2025-07):")
    stats = loader.get_monthly_statistics(2025, 7, hours=[8, 9, 10, 14, 18])
    print(f"  Ta_mean: {stats['Ta_mean']:.1f}°C")
    print(f"  RH_mean: {stats['RH_mean']:.1f}%")
    print(f"  WS_mean: {stats['WS_mean']:.1f}m/s")
    print(f"  n_records: {stats['n_records']}")
