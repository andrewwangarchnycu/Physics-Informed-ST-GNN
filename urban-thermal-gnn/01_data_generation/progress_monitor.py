"""
progress_monitor.py
════════════════════════════════════════════════════════════════
Real-time progress monitoring - terminal progress bar + ETA estimation
"""

from __future__ import annotations

import time
import sys
from typing import Dict, Optional


class ProgressMonitor:
    """Real-time progress monitor - terminal progress bar + ETA"""

    def __init__(self, total: int, phase_name: str = "Phase"):
        self.total = max(1, total)
        self.phase_name = phase_name
        self.current = 0
        self.start_time = time.time()
        self.last_update_time = self.start_time
        self.metrics_history = []
        self.bar_length = 30

    def update(self, current: int, metrics: Optional[Dict] = None):
        """
        Update progress

        Args:
            current: Current progress (0..total)
            metrics: Additional metrics dict, e.g., {"nodes": 1500, "utci_mean": 28.5}
        """
        self.current = current
        now = time.time()

        # Only update every 0.5 seconds (avoid excessive flicker)
        if now - self.last_update_time < 0.5 and current < self.total:
            return

        self.last_update_time = now
        elapsed = now - self.start_time

        # ETA calculation
        if self.current > 0:
            rate = self.current / max(elapsed, 0.1)
            remaining = (self.total - self.current) / max(rate, 0.001)
        else:
            remaining = 0

        # Progress bar rendering
        progress = min(self.current / self.total, 1.0)
        filled = int(self.bar_length * progress)
        bar = "[" + "=" * filled + " " * (self.bar_length - filled) + "]"

        # Format time
        def fmt_time(seconds):
            h = int(seconds // 3600)
            m = int((seconds % 3600) // 60)
            s = int(seconds % 60)
            if h > 0:
                return f"{h}h {m}m"
            elif m > 0:
                return f"{m}m {s}s"
            else:
                return f"{s}s"

        elapsed_str = fmt_time(elapsed)
        remaining_str = fmt_time(remaining)

        # Main progress line
        progress_line = (
            f"\r{bar} {self.current:4d}/{self.total} | "
            f"Elapsed: {elapsed_str:12s} | ETA: {remaining_str:12s}"
        )

        # Metrics line (if any)
        if metrics:
            metrics_str = " | ".join(
                f"{k}={v}" for k, v in list(metrics.items())[:3]
            )
            progress_line += f"\n  {metrics_str}"

        try:
            sys.stdout.write(progress_line)
            sys.stdout.flush()
        except UnicodeEncodeError:
            # Fallback: replace problematic characters
            try:
                progress_line_safe = progress_line.encode('ascii', errors='replace').decode('ascii')
                sys.stdout.write(progress_line_safe)
                sys.stdout.flush()
            except:
                # Last resort: just try stderr
                try:
                    sys.stderr.write(progress_line + "\n")
                except:
                    pass

        if metrics:
            self.metrics_history.append((self.current, metrics))

    def summary(self):
        """Completion summary"""
        total_time = time.time() - self.start_time

        def fmt_time(seconds):
            h = int(seconds // 3600)
            m = int((seconds % 3600) // 60)
            s = int(seconds % 60)
            if h > 0:
                return f"{h}h {m}m {s}s"
            else:
                return f"{m}m {s}s"

        summary_text = (
            "\n\n"
            f"{'-'*60}\n"
            f"  {self.phase_name} Complete\n"
            f"  Total: {self.total} items\n"
            f"  Time: {fmt_time(total_time)}\n"
            f"  Rate: {self.total/max(total_time, 0.1):.1f} items/sec\n"
            f"{'-'*60}\n"
        )

        try:
            sys.stdout.write(summary_text)
            sys.stdout.flush()
        except UnicodeEncodeError:
            try:
                summary_safe = summary_text.encode('ascii', errors='replace').decode('ascii')
                sys.stdout.write(summary_safe)
                sys.stdout.flush()
            except:
                try:
                    sys.stderr.write(summary_text + "\n")
                except:
                    pass


if __name__ == "__main__":
    # Test
    import time

    monitor = ProgressMonitor(100, "Test Phase")
    for i in range(101):
        metrics = {
            "nodes": 1000 + i * 10,
            "utci_mean": 28.0 + i * 0.1,
        }
        monitor.update(i, metrics)
        time.sleep(0.03)

    monitor.summary()
