"""
Optional energy monitoring using Zeus.

This module provides a lightweight wrapper around ZeusMonitor that:
- Lazily imports Zeus (warns if not installed)
- Provides context manager for energy measurement
- Returns EnergyMeasurement dataclass with gpu_energy and total_energy
"""

import warnings
from dataclasses import dataclass
from typing import Optional


@dataclass
class EnergyMeasurement:
    """Result of an energy measurement window."""

    gpu_energy: dict[int, float]
    total_energy: float


class EnergyMonitor:
    """
    Context manager for Zeus energy measurement.

    This class wraps ZeusMonitor to provide optional energy monitoring.
    If Zeus is not installed, all operations become no-ops and a warning is logged.

    Usage:
        monitor = EnergyMonitor(gpu_indices=[0, 1])
        with monitor:
            # code to measure
            pass
        measurement = monitor.last_measurement
    """

    def __init__(
        self,
        gpu_indices: Optional[list[int]] = None,
        window_name: str = "default",
        approx_instant_energy: bool = False,
    ):
        """
        Initialize the energy monitor.

        Args:
            gpu_indices: List of GPU indices to monitor. If None, all GPUs are used.
            window_name: Name for the measurement window.
            approx_instant_energy: If True, approximate energy for short measurement windows
                using instant power draw x window duration. Useful when batches are fast.
        """
        self.gpu_indices = gpu_indices
        self.window_name = window_name
        self.approx_instant_energy = approx_instant_energy
        self._monitor = None
        self._enabled = False
        self._measurement: Optional[EnergyMeasurement] = None

    def _init_monitor(self) -> None:
        """Lazily initialize the ZeusMonitor."""
        if self._monitor is not None:
            return

        try:
            from zeus.monitor import ZeusMonitor

            try:
                self._monitor = ZeusMonitor(
                    gpu_indices=self.gpu_indices,
                    approx_instant_energy=self.approx_instant_energy,
                )
                self._enabled = True
            except RuntimeError as e:
                if "torch" in str(e).lower() or "cuda" in str(e).lower():
                    warnings.warn(
                        f"ZeusMonitor initialization failed (likely torch/CUDA issue): {e}. "
                        "Energy monitoring disabled."
                    )
                else:
                    warnings.warn(
                        f"ZeusMonitor initialization failed: {e}. "
                        "Energy monitoring disabled."
                    )
                self._enabled = False
        except ImportError:
            warnings.warn(
                "zeus not installed. Energy monitoring disabled. "
                "Install with: pip install lm-eval[energy]"
            )
            self._enabled = False

    def begin_window(self, key: str) -> None:
        """Begin a measurement window."""
        if not self._enabled:
            return
        self._init_monitor()
        if self._enabled and self._monitor is not None:
            self._monitor.begin_window(key)

    def end_window(self, key: str) -> EnergyMeasurement:
        """End a measurement window and return the energy measurement."""
        if not self._enabled:
            return EnergyMeasurement(gpu_energy={}, total_energy=0.0)
        if self._monitor is None:
            return EnergyMeasurement(gpu_energy={}, total_energy=0.0)

        result = self._monitor.end_window(key)
        measurement = EnergyMeasurement(
            gpu_energy=result.gpu_energy,
            total_energy=result.total_energy,
        )
        self._measurement = measurement
        return measurement

    def __enter__(self) -> "EnergyMonitor":
        """Enter the context manager."""
        self._init_monitor()
        if self._enabled and self._monitor is not None:
            try:
                self._monitor.begin_window(self.window_name)
            except RuntimeError as e:
                warnings.warn(
                    f"ZeusMonitor begin_window failed: {e}. "
                    "Energy monitoring will be disabled for this measurement."
                )
                self._enabled = False
        return self

    def __exit__(self, *args) -> None:
        """Exit the context manager."""
        if self._enabled and self._monitor is not None:
            try:
                self.end_window(self.window_name)
            except RuntimeError as e:
                warnings.warn(
                    f"ZeusMonitor end_window failed: {e}. "
                    "Energy measurement may be incomplete."
                )

    @property
    def last_measurement(self) -> Optional[EnergyMeasurement]:
        """Get the last energy measurement."""
        return self._measurement
