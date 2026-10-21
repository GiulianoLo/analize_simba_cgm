"""Debugging and profiling helpers."""

import psutil


def print_ram_usage():
    """Print the current process RAM usage in MB."""
    process = psutil.Process()
    ram_mb = process.memory_info().rss / (1024 ** 2)
    print(f"RAM Usage: {ram_mb:.2f} MB")
