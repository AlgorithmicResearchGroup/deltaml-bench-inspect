"""Inspect-native DeltaMLBench runtime."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("deltamlbench-inspect")
except PackageNotFoundError:
    __version__ = "0.1.0"
