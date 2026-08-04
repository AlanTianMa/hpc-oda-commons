"""
Experimental models for runtime prediction.

These models follow the production RollingTabularModel pattern and can be
evaluated with the same rolling-window framework as the production models.
They are intended for experimentation and benchmarking — promising models
may be promoted to production after evaluation.
"""

from __future__ import annotations
