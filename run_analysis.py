#!/usr/bin/env python3
"""Run the analysis pipeline. Wrapper that sets up the import path."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "uber-analysis" / "src"))

from analysis.run import main

main()
