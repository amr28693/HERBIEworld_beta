"""
Allow running as module: python -m herbie_world
With launcher GUI:       python -m herbie_world --launcher
"""

import sys

if __name__ == "__main__":
    if '--launcher' in sys.argv or '-l' in sys.argv:
        from .launcher import main as launcher_main
        launcher_main()
    else:
        from .main import main
        main()
