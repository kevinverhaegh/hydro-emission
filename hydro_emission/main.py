import sys
import os

# Add the directory where emission_tools.py is located
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Import the necessary module after updating sys.path
import emission_tools

if __name__ == "__main__":
    import numpy as np
    result = emission_tools.Balmer_line_emissivity(3, 1e19*np.ones((2,2)), 1e18*np.ones((2,2)), 1e18*np.ones((2,2)), 1e19*np.ones((2,2)), 1*np.ones((2,2)))
    print(result)