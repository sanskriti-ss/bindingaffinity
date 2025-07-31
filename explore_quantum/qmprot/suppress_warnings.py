#!/usr/bin/env python3
"""
Utility to suppress common warnings during VQE execution
"""

import warnings
import os
import logging

def suppress_all_warnings():
    """Suppress all common warnings"""
    # Suppress protobuf warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    # Suppress all warnings
    warnings.filterwarnings('ignore')
    
    # Configure logging to reduce verbosity
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    logging.getLogger('jax').setLevel(logging.ERROR)
    
    # Suppress specific warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf')
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    
    print("✓ Warnings suppressed for cleaner output")

if __name__ == "__main__":
    suppress_all_warnings()
