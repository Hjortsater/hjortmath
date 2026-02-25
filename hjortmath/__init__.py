"""MATMATH - A matrix library with C backend for performance"""

from .imports import (
    random, time, 
    Self, Any, Union, List, Tuple, Optional,
    ctypes, os
)

from .customdecorators import (
    alias,
    validate_dimensions,
    performance_warning,
)

from .pymat import Matrix

from .cmat import (
    mat_det,
    mat_inv,
    mat_add,
    mat_sub,
    mat_mul,
    hadamard,
)

__all__ = [
    # Main class
    'Matrix',
    
    # C functions
    'mat_det', 'mat_inv', 'mat_add', 'mat_sub', 'mat_mul', 'hadamard',
    
    # Decorators
    'alias', 'validate_dimensions', 'performance_warning',
    
    # Types and modules
    'Self', 'Any', 'Union', 'List', 'Tuple', 'Optional',
    'random', 'time', 'ctypes', 'os',
]

__version__ = '0.1.0'