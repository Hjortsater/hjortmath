from __future__ import annotations # Fixes the "Matrix not defined" error
from functools import wraps
from typing import Callable, Any, TYPE_CHECKING
from contextlib import contextmanager
from functools import wraps
from typing import Callable, Any

@contextmanager
def internal():
    """MARKS OPERATIONS AS INTERNAL TO SURPRESS PERFORMANCE WARNINGS"""
    yield {"_internal": True}

def internal_call(func):
    @wraps(func)
    def wrapper(self, *args, _internal=False, **kwargs):
        return func(self, *args, _internal=_internal, **kwargs)
    return wrapper




if TYPE_CHECKING:
    Help = Helpers




def alias(*names):
    """
    ALLOWS FOR CLEANER IDENTITY ASSIGNMENT OF CLASS METHODS.
    """
    def decorator(obj):
        """CLASS DECORATOR"""
        if isinstance(obj, type):
            import sys
            module = sys.modules[obj.__module__]
            for name in names:
                setattr(module, name, obj)
            return obj
        
        """METHOD DECORATOR"""
        class AliasDescriptor:
            def __init__(self, func):
                self.func = func

            def __set_name__(self, owner, name):
                for alias_name in names:
                    setattr(owner, alias_name, self.func)
                setattr(owner, name, self.func)

            def __get__(self, instance, owner):

                if hasattr(self.func, "__get__"):
                    return self.func.__get__(instance, owner)
                return self.func

        return AliasDescriptor(obj)
    return decorator




if TYPE_CHECKING:
    from your_filename import Matrix 

def validate_dimensions(op_type: str) -> Callable:
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self: 'Matrix', *args: Any) -> Any:

            if op_type == "square":
                if self.m != self.n:
                    raise ValueError(f"Matrix must be square for {func.__name__} (got {self.m}x{self.n})")
                return func(self, *args)

            if not args:
                return func(self)
            
            other = args[0]
            if type(other).__name__ != "Matrix":
                return func(self, other)
            
            if op_type == "elementwise":
                if self.m != other.m or self.n != other.n:
                    raise ValueError(f"Dimensions must match for {func.__name__}: {self.m}x{self.n} vs {other.m}x{other.n}")
            
            elif op_type == "matmul":
                if self.n != other.m:
                    raise ValueError(f"Incompatible dimensions for multiplication: {self.n} != {other.m}")
            
            return func(self, other)
        return wrapper
    return decorator



def performance_warning(threshold: int = 50_000) -> Callable:
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, *args, _internal=False, **kwargs):
            # Only print warnings if NOT internal
            if not _internal:
                is_disabled = getattr(self, "disable_perf_hints", False)

                if not is_disabled and getattr(self, "force_C", False):
                    print("\033[1;33mJust a heads up!\033[0m "
                          "Program forces C, potentially slower than pure Python.")

                elif not is_disabled and not getattr(self, "use_C", True):
                    if self.m * self.n > threshold:
                        print(f"\033[1;33mJust a heads up!\033[0m "
                              f"Large matrix op ({self.m}x{self.n}) is running in pure Python.")

            return func(self, *args, _internal=_internal, **kwargs)
        return wrapper
    return decorator