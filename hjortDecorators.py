from typing import Any, Callable
import functools

def alias(*names: str) -> Callable[[Callable], Callable]:
    def decorator(obj: Any) -> Any:
        if isinstance(obj, property):
            return AliasProperty(obj, names)
        elif isinstance(obj, (classmethod, staticmethod)):
            return AliasMethod(obj, names)
        else:
            return AliasMethod(obj, names)
    return decorator

class AliasMethod:
    def __init__(self, func: Any, alias_names: tuple[str, ...]):
        self.func = func
        self.alias_names = alias_names

    def __set_name__(self, owner: type, name: str):
        for alias_name in self.alias_names:
            setattr(owner, alias_name, self.func)

    def __get__(self, instance: Any, owner: type = None) -> Any:
        return self.func.__get__(instance, owner)

class AliasProperty(property):
    def __init__(self, prop: property, alias_names: tuple[str, ...]):
        super().__init__(prop.fget, prop.fset, prop.fdel, prop.__doc__)
        self.alias_names = alias_names

    def __set_name__(self, owner: type, name: str):
        super().__set_name__(owner, name)
        for alias_name in self.alias_names:
            setattr(owner, alias_name, self)


def lazy(threshold=256):
    def decorator(method):
        def wrapper(self, *args, **kwargs):
            from hjortMatrix import LazyMatrix, SETTINGS
            Op = LazyMatrix.OpEnum
            
            val = getattr(SETTINGS, "lazy_eval", 0)
            op_map = {"__add__": Op.ADD, "__sub__": Op.SUB, "__mul__": Op.MUL}
            
            if (val == 2) or (val == 1 and (self.m * self.n) <= threshold):
                other = args[0]
                return LazyMatrix(self, [(op_map[method.__name__], other)])

            return method(self, *args, **kwargs)
        return wrapper
    return decorator