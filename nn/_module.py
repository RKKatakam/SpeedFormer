import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class
import dataclasses
from dataclasses import fields
from typing import Generic, TypeVar




# ------------------------------------------------------------------------------
# Utility: Static type wrapper
# ------------------------------------------------------------------------------
T = TypeVar("T")
class Static(Generic[T]):
    """
    A type wrapper to indicate that a field is static.
    For example, writing:
    
        in_features: Static[int]
    
    will mark `in_features` as static so that it is not traversed
    by JAX transformations.
    """
    pass

# ------------------------------------------------------------------------------
# Hidden Implementation: Module base class with static field support
# ------------------------------------------------------------------------------
class ModuleMeta(type):
    def __new__(metacls, name, bases, namespace):
        # Create the class.
        cls = super().__new__(metacls, name, bases, namespace)
        # Automatically turn the class into a dataclass.
        cls = dataclasses.dataclass(cls)
        # Register the class as a JAX PyTree node.
        register_pytree_node_class(cls)
        # Also allow explicit marking of static fields by name via __static_fields__,
        # but this is optional if you use the Static[...] annotation.
        if not hasattr(cls, "__static_fields__"):
            cls.__static_fields__ = ()
        return cls

class Module(metaclass=ModuleMeta):
    """
    Base module class for building layers or models in JAX.
    
    **Usage:**
      - Subclass Module to define your layer or model.
      - Define your fields as normal attributes.
      - To mark a field as static, annotate its type with Static, e.g.:
      
            in_features: Static[int]
    
    All the necessary boilerplate is hidden.
    """
    __static_fields__ = ()  # Optionally, you can list static field names here.

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement the __call__ method.")

    # Flatten the module into dynamic leaves (trainable parameters) and static auxiliary data.
    def tree_flatten(self):
        dynamic_leaves = []
        dynamic_keys = []
        static_data = {}
        for f in fields(self):
            value = getattr(self, f.name)
            # Determine if this field should be treated as static.
            # Priority: if the field name is in __static_fields__ OR its type annotation
            # is of the form Static[...], then treat it as static.
            is_static = f.name in self.__static_fields__
            if not is_static:
                origin = getattr(f.type, "__origin__", None)
                if origin is Static:
                    is_static = True
            if is_static:
                static_data[f.name] = value
            else:
                dynamic_leaves.append(value)
                dynamic_keys.append(f.name)
        return dynamic_leaves, (dynamic_keys, static_data)

    # Reconstruct the module from flattened leaves and auxiliary data.
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        dynamic_keys, static_data = aux_data
        # Create a new (uninitialized) instance without calling __init__
        obj = object.__new__(cls)
        # Set the dynamic fields.
        for key, child in zip(dynamic_keys, children):
            setattr(obj, key, child)
        # Set the static fields.
        for key, value in static_data.items():
            setattr(obj, key, value)
        return obj