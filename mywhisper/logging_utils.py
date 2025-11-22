"""
Automatic function logging utilities for mywhisper.

Provides decorators and metaclasses to automatically log function inputs and outputs
at appropriate log levels (INFO for public methods, DEBUG for private methods).
"""

from __future__ import annotations

import functools
import inspect
import logging
from enum import Enum
from typing import Any, Callable, Optional, TypeVar, cast

F = TypeVar("F", bound=Callable[..., Any])


class SerializationMode(Enum):
    """Serialization modes for function inputs and outputs."""

    SUMMARIZED = "summarized"  # Default: type, size, key attributes
    FULL = "full"  # Full JSON-like serialization with truncation
    CUSTOM = "custom"  # Per-function configuration


def _summarize_value(value: Any, max_length: int = 200) -> str:
    """
    Summarize a value for logging purposes.

    Default strategy: type, size, and key attributes for complex objects.
    """
    if value is None:
        return "None"

    value_type = type(value).__name__

    # Handle basic types
    if isinstance(value, (str, int, float, bool)):
        str_repr = str(value)
        if len(str_repr) > max_length:
            return f"{value_type}({str_repr[:max_length]}...)"
        return f"{value_type}({str_repr})"

    # Handle sequences
    if isinstance(value, (list, tuple)):
        length = len(value)
        if length == 0:
            return f"{value_type}(empty)"
        # Show first item's type and total length
        first_type = type(value[0]).__name__ if length > 0 else "unknown"
        return f"{value_type}(len={length}, first={first_type})"

    # Handle mappings
    if isinstance(value, dict):
        length = len(value)
        if length == 0:
            return f"{value_type}(empty)"
        # Show keys if few, or just count
        if length <= 5:
            keys = list(value.keys())[:5]
            return f"{value_type}(len={length}, keys={keys})"
        return f"{value_type}(len={length})"

    # Handle Path objects
    if hasattr(value, "__fspath__"):
        return f"Path({value})"

    # Handle dataclasses and objects with common attributes
    if hasattr(value, "__dict__"):
        attrs = {k: type(v).__name__ for k, v in list(value.__dict__.items())[:3]}
        return f"{value_type}({attrs})"

    # Generic object summary
    str_repr = str(value)
    if len(str_repr) > max_length:
        return f"{value_type}({str_repr[:max_length]}...)"
    return f"{value_type}({str_repr})"


def serialize_input(value: Any, mode: SerializationMode = SerializationMode.SUMMARIZED) -> str:
    """Serialize an input value for logging."""
    if mode == SerializationMode.SUMMARIZED:
        return _summarize_value(value)
    if mode == SerializationMode.FULL:
        try:
            import json

            json_str = json.dumps(value, default=str, ensure_ascii=False)
            if len(json_str) > 500:
                return f"{json_str[:500]}..."
            return json_str
        except (TypeError, ValueError):
            return _summarize_value(value)
    return str(value)


def serialize_output(value: Any, mode: SerializationMode = SerializationMode.SUMMARIZED) -> str:
    """Serialize an output value for logging."""
    return serialize_input(value, mode)


def _is_public_method(name: str) -> bool:
    """Check if a method name indicates it's public (no underscore prefix)."""
    return not name.startswith("_")


def _get_log_level_for_method(name: str, override: Optional[int] = None) -> int:
    """Determine log level for a method based on its name."""
    if override is not None:
        return override
    return logging.INFO if _is_public_method(name) else logging.DEBUG


def _get_logger_for_function(func: Callable[..., Any]) -> logging.Logger:
    """Get or create a logger for a function based on its module and qualname."""
    module = inspect.getmodule(func)
    if module is None:
        module_name = "unknown"
    else:
        module_name = module.__name__

    qualname = getattr(func, "__qualname__", func.__name__)
    logger_name = f"{module_name}.{qualname.split('.')[0]}"
    return logging.getLogger(logger_name)


def _should_log_function(func: Callable[..., Any]) -> bool:
    """Check if a function should be logged (not marked with _no_log)."""
    return not getattr(func, "_no_log", False)


def log_function(
    level: Optional[int] = None,
    serialize: SerializationMode = SerializationMode.SUMMARIZED,
    log_inputs: bool = True,
    log_output: bool = True,
) -> Callable[[F], F]:
    """
    Decorator to log function inputs and outputs.

    Parameters
    ----------
    level
        Log level override. If None, uses INFO for public functions, DEBUG for private.
    serialize
        Serialization mode for inputs/outputs. Default is SUMMARIZED.
    log_inputs
        Whether to log function inputs. Default True.
    log_output
        Whether to log function output. Default True.
    """
    def decorator(func: F) -> F:
        if not _should_log_function(func):
            return func

        func_name = func.__name__
        log_level = _get_log_level_for_method(func_name, level)
        logger = _get_logger_for_function(func)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Log inputs
            if log_inputs:
                sig = inspect.signature(func)
                bound_args = sig.bind(*args, **kwargs)
                bound_args.apply_defaults()

                arg_reprs = []
                for param_name, param_value in bound_args.arguments.items():
                    if param_name == "self":
                        continue  # Skip self parameter
                    serialized = serialize_input(param_value, serialize)
                    arg_reprs.append(f"{param_name}={serialized}")

                inputs_str = ", ".join(arg_reprs)
                logger.log(log_level, "Calling %s(%s)", func_name, inputs_str)

            # Call function
            result = func(*args, **kwargs)

            # Log output
            if log_output:
                # For generators, log entry/exit but not each yield
                if inspect.isgeneratorfunction(func):
                    logger.log(log_level, "Generator %s created", func_name)
                else:
                    output_str = serialize_output(result, serialize)
                    logger.log(log_level, "%s returned %s", func_name, output_str)

            return result

        # Handle async functions
        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                if log_inputs:
                    sig = inspect.signature(func)
                    bound_args = sig.bind(*args, **kwargs)
                    bound_args.apply_defaults()

                    arg_reprs = []
                    for param_name, param_value in bound_args.arguments.items():
                        if param_name == "self":
                            continue
                        serialized = serialize_input(param_value, serialize)
                        arg_reprs.append(f"{param_name}={serialized}")

                    inputs_str = ", ".join(arg_reprs)
                    logger.log(log_level, "Calling async %s(%s)", func_name, inputs_str)

                result = await func(*args, **kwargs)

                if log_output:
                    output_str = serialize_output(result, serialize)
                    logger.log(log_level, "async %s returned %s", func_name, output_str)

                return result

            return cast(F, async_wrapper)

        return cast(F, wrapper)

    return decorator


class LoggingMeta(type):
    """
    Metaclass that automatically wraps all methods in a class with logging.

    Public methods (no underscore prefix) are logged at INFO level.
    Private methods (underscore prefix) are logged at DEBUG level.
    """

    def __new__(
        mcs,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        **kwargs: Any,
    ) -> type:
        """Create a new class with logging wrapped methods."""
        # Get logger for this class
        module = namespace.get("__module__", "unknown")
        logger_name = f"{module}.{name}"
        class_logger = logging.getLogger(logger_name)

        # Wrap methods
        new_namespace = {}
        for key, value in namespace.items():
            # Skip special/dunder methods
            if key.startswith("__") and key.endswith("__"):
                new_namespace[key] = value
                continue

            # Handle properties (don't wrap)
            if isinstance(value, property):
                new_namespace[key] = value
                continue

            # Handle classmethods
            if isinstance(value, classmethod):
                original_func = value.__func__
                if callable(original_func) and _should_log_function(original_func):
                    log_level = _get_log_level_for_method(key)
                    wrapped_func = _wrap_classmethod_with_logging(original_func, key, class_logger, log_level)
                    new_namespace[key] = classmethod(wrapped_func)
                else:
                    new_namespace[key] = value
                continue

            # Handle staticmethods
            if isinstance(value, staticmethod):
                original_func = value.__func__
                if callable(original_func) and _should_log_function(original_func):
                    log_level = _get_log_level_for_method(key)
                    wrapped_func = _wrap_staticmethod_with_logging(original_func, key, class_logger, log_level)
                    new_namespace[key] = staticmethod(wrapped_func)
                else:
                    new_namespace[key] = value
                continue

            # Handle regular methods
            if callable(value):
                if _should_log_function(value):
                    log_level = _get_log_level_for_method(key)
                    wrapped = _wrap_method_with_logging(value, key, class_logger, log_level)
                    new_namespace[key] = wrapped
                else:
                    new_namespace[key] = value
            else:
                new_namespace[key] = value

        return super().__new__(mcs, name, bases, new_namespace, **kwargs)


def _wrap_method_with_logging(
    method: Callable[..., Any],
    method_name: str,
    logger: logging.Logger,
    log_level: int,
) -> Callable[..., Any]:
    """Wrap a method with logging, handling various method types."""
    # Check if already wrapped
    if hasattr(method, "_logging_wrapped"):
        return method

    # Don't wrap properties or classmethods/staticmethods that need special handling
    if isinstance(method, (property, classmethod, staticmethod)):
        return method

    @functools.wraps(method)
    def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        # Try to use instance logger if available
        instance_logger = getattr(self, "logger", None)
        if instance_logger is None:
            instance_logger = logger

        # Log inputs
        sig = inspect.signature(method)
        try:
            bound_args = sig.bind(self, *args, **kwargs)
            bound_args.apply_defaults()
        except TypeError:
            # If binding fails, log without details
            bound_args = None

        if bound_args:
            arg_reprs = []
            for param_name, param_value in bound_args.arguments.items():
                if param_name == "self":
                    continue
                serialized = serialize_input(param_value, SerializationMode.SUMMARIZED)
                arg_reprs.append(f"{param_name}={serialized}")

            inputs_str = ", ".join(arg_reprs) if arg_reprs else "(no args)"
            instance_logger.log(log_level, "Calling %s.%s(%s)", type(self).__name__, method_name, inputs_str)
        else:
            instance_logger.log(log_level, "Calling %s.%s(...)", type(self).__name__, method_name)

        # Call method
        result = method(self, *args, **kwargs)

        # Log output (skip for generators - they log entry only)
        if not inspect.isgeneratorfunction(method):
            output_str = serialize_output(result, SerializationMode.SUMMARIZED)
            instance_logger.log(log_level, "%s.%s returned %s", type(self).__name__, method_name, output_str)
        else:
            instance_logger.log(log_level, "Generator %s.%s created", type(self).__name__, method_name)

        return result

    # Handle async methods
    if inspect.iscoroutinefunction(method):
        @functools.wraps(method)
        async def async_wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
            instance_logger = getattr(self, "logger", None)
            if instance_logger is None:
                instance_logger = logger

            sig = inspect.signature(method)
            try:
                bound_args = sig.bind(self, *args, **kwargs)
                bound_args.apply_defaults()
            except TypeError:
                bound_args = None

            if bound_args:
                arg_reprs = []
                for param_name, param_value in bound_args.arguments.items():
                    if param_name == "self":
                        continue
                    serialized = serialize_input(param_value, SerializationMode.SUMMARIZED)
                    arg_reprs.append(f"{param_name}={serialized}")

                inputs_str = ", ".join(arg_reprs) if arg_reprs else "(no args)"
                instance_logger.log(log_level, "Calling async %s.%s(%s)", type(self).__name__, method_name, inputs_str)
            else:
                instance_logger.log(log_level, "Calling async %s.%s(...)", type(self).__name__, method_name)

            result = await method(self, *args, **kwargs)

            output_str = serialize_output(result, SerializationMode.SUMMARIZED)
            instance_logger.log(log_level, "async %s.%s returned %s", type(self).__name__, method_name, output_str)

            return result

        async_wrapped._logging_wrapped = True  # type: ignore[attr-defined]
        return async_wrapped

    wrapped._logging_wrapped = True  # type: ignore[attr-defined]
    return wrapped


def _wrap_classmethod_with_logging(
    method: Callable[..., Any],
    method_name: str,
    logger: logging.Logger,
    log_level: int,
) -> Callable[..., Any]:
    """Wrap a classmethod with logging."""
    if hasattr(method, "_logging_wrapped"):
        return method

    @functools.wraps(method)
    def wrapped(cls: Any, *args: Any, **kwargs: Any) -> Any:
        # Log inputs
        sig = inspect.signature(method)
        try:
            bound_args = sig.bind(cls, *args, **kwargs)
            bound_args.apply_defaults()
        except TypeError:
            bound_args = None

        if bound_args:
            arg_reprs = []
            for param_name, param_value in bound_args.arguments.items():
                if param_name == "cls":
                    continue
                serialized = serialize_input(param_value, SerializationMode.SUMMARIZED)
                arg_reprs.append(f"{param_name}={serialized}")

            inputs_str = ", ".join(arg_reprs) if arg_reprs else "(no args)"
            logger.log(log_level, "Calling %s.%s(%s)", cls.__name__, method_name, inputs_str)
        else:
            logger.log(log_level, "Calling %s.%s(...)", cls.__name__, method_name)

        result = method(cls, *args, **kwargs)

        if not inspect.isgeneratorfunction(method):
            output_str = serialize_output(result, SerializationMode.SUMMARIZED)
            logger.log(log_level, "%s.%s returned %s", cls.__name__, method_name, output_str)
        else:
            logger.log(log_level, "Generator %s.%s created", cls.__name__, method_name)

        return result

    if inspect.iscoroutinefunction(method):
        @functools.wraps(method)
        async def async_wrapped(cls: Any, *args: Any, **kwargs: Any) -> Any:
            sig = inspect.signature(method)
            try:
                bound_args = sig.bind(cls, *args, **kwargs)
                bound_args.apply_defaults()
            except TypeError:
                bound_args = None

            if bound_args:
                arg_reprs = []
                for param_name, param_value in bound_args.arguments.items():
                    if param_name == "cls":
                        continue
                    serialized = serialize_input(param_value, SerializationMode.SUMMARIZED)
                    arg_reprs.append(f"{param_name}={serialized}")

                inputs_str = ", ".join(arg_reprs) if arg_reprs else "(no args)"
                logger.log(log_level, "Calling async %s.%s(%s)", cls.__name__, method_name, inputs_str)
            else:
                logger.log(log_level, "Calling async %s.%s(...)", cls.__name__, method_name)

            result = await method(cls, *args, **kwargs)

            output_str = serialize_output(result, SerializationMode.SUMMARIZED)
            logger.log(log_level, "async %s.%s returned %s", cls.__name__, method_name, output_str)

            return result

        async_wrapped._logging_wrapped = True  # type: ignore[attr-defined]
        return async_wrapped

    wrapped._logging_wrapped = True  # type: ignore[attr-defined]
    return wrapped


def _wrap_staticmethod_with_logging(
    method: Callable[..., Any],
    method_name: str,
    logger: logging.Logger,
    log_level: int,
) -> Callable[..., Any]:
    """Wrap a staticmethod with logging."""
    if hasattr(method, "_logging_wrapped"):
        return method

    @functools.wraps(method)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        # Log inputs
        sig = inspect.signature(method)
        try:
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
        except TypeError:
            bound_args = None

        if bound_args:
            arg_reprs = []
            for param_name, param_value in bound_args.arguments.items():
                serialized = serialize_input(param_value, SerializationMode.SUMMARIZED)
                arg_reprs.append(f"{param_name}={serialized}")

            inputs_str = ", ".join(arg_reprs) if arg_reprs else "(no args)"
            logger.log(log_level, "Calling staticmethod %s(%s)", method_name, inputs_str)
        else:
            logger.log(log_level, "Calling staticmethod %s(...)", method_name)

        result = method(*args, **kwargs)

        if not inspect.isgeneratorfunction(method):
            output_str = serialize_output(result, SerializationMode.SUMMARIZED)
            logger.log(log_level, "staticmethod %s returned %s", method_name, output_str)
        else:
            logger.log(log_level, "Generator staticmethod %s created", method_name)

        return result

    if inspect.iscoroutinefunction(method):
        @functools.wraps(method)
        async def async_wrapped(*args: Any, **kwargs: Any) -> Any:
            sig = inspect.signature(method)
            try:
                bound_args = sig.bind(*args, **kwargs)
                bound_args.apply_defaults()
            except TypeError:
                bound_args = None

            if bound_args:
                arg_reprs = []
                for param_name, param_value in bound_args.arguments.items():
                    serialized = serialize_input(param_value, SerializationMode.SUMMARIZED)
                    arg_reprs.append(f"{param_name}={serialized}")

                inputs_str = ", ".join(arg_reprs) if arg_reprs else "(no args)"
                logger.log(log_level, "Calling async staticmethod %s(%s)", method_name, inputs_str)
            else:
                logger.log(log_level, "Calling async staticmethod %s(...)", method_name)

            result = await method(*args, **kwargs)

            output_str = serialize_output(result, SerializationMode.SUMMARIZED)
            logger.log(log_level, "async staticmethod %s returned %s", method_name, output_str)

            return result

        async_wrapped._logging_wrapped = True  # type: ignore[attr-defined]
        return async_wrapped

    wrapped._logging_wrapped = True  # type: ignore[attr-defined]
    return wrapped


class LoggingBase(metaclass=LoggingMeta):
    """
    Base class that provides automatic method logging.

    Classes that inherit from this will automatically have their methods wrapped
    with logging at appropriate levels (INFO for public, DEBUG for private).

    Example:
        class MyClass(LoggingBase):
            def public_method(self, arg: str) -> str:  # Logged at INFO
                return f"Hello {arg}"

            def _private_method(self, arg: int) -> int:  # Logged at DEBUG
                return arg * 2
    """

    pass

