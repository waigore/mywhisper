from __future__ import annotations

import pytest

from mywhisper.myw import MywApp


def test_myw_getattr_imports_mywapp():
    """Test that accessing MywApp attribute imports and returns the class"""
    # This should trigger __getattr__ and import MywApp
    assert MywApp is not None
    assert hasattr(MywApp, "__name__")
    assert MywApp.__name__ == "MywApp"


def test_myw_getattr_raises_attribute_error():
    """Test that accessing invalid attribute raises AttributeError"""
    import mywhisper.myw as myw_module
    
    with pytest.raises(AttributeError, match="invalid_attr"):
        _ = myw_module.invalid_attr

