"""Stop the machine sleeping during an unattended render."""

from __future__ import annotations

import ctypes
import sys
from collections.abc import Iterator
from contextlib import contextmanager

_ES_CONTINUOUS = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001


class _SystemPowerStatus(ctypes.Structure):
    _fields_ = [
        ("ACLineStatus", ctypes.c_ubyte),
        ("BatteryFlag", ctypes.c_ubyte),
        ("BatteryLifePercent", ctypes.c_ubyte),
        ("SystemStatusFlag", ctypes.c_ubyte),
        ("BatteryLifeTime", ctypes.c_ulong),
        ("BatteryFullLifeTime", ctypes.c_ulong),
    ]


@contextmanager
def keep_awake() -> Iterator[None]:
    """Hold off idle sleep. Closing the lid can still suspend the machine."""
    if sys.platform != "win32":
        yield
        return
    set_state = ctypes.windll.kernel32.SetThreadExecutionState
    set_state.restype = ctypes.c_uint
    set_state(ctypes.c_uint(_ES_CONTINUOUS | _ES_SYSTEM_REQUIRED))
    try:
        yield
    finally:
        set_state(ctypes.c_uint(_ES_CONTINUOUS))


def on_battery() -> bool:
    """True when a Windows laptop is unplugged; always False elsewhere."""
    if sys.platform != "win32":
        return False
    status = _SystemPowerStatus()
    if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status)):
        return False
    return status.ACLineStatus == 0
