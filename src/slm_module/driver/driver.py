import ctypes
import os
import queue
import threading
import time
from pathlib import Path
from typing import Any, Callable


# load SLM DLL
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DLL_DIR = _PROJECT_ROOT / "SLM_DLL_ver.2.51" / "dll" / "x64"
_DLL_DIR = Path(os.environ.get("SLM_DLL_DIR", _DLL_DIR))
_DLL_PATH = _DLL_DIR / "SLMFunc.dll"

# Flags (Programmer's Guide 3.6 "BMP, CSV, Data Flags")
FLAGS_COLOR_R = 0x00000001
FLAGS_COLOR_G = 0x00000002
FLAGS_COLOR_B = 0x00000004
FLAGS_COLOR_GRAY = 0x00000008
FLAGS_RATE120 = 0x20000000

# Video interface modes (Programmer's Guide 3.2.2 SLM_Ctrl_WriteVI)
MODE_MEMORY = 0
MODE_DVI = 1

# SLM_STATUS codes (Programmer's Guide 3.5)
SLM_OK = 0
SLM_BS = 2

_STATUS_NAMES = {
    0: "SLM_OK",
    1: "SLM_NG",
    2: "SLM_BS (busy)",
    3: "SLM_ER (parameter error)",
    -1: "SLM_INVALID_MONITOR (display number not found)",
    -2: "SLM_NOT_OPEN_MONITOR (display not opened)",
    -3: "SLM_OPEN_WINDOW_ERR (window open error)",
    -4: "SLM_DATA_FORMAT_ERR (data format error)",
    -101: "SLM_FILE_READ_ERR (file not found or value over 1023)",
    -200: "SLM_NOT_OPEN_USB (USB not opened)",
    -1000: "SLM_OTHER_ERROR",
}


def _describe_status(code: int) -> str:
    if code in _STATUS_NAMES:
        return _STATUS_NAMES[code]
    if -10032 <= code <= -10001:
        return f"FTDI USB driver error ({code})"
    return f"unknown status ({code})"


def _load_slm_dll() -> ctypes.CDLL:
    if not _DLL_PATH.exists():
        raise FileNotFoundError(f"not found SLMFunc.dll: {_DLL_PATH}")
    if hasattr(os, "add_dll_directory"):
        # so the dependent FTD3XX.dll next to SLMFunc.dll is found
        os.add_dll_directory(str(_DLL_DIR))
    return ctypes.CDLL(str(_DLL_PATH))


def _make_message_pump():
    """Win32 message pump for windows owned by the device thread.

    SLM_Disp_Open creates the SLM window on the calling thread; pumping its
    pending messages while idle mirrors the vendor sample, which drives the
    DLL from a message-pumping UI thread. Returns a no-op off Windows.
    """
    if not hasattr(ctypes, "windll"):
        return lambda: None
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    msg = wintypes.MSG()
    pm_remove = 0x0001

    def pump() -> None:
        try:
            while user32.PeekMessageW(ctypes.byref(msg), None, 0, 0, pm_remove):
                user32.TranslateMessage(ctypes.byref(msg))
                user32.DispatchMessageW(ctypes.byref(msg))
        except Exception:
            pass

    return pump


class _DeviceThread:
    """Executes every DLL call on one long-lived thread.

    The SLM window created by SLM_Disp_Open belongs to the thread that called
    it, and the vendor samples call all display functions from that same
    single thread. Pooled worker threads (which expire and get replaced) or a
    separate keep-alive thread calling the DLL directly can therefore stall
    inside the DLL. Funneling every call through this thread both pins the
    window to a thread that never dies and serializes all DLL entry.
    """

    _IDLE_POLL_SECONDS = 0.05

    def __init__(self):
        self._jobs: queue.SimpleQueue = queue.SimpleQueue()
        self._thread = threading.Thread(target=self._run, name="slm-dll", daemon=True)
        self._thread.start()

    def call(self, label: str, func: Callable[[], Any], timeout: float = 30.0) -> Any:
        """Run func() on the device thread and return its result.

        Raises RuntimeError when the call does not finish within timeout so a
        stalled DLL call becomes a visible error instead of blocking every
        later operation forever.
        """
        done = threading.Event()
        box: dict[str, Any] = {}
        self._jobs.put((func, box, done))
        if not done.wait(timeout):
            raise RuntimeError(
                f"{label}: SLM DLL call did not return within {timeout:.0f} s; "
                "the display link may be stalled"
            )
        if "error" in box:
            raise box["error"]
        return box["result"]

    def _run(self) -> None:
        pump = _make_message_pump()
        while True:
            try:
                job = self._jobs.get(timeout=self._IDLE_POLL_SECONDS)
            except queue.Empty:
                pump()
                continue
            func, box, done = job
            try:
                box["result"] = func()
            except BaseException as exc:
                box["error"] = exc
            finally:
                done.set()


class SLM_DVI_Driver:
    """DVI-mode driver following the documented flow (Guide 1.3.2):

    search display (SLM_Disp_Info/Info2) -> SLM_Disp_Open ->
    display functions -> SLM_Disp_Close

    All DLL calls run on a dedicated device thread (see _DeviceThread);
    public methods are safe to call from any thread.
    """

    def __init__(self, display_no: int = 1, rate120: bool = False):
        self.display_no = int(display_no)
        self.flags = FLAGS_RATE120 if rate120 else 0
        self.is_open = False
        self.dll = _load_slm_dll()
        self._bind_functions()
        self._device = _DeviceThread()

    def _bind_functions(self):
        self.dll.SLM_Disp_Info.argtypes = [
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint16),
            ctypes.POINTER(ctypes.c_uint16),
        ]
        self.dll.SLM_Disp_Info.restype = ctypes.c_int32

        self.dll.SLM_Disp_Info2.argtypes = [
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint16),
            ctypes.POINTER(ctypes.c_uint16),
            ctypes.c_char_p,
        ]
        self.dll.SLM_Disp_Info2.restype = ctypes.c_int32

        self.dll.SLM_Disp_Open.argtypes = [ctypes.c_uint32]
        self.dll.SLM_Disp_Open.restype = ctypes.c_int32

        self.dll.SLM_Disp_Close.argtypes = [ctypes.c_uint32]
        self.dll.SLM_Disp_Close.restype = ctypes.c_int32

        self.dll.SLM_Disp_GrayScale.argtypes = [
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint16,
        ]
        self.dll.SLM_Disp_GrayScale.restype = ctypes.c_int32

        self.dll.SLM_Disp_ReadCSV.argtypes = [
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_wchar_p,
        ]
        self.dll.SLM_Disp_ReadCSV.restype = ctypes.c_int32

        # USB control functions for switching Memory/DVI mode (Guide 1.3.1)
        self.dll.SLM_Ctrl_Open.argtypes = [ctypes.c_uint32]
        self.dll.SLM_Ctrl_Open.restype = ctypes.c_int32

        self.dll.SLM_Ctrl_Close.argtypes = [ctypes.c_uint32]
        self.dll.SLM_Ctrl_Close.restype = ctypes.c_int32

        self.dll.SLM_Ctrl_WriteVI.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
        self.dll.SLM_Ctrl_WriteVI.restype = ctypes.c_int32

        self.dll.SLM_Ctrl_ReadVI.argtypes = [
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32),
        ]
        self.dll.SLM_Ctrl_ReadVI.restype = ctypes.c_int32

        self.dll.SLM_Ctrl_ReadSU.argtypes = [ctypes.c_uint32]
        self.dll.SLM_Ctrl_ReadSU.restype = ctypes.c_int32

    def _check_error(self, result, func_name=None):
        if result != SLM_OK:
            raise RuntimeError(f"Error in {func_name}: {_describe_status(result)}")

    def slm_info(self, display_no: int | None = None):
        display_no = self.display_no if display_no is None else int(display_no)
        height = ctypes.c_uint16()
        width = ctypes.c_uint16()
        ret = self._device.call(
            "SLM_Disp_Info",
            lambda: self.dll.SLM_Disp_Info(
                display_no, ctypes.byref(width), ctypes.byref(height)
            ),
        )
        self._check_error(ret, "SLM_Disp_Info")
        return width.value, height.value

    def slm_info2(self, display_no: int | None = None):
        """Return (width, height, display_name) for a display.

        display_name is "UserFriendlyName,ManufactureName,ProductCodeID,SerialNumberID";
        the SLM reports a name starting with "LCOS-SLM" (Guide 2.4.2).
        """
        display_no = self.display_no if display_no is None else int(display_no)
        height = ctypes.c_uint16()
        width = ctypes.c_uint16()
        name = ctypes.create_string_buffer(128)
        ret = self._device.call(
            "SLM_Disp_Info2",
            lambda: self.dll.SLM_Disp_Info2(
                display_no, ctypes.byref(width), ctypes.byref(height), name
            ),
        )
        self._check_error(ret, "SLM_Disp_Info2")
        return width.value, height.value, name.value.decode("mbcs", errors="replace")

    def search_displays(self, max_display: int = 8):
        """Documented "search display number" step: probe displays and
        return [(display_no, width, height, name)] for every display found."""
        found = []
        for display_no in range(1, max_display + 1):
            try:
                width, height, name = self.slm_info2(display_no)
            except RuntimeError:
                continue
            found.append((display_no, width, height, name))
        return found

    def open_slm(self):
        ret = self._device.call(
            "SLM_Disp_Open", lambda: self.dll.SLM_Disp_Open(self.display_no)
        )
        self._check_error(ret, "SLM_Disp_Open")
        self.is_open = True

    def close_slm(self):
        ret = self._device.call(
            "SLM_Disp_Close", lambda: self.dll.SLM_Disp_Close(self.display_no)
        )
        self.is_open = False
        self._check_error(ret, "SLM_Disp_Close")

    def load_csv(self, csv_path: str, interval: float = 0.2, flags: int | None = None):
        csv_path = str(Path(csv_path).resolve())
        flags = self.flags if flags is None else int(flags)
        # the settle sleep stays on the calling thread so it never delays
        # other device-thread jobs
        ret = self._device.call(
            "SLM_Disp_ReadCSV",
            lambda: self.dll.SLM_Disp_ReadCSV(self.display_no, flags, csv_path),
        )
        self._check_error(ret, "SLM_Disp_ReadCSV")
        time.sleep(interval)

    def load_grayscale(self, grayscale: int, interval: float = 0.2, flags: int | None = None):
        flags = self.flags if flags is None else int(flags)
        ret = self._device.call(
            "SLM_Disp_GrayScale",
            lambda: self.dll.SLM_Disp_GrayScale(self.display_no, flags, grayscale),
        )
        self._check_error(ret, "SLM_Disp_GrayScale")
        time.sleep(interval)

    def set_video_mode(self, mode: int, slm_number: int = 1, timeout: float = 60.0):
        """Switch the SLM between Memory (0) and DVI (1) mode over USB.

        Follows Guide 1.3.1: SLM_Ctrl_Open -> wait ready (SLM_Ctrl_ReadSU)
        -> SLM_Ctrl_WriteVI -> SLM_Ctrl_Close.
        """
        if mode not in (MODE_MEMORY, MODE_DVI):
            raise ValueError("mode must be 0 (Memory) or 1 (DVI)")
        slm_number = int(slm_number)

        # the whole USB session runs as one device-thread job so it is never
        # interleaved with other DLL calls
        def session():
            ret = self.dll.SLM_Ctrl_Open(slm_number)
            self._check_error(ret, "SLM_Ctrl_Open")
            try:
                deadline = time.monotonic() + timeout
                while True:
                    ret = self.dll.SLM_Ctrl_ReadSU(slm_number)
                    if ret == SLM_OK:
                        break
                    if ret != SLM_BS or time.monotonic() >= deadline:
                        self._check_error(ret, "SLM_Ctrl_ReadSU")
                        raise RuntimeError("SLM_Ctrl_ReadSU: timed out waiting for ready")
                    time.sleep(0.5)
                ret = self.dll.SLM_Ctrl_WriteVI(slm_number, mode)
                self._check_error(ret, "SLM_Ctrl_WriteVI")
            finally:
                self.dll.SLM_Ctrl_Close(slm_number)

        self._device.call("SLM_Ctrl_WriteVI session", session, timeout=timeout + 30.0)

    def get_video_mode(self, slm_number: int = 1) -> int:
        """Read the current Memory/DVI mode over USB."""
        slm_number = int(slm_number)

        def session():
            ret = self.dll.SLM_Ctrl_Open(slm_number)
            self._check_error(ret, "SLM_Ctrl_Open")
            try:
                mode = ctypes.c_uint32()
                ret = self.dll.SLM_Ctrl_ReadVI(slm_number, ctypes.byref(mode))
                self._check_error(ret, "SLM_Ctrl_ReadVI")
                return mode.value
            finally:
                self.dll.SLM_Ctrl_Close(slm_number)

        return int(self._device.call("SLM_Ctrl_ReadVI session", session))

    def ping(self, slm_number: int = 1, verify_video_mode: bool = False) -> int | None:
        """Heartbeat over USB to keep the SLM responsive (keep-alive).

        Sends SLM_Ctrl_ReadSU, the documented command for resynchronizing
        communication; SLM_BS (busy) still counts as alive. Optionally reads
        back the video mode so callers can verify DVI is still active.
        """
        slm_number = int(slm_number)

        def session():
            ret = self.dll.SLM_Ctrl_Open(slm_number)
            self._check_error(ret, "SLM_Ctrl_Open")
            try:
                ret = self.dll.SLM_Ctrl_ReadSU(slm_number)
                if ret not in (SLM_OK, SLM_BS):
                    self._check_error(ret, "SLM_Ctrl_ReadSU")
                if not verify_video_mode:
                    return None
                mode = ctypes.c_uint32()
                ret = self.dll.SLM_Ctrl_ReadVI(slm_number, ctypes.byref(mode))
                self._check_error(ret, "SLM_Ctrl_ReadVI")
                return mode.value
            finally:
                self.dll.SLM_Ctrl_Close(slm_number)

        return self._device.call("SLM_Ctrl_ReadSU session", session)

    def __enter__(self):
        self.open_slm()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close_slm()
