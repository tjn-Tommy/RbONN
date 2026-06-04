import ctypes
from sys import flags
import time
import numpy as np
import os
from pathlib import Path


# load SLM DLL
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DLL_DIR = _PROJECT_ROOT / "SLM_DLL_ver.2.51" / "dll" / "x64"
_DLL_DIR = Path(os.environ.get("SLM_DLL_DIR", _DLL_DIR))
_DLL_PATH = _DLL_DIR / "SLMFunc.dll"

def _load_slm_dll() -> ctypes.CDLL:
    if not _DLL_PATH.exists():
        raise FileNotFoundError(f"not found SLMFunc.dll: {_DLL_PATH}")
    return ctypes.CDLL(str(_DLL_PATH))

class SLM_DVI_Driver:
    def __init__(self, display_no: int = 1):
        self.display_no = int(display_no)
        self.dll = _load_slm_dll()
        self._bind_functions()

    def _bind_functions(self):
        self.dll.SLM_Disp_Info.argtypes = [
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint16),
            ctypes.POINTER(ctypes.c_uint16),
        ]
        self.dll.SLM_Disp_Info.restype = ctypes.c_int32

        self.dll.SLM_Disp_Open.argtypes = [ctypes.c_uint32]
        self.dll.SLM_Disp_Open.restype = ctypes.c_int32

        self.dll.SLM_Disp_Close.argtypes = [ctypes.c_uint32]
        self.dll.SLM_Disp_Close.restype = ctypes.c_int32

        self.dll.SLM_Disp_GrayScale.argtypes = [
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint16
        ]
        self.dll.SLM_Disp_GrayScale.restype = ctypes.c_int32

        self.dll.SLM_Disp_ReadCSV.argtypes = [
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_wchar_p
        ]
        self.dll.SLM_Disp_ReadCSV.restype = ctypes.c_int32

    def _check_error(self, result, func_name=None):
        if result != 0:
            raise RuntimeError(f"Error in {func_name}: {result}")

    def slm_info(self):
        height = ctypes.c_uint16()
        width = ctypes.c_uint16()
        ret = self.dll.SLM_Disp_Info(self.display_no, ctypes.byref(width), ctypes.byref(height))
        self._check_error(ret, "SLM_Disp_Info")
        return width.value, height.value

    def open_slm(self):
        ret = self.dll.SLM_Disp_Open(self.display_no)
        self._check_error(ret, "SLM_Disp_Open")

    def close_slm(self):
        ret = self.dll.SLM_Disp_Close(self.display_no)
        self._check_error(ret, "SLM_Disp_Close")

    def load_csv(self, csv_path: str, interval: float = 0.2):    
        csv_path = str(Path(csv_path).resolve())
        ret = self.dll.SLM_Disp_ReadCSV(self.display_no,0,csv_path)
        self._check_error(ret, "SLM_Disp_ReadCSV")
        time.sleep(interval)

    def load_grayscale(self, grayscale: int, interval: float = 0.2):
        ret = self.dll.SLM_Disp_GrayScale(self.display_no, 0, grayscale)
        self._check_error(ret, "SLM_Disp_GrayScale")
        time.sleep(interval)

    def __enter__(self):
        self.open_slm()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close_slm()