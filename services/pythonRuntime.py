"""选择运行本地服务（如 whisper）使用的 Python 解释器。

职责：
- 优先读环境变量 `WHISPER_PYTHON` / `DSH_PYTHON`（便于换机器）；
- 默认使用发行目录的 .venv 解释器；
- 都不存在时退回当前解释器，至少不会崩在路径判断上。

注册参数：无。调用方传项目根与模型参数即可。
"""

import os
import sys
from pathlib import Path

DEFAULT_PYTHON = str(Path(__file__).resolve().parent.parent / ".venv" /
                     ("Scripts/python.exe" if os.name == "nt" else "bin/python"))


def service_python():
    """返回用于启动本地服务的解释器路径（优先带依赖的那个）。"""
    for key in ("WHISPER_PYTHON", "DSH_PYTHON"):
        value = os.environ.get(key)
        if value and Path(value).is_file():
            return value
    if Path(DEFAULT_PYTHON).is_file():
        return DEFAULT_PYTHON
    return sys.executable
