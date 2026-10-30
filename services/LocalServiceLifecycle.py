"""本机音频服务生命周期。LocalServiceLifecycle 接收已校验配置和服务名。

external 永不探测本机端口或启动进程；managed 使用跨进程文件锁和 HTTP 健康检查，
默认空闲退出代替“响应后立即退出”，避免连续请求被上一次请求的退出定时器杀死。
"""

import os
import socket
import subprocess
import time
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from services.ServiceClient import ServiceClient
from services.pythonRuntime import service_python


class LocalServiceLifecycle:
    """只管理本工程的本机 TTS/ASR 服务，远程生命周期由部署平台负责。"""

    def __init__(self, name, config):
        """name 为 tts/asr；config 为统一注册表结果。"""
        self.name = name
        self.config = config
        self.root = Path(__file__).resolve().parent.parent

    def is_ready(self):
        """访问配置的健康接口；任何失败均视为未就绪，不把占用端口等同服务正常。"""
        cfg = dict(self.config)
        cfg["timeoutSeconds"] = min(2.0, cfg["timeoutSeconds"])
        url = urlsplit(cfg["baseURL"])
        cfg["baseURL"] = urlunsplit((url.scheme, url.netloc, "", "", ""))
        try:
            data, _ = ServiceClient(cfg).request(self.config.get("healthPath", "/health"), limit=65536)
            import json
            return json.loads(data).get("ok") is True
        except (OSError, ValueError, RuntimeError):
            return False

    def ensure_service(self, idle_seconds=300, max_chars=120, model=None):
        """可选 idle_seconds/max_chars/model 是本机启动参数；external 直接返回，由请求报告故障。"""
        if self.config["management"] == "external":
            return
        runtime = self.root / "runtime"
        runtime.mkdir(parents=True, exist_ok=True)
        with (runtime / (self.name + "-startup.lock")).open("a+b") as lock:
            if lock.seek(0, os.SEEK_END) == 0:
                lock.write(b"0")
                lock.flush()
            lock.seek(0)
            if os.name == "nt":
                import msvcrt
                # 默认锁重试仅 10 秒；启动持锁时间可能更长，使用有总上限的非阻塞等待。
                deadline = time.monotonic() + 40
                while True:
                    try:
                        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
                        break
                    except OSError:
                        if time.monotonic() >= deadline:
                            raise RuntimeError("等待本机服务启动锁超时")
                        time.sleep(0.2)
            else:
                import fcntl
                fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                if self.is_ready():
                    return
                try:
                    with socket.create_connection((self.config["host"], self.config["port"]), timeout=0.5):
                        raise RuntimeError("服务端口已监听但健康检查失败，请检查鉴权、API 版本或端口冲突")
                except OSError:
                    pass
                port = str(self.config["port"])
                key_env = self.config.get("apiKeyEnv", "")
                if self.name == "tts":
                    python = os.environ.get("BREEZE_PYTHON", service_python())
                    args = [python, str(self.root / "services" / "breezeTtsService.py"),
                            "--ckpt", os.environ.get("BREEZE_CKPT", str(self.root / "models" / "breezeTts2")),
                            "--max-chars", str(max_chars), "--exit-after-request", "0"]
                else:
                    args = [service_python(), str(self.root / "services" / "whisperService.py"),
                            "--model", model or self.config["model"]]
                args += ["--port", port, "--idle-seconds", str(idle_seconds), "--api-key-env", key_env]
                repo = Path(os.environ.get("BREEZE_REPO", str(self.root / "vendor" / "breeze-tts")))
                cwd = repo if self.name == "tts" and repo.is_dir() else self.root
                with (runtime / (self.name + "-service.log")).open("ab") as log:
                    subprocess.Popen(args, cwd=str(cwd), stdout=log, stderr=log,
                                     env={**os.environ, "PYTHONUTF8": "1"},
                                     stdin=subprocess.DEVNULL,
                                     creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                deadline = time.monotonic() + 30
                while time.monotonic() < deadline:
                    if self.is_ready():
                        return
                    time.sleep(0.5)
                raise RuntimeError(f"{self.name} 本机服务启动超时，请检查 runtime/{self.name}-service.log")
            finally:
                if os.name == "nt":
                    lock.seek(0)
                    msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(lock, fcntl.LOCK_UN)
