"""音频服务公共 HTTP 基类：鉴权、上传限制、临时文件与二进制响应。

AudioHttp 供 Whisper/Breeze Handler 继承；token_env 由服务启动参数指定，
默认限制 34 MiB 请求、32 MiB 音频。旧本地路径接口和卸载接口仅允许回环客户端。
"""

import hmac
import json
import os
import tempfile
from email.parser import BytesParser
from email.policy import default
from http.server import BaseHTTPRequestHandler
from pathlib import Path


class AudioHttp(BaseHTTPRequestHandler):
    """无模型依赖的 HTTP 传输层；子类提供业务处理和 _reply。"""

    token_env = ""
    max_body = 34 * 1024 * 1024

    def setup(self):
        """设置连接读取超时，防止不完整上传长期占用模型请求锁。"""
        super().setup()
        self.connection.settimeout(90)

    def authorized(self):
        """校验环境变量中的 Bearer 密钥；密钥未配置时仅接受回环客户端。"""
        key = os.environ.get(self.token_env, "")
        valid = hmac.compare_digest(self.headers.get("Authorization", ""), "Bearer " + key) if key else self.is_local()
        if not valid:
            self._reply(401, {"ok": False, "error": "unauthorized"})
            self.close_connection = True
        return valid

    def is_local(self):
        """返回请求方是否为回环客户端；旧路径接口不能被远程调用。"""
        return self.client_address[0] in ("127.0.0.1", "::1")

    def read_body(self):
        """读取定长请求，拒绝分块传输、缺失长度和超限正文，避免无界分配。"""
        length = int(self.headers.get("Content-Length", "-1"))
        if self.headers.get("Transfer-Encoding") or length < 0 or length > self.max_body:
            raise ValueError("请求缺少有效 Content-Length 或超过 34 MiB")
        body = self.rfile.read(length)
        if len(body) != length:
            raise ValueError("请求正文不完整")
        return body

    def read_json(self):
        """读取 JSON 对象；TTS JSON 另限制为 4 MiB（含参考音频）。"""
        if int(self.headers.get("Content-Length", "-1")) > 4 * 1024 * 1024:
            raise ValueError("JSON 超过 4 MiB")
        data = json.loads(self.read_body())
        if not isinstance(data, dict):
            raise ValueError("JSON 必须是对象")
        return data

    def read_upload(self):
        """解析 multipart 音频及语言字段；返回临时路径和字段，调用方必须删除临时文件。"""
        kind = self.headers.get("Content-Type", "")
        if not kind.lower().startswith("multipart/form-data"):
            raise ValueError("ASR 需要 multipart/form-data")
        msg = BytesParser(policy=default).parsebytes(
            ("Content-Type: " + kind + "\r\nMIME-Version: 1.0\r\n\r\n").encode() + self.read_body())
        fields, audio, suffix = {}, None, ".audio"
        if not msg.is_multipart():
            raise ValueError("无效 multipart 请求")
        for part in msg.iter_parts():
            name = part.get_param("name", header="content-disposition")
            raw = part.get_payload(decode=True) or b""
            if name == "file":
                if audio is not None or not raw or len(raw) > 32 * 1024 * 1024:
                    raise ValueError("只接受一个不超过 32 MiB 的非空音频")
                audio = raw
                ext = Path(part.get_filename() or "").suffix.lower()
                suffix = ext if ext in (".wav", ".mp3", ".mp4", ".m4a", ".ogg", ".opus", ".webm", ".flac") else suffix
            elif name in ("language", "model", "prompt", "vad_filter"):
                if len(raw) > 16000:
                    raise ValueError("multipart 字段过长")
                fields[name] = raw.decode("utf-8")
        if audio is None:
            raise ValueError("缺少 file 字段")
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
            handle.write(audio)
            return handle.name, fields

    def reply_audio(self, path):
        """path 为本服务生成的 WAV，流式写回音频，避免向调用方泄露本机文件路径。"""
        path = Path(path)
        size = path.stat().st_size
        if size > 32 * 1024 * 1024:
            raise ValueError("音频结果超过 32 MiB")
        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(size))
        self.end_headers()
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(65536)
                if not chunk:
                    break
                self.wfile.write(chunk)
