"""OpenAI 风格微服务客户端。ServiceClient 接收服务配置，不持有模型或本地进程。

TTS 返回音频字节，ASR 上传文件字节；没有服务端路径依赖。默认限制音频 32 MiB，
禁止重定向以免把 Authorization 带到其他主机；错误不回显请求正文或密钥。
"""

import json
import mimetypes
import os
import uuid
import urllib.error
import urllib.request
from pathlib import Path

MAX_AUDIO = 32 * 1024 * 1024


class ServiceClient:
    """提供有超时、有大小限制的音频微服务请求。"""

    def __init__(self, config):
        """config 为 ServiceRegistry.get_service 的结果，密钥按每次请求读取环境。"""
        self.config = config

    def request(self, path, body=None, content_type=None, limit=MAX_AUDIO):
        """path 为 API 相对路径；body 为字节；limit 为完整响应字节上限；返回字节与 MIME。"""
        cfg = self.config
        headers = {}
        if content_type:
            headers["Content-Type"] = content_type
        key = os.environ.get(cfg.get("apiKeyEnv", ""), "")
        if key:
            headers["Authorization"] = "Bearer " + key
        req = urllib.request.Request(cfg["baseURL"] + path, data=body, headers=headers)
        # 禁止自动跟随跳转，避免认证跨主机传播。
        opener = urllib.request.build_opener()
        for handler in opener.handlers:
            if isinstance(handler, urllib.request.HTTPRedirectHandler):
                handler.redirect_request = lambda *args, **kwargs: None
        try:
            with opener.open(req, timeout=cfg["timeoutSeconds"]) as res:
                data = res.read(limit + 1)
                if len(data) > limit:
                    raise ValueError("微服务响应超过大小上限")
                return data, res.headers.get_content_type()
        except urllib.error.HTTPError as error:
            raise RuntimeError(f"微服务请求失败 HTTP {error.code}") from None

    def synthesize(self, text, **options):
        """text 为待合成文字；options 为音色或 Breeze 扩展参数；返回音频字节与格式。"""
        fmt = self.config.get("format", "wav")
        payload = {"model": self.config["model"], "input": text,
                   "voice": self.config.get("voice", "default"), "response_format": fmt}
        if self.config.get("dialect") == "breeze":
            payload.update(options)
        else:
            if options.get("refAudioBase64"):
                raise ValueError("参考音频上传需要 dialect=breeze，标准 TTS 不支持此扩展")
            payload.update({key: value for key, value in options.items()
                            if key in ("voice", "response_format", "speed")})
        data, mime = self.request("/audio/speech", json.dumps(payload).encode("utf-8"), "application/json")
        if not data or not mime.startswith("audio/"):
            raise ValueError("TTS 未返回音频字节")
        return data, fmt

    def transcribe(self, path, language=None, vad_filter=None):
        """path 为调用机音频路径；language 为可选语言代码；上传真实内容并返回 JSON 转写。"""
        path = Path(path)
        if path.stat().st_size > MAX_AUDIO:
            raise ValueError("ASR 文件超过 32 MiB")
        boundary = uuid.uuid4().hex
        fields = {"model": self.config["model"]}
        if language:
            fields["language"] = language
        if vad_filter is not None and self.config.get("dialect") == "whisper":
            fields["vad_filter"] = "true" if vad_filter else "false"
        chunks = []
        for key, value in fields.items():
            chunks.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'.encode())
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        suffix = path.suffix if path.suffix.isascii() and path.suffix[1:].isalnum() else ".audio"
        chunks.append(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="input{suffix}"\r\nContent-Type: {mime}\r\n\r\n'.encode())
        chunks += [path.read_bytes(), f"\r\n--{boundary}--\r\n".encode()]
        data, _ = self.request("/audio/transcriptions", b"".join(chunks),
                               "multipart/form-data; boundary=" + boundary, limit=2 * 1024 * 1024)
        result = json.loads(data)
        if not isinstance(result.get("text"), str) or result.get("ok") is False:
            raise ValueError("ASR 返回结构不正确")
        return result

    def chat(self, payload):
        """payload 为兼容 Chat Completions 的请求体；模型由注册表指定，返回 JSON 对象。"""
        body = {**payload, "model": self.config["model"]}
        data, _ = self.request("/chat/completions", json.dumps(body).encode("utf-8"),
                               "application/json", limit=4 * 1024 * 1024)
        result = json.loads(data)
        if not isinstance(result, dict) or not isinstance(result.get("choices"), list):
            raise ValueError("LLM 返回结构不正确")
        return result
