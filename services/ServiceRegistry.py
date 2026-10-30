"""统一服务配置。ServiceRegistry 接收可选配置路径；默认读取 config/serviceConfig.json。

只负责地址、鉴权引用和生命周期归属，不启动进程、不加载模型。management 默认 external，
managed 仅允许回环地址；显式环境地址覆盖后默认 external，防止误启动或监控重启。
"""

import json
import os
from pathlib import Path
from urllib.parse import urlsplit


class ServiceRegistry:
    """管理 TTS、ASR、LLM 服务配置，密钥只从 apiKeyEnv 指定的环境变量读取。"""

    def __init__(self, path=None):
        """path 为独立服务配置文件；未传时允许 SYNLOVE_SERVICE_CONFIG 覆盖默认路径。"""
        self.path = Path(path or os.environ.get("SYNLOVE_SERVICE_CONFIG") or
                         Path(__file__).resolve().parent.parent / "config" / "serviceConfig.json")
        self.data = json.loads(self.path.read_text(encoding="utf-8-sig"))
        if self.data.get("version") != 1:
            raise ValueError("serviceConfig.version 必须是 1")

    def get_service(self, name):
        """name 为 tts/asr/llm；校验并返回服务副本，拒绝带内嵌凭据或非法协议的地址。"""
        cfg = dict(self.data["services"][name])
        prefix = "SYNLOVE_" + name.upper()
        override = os.environ.get(prefix + "_BASE_URL")
        if override:
            cfg["baseURL"] = override
            cfg["management"] = "external"
        cfg["management"] = os.environ.get(prefix + "_MANAGEMENT", cfg.get("management", "external"))
        cfg["baseURL"] = str(cfg["baseURL"]).rstrip("/")
        url = urlsplit(cfg["baseURL"])
        if (url.scheme not in ("http", "https") or not url.hostname or
                url.username or url.password or url.query or url.fragment):
            raise ValueError(f"{name} 的 baseURL 必须是无内嵌凭据的 HTTP(S) API 根地址")
        cfg["port"] = url.port or (443 if url.scheme == "https" else 80)
        cfg["host"] = url.hostname
        if cfg["management"] not in ("managed", "external"):
            raise ValueError(f"{name}.management 只允许 managed 或 external")
        if cfg["management"] == "managed" and url.hostname not in ("127.0.0.1", "localhost"):
            raise ValueError(f"{name} 托管服务必须使用回环地址；远程服务请设 external")
        if cfg["management"] == "managed" and (url.scheme != "http" or url.path.rstrip("/") != "/v1"):
            raise ValueError(f"{name} 本机托管服务使用 http://回环地址:端口/v1；代理地址请设 external")
        cfg["timeoutSeconds"] = float(cfg.get("timeoutSeconds", 240))
        if not 0 < cfg["timeoutSeconds"] <= 3600:
            raise ValueError(f"{name}.timeoutSeconds 必须在 (0, 3600] 内")
        if not isinstance(cfg.get("model"), str) or not cfg["model"].strip():
            raise ValueError(f"{name}.model 不能为空")
        return cfg

    def apply_environment(self, env):
        """env 为子进程环境；同步模型客户端和服务归属，外部服务禁用本机显存门控。"""
        cfg = self.get_service("llm")
        env["SYNLOVE_SERVICE_CONFIG"] = str(self.path.resolve())
        env["GEMMA_LOCAL_BASE_URL"] = cfg["baseURL"]
        env["DSH_GEMMA_MODEL"] = str(cfg["model"])
        env["SYNLOVE_LLM_MANAGEMENT"] = cfg["management"]
        env["GEMMA_SERVER_PORT"] = str(cfg["port"])
        env["SYNLOVE_LLM_API_KEY_ENV"] = cfg.get("apiKeyEnv", "")
        env["SYNLOVE_LLM_TIMEOUT_MS"] = str(int(cfg["timeoutSeconds"] * 1000))
        if cfg["management"] == "external":
            env["DSH_GEMMA_MIN_FREE_VRAM_MB"] = "0"

    def apply_model_config(self, config, env=None):
        """config 为已合并的模型配置；仅覆盖 llm.provider 指定的本地能力提供方。"""
        cfg = self.get_service("llm")
        provider = config.get("providers", {}).get(cfg.get("provider", "gemma-local"))
        if provider is not None:
            provider["baseURL"] = cfg["baseURL"]
            key_env = cfg.get("apiKeyEnv") or "SYNLOVE_LLM_API_KEY"
            # Harness/兼容 SDK 要求非空凭据；无鉴权服务使用明确占位值，不复用旧供应商密钥。
            key = (os.environ if env is None else env).get(key_env) or "synlove-no-auth"
            provider["apiKeyEnv"] = key_env
            provider["apiKey"] = key
            if cfg.get("api"):
                provider["api"] = cfg["api"]
            models = provider.get("models", [])
            if models:
                provider["models"] = [{**models[0], "id": cfg["model"]}]
            default = config.get("defaultModel", {})
            if default.get("provider") == cfg.get("provider", "gemma-local"):
                default["model"] = cfg["model"]
