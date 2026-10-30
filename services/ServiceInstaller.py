"""独立服务安装器：管理 ASR、TTS、本地 LLM 和音乐 API 的隔离环境与启动命令。

注册参数 root：服务根目录，默认本文件上级；配置 config/serviceInstall.json。
模型仅在显式 --download-model 时下载，ASR 自身首次转写的懒下载除外。
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path


class ServiceInstaller:
    """按独立清单安装和启动一个服务，不依赖 SynLove 主项目或其他服务仓库。"""

    def __init__(self, root=None):
        """root 为仓库根目录；加载不含凭据的服务安装清单。"""
        self.root = Path(root or Path(__file__).resolve().parent.parent).resolve()
        self.config = json.loads((self.root / "config" / "serviceInstall.json").read_text(encoding="utf-8"))

    def run_command(self, args, cwd=None, env=None):
        """args 为命令数组；cwd/env 为可选运行目录与环境，失败立即终止。"""
        subprocess.run([str(arg) for arg in args], cwd=str(cwd or self.root),
                       env=env, check=True, shell=os.name == "nt" and str(args[0]).endswith(".cmd"))

    def clone_source(self, spec):
        """spec 为包含 repo/ref/directory 的固定源码清单；首次下载后保持版本不自动跟随主分支。"""
        target = self.root / spec["directory"]
        if not (target / ".git").is_dir():
            if not shutil.which("git"):
                raise RuntimeError("此服务首次安装需要 Git")
            if target.exists() and any(target.iterdir()):
                raise RuntimeError("目标目录非空，未覆盖：" + str(target))
            self.run_command(["git", "clone", "--no-checkout", spec["repo"], target])
            self.run_command(["git", "checkout", "--detach", spec["ref"]], target)
        actual = subprocess.check_output(["git", "-C", str(target), "rev-parse", "HEAD"], text=True).strip()
        if actual != spec["ref"]:
            raise RuntimeError("上游版本与安装清单不一致，请保留改动后重新安装")
        return target

    def prepare_python(self, name, spec, executable):
        """name/spec 为服务名与安装要求；executable 指定满足要求的 Python，创建独立环境。"""
        version = json.loads(subprocess.check_output(
            [executable, "-c", "import sys,json;print(json.dumps(list(sys.version_info[:2])))"], text=True))
        if tuple(version) < tuple(spec.get("python", [3, 9])):
            raise RuntimeError("此服务需要 Python " + ".".join(map(str, spec["python"])) + "，可用 --python 指定")
        home = self.root / ".serviceEnvs" / name
        python = home / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        if not python.is_file():
            self.run_command([executable, "-m", "venv", home])
        if spec.get("requirements"):
            self.run_command([python, "-m", "pip", "install", *spec["requirements"]])
        return python

    def install_llama(self):
        """无参数；Windows 从官方发行下载 CPU 推理引擎，核对发行 digest 并安全解压。"""
        directory = self.root / "vendor" / "llama"
        binary = directory / ("llama-server.exe" if os.name == "nt" else "llama-server")
        if binary.is_file():
            return binary
        existing = list(directory.rglob(binary.name)) if directory.exists() else []
        if existing:
            return existing[0]
        found = shutil.which("llama-server")
        if found:
            return Path(found)
        if os.name != "nt":
            raise RuntimeError("此平台请安装 llama-server 并加入 PATH")
        request = urllib.request.Request("https://api.github.com/repos/ggml-org/llama.cpp/releases/latest",
                                         headers={"User-Agent": "SynLove-ServiceInstaller"})
        with urllib.request.urlopen(request, timeout=30) as response:
            release = json.load(response)
        asset = next((item for item in release["assets"]
                      if item["name"].endswith("-bin-win-cpu-x64.zip")), None)
        if not asset:
            raise RuntimeError("官方发行未找到 Windows x64 CPU 包")
        directory.mkdir(parents=True, exist_ok=True)
        archive = directory / "engine.zip"
        with urllib.request.urlopen(asset["browser_download_url"], timeout=60) as response, archive.open("wb") as out:
            shutil.copyfileobj(response, out)
        digest = asset.get("digest", "")
        if not digest.startswith("sha256:"):
            raise RuntimeError("官方发行未提供 SHA-256，未解压未知二进制")
        if hashlib.sha256(archive.read_bytes()).hexdigest() != digest[7:]:
            raise RuntimeError("推理引擎下载校验失败")
        with zipfile.ZipFile(archive) as bundle:
            for info in bundle.infolist():
                if not (directory / info.filename).resolve().is_relative_to(directory.resolve()):
                    raise RuntimeError("推理引擎压缩包路径越界")
            bundle.extractall(directory)
        candidates = list(directory.rglob("llama-server.exe"))
        if not candidates:
            raise RuntimeError("推理引擎未生成 llama-server.exe")
        return candidates[0]

    def install_service(self, name, spec, executable, download=False):
        """name/spec 为选定服务；安装依赖，download 为用户显式授权下载模型。"""
        python = self.prepare_python(name, spec, executable) if spec.get("python") else Path(executable)
        if spec.get("repo"):
            self.clone_source(spec)
        if name == "llm":
            self.install_llama()
        if name == "kugou":
            npm = shutil.which("npm")
            if not npm:
                raise RuntimeError("音乐服务需要 Node.js 和 npm")
            self.run_command([npm, "install", "--omit=dev", "--no-audit", "--no-fund"],
                             self.root / spec["directory"])
        if download and name == "tts":
            self.run_command([python, "-c",
                              "from huggingface_hub import snapshot_download;"
                              "snapshot_download(repo_id='BreezeBlue/breeze-tts-2',"
                              "local_dir=" + repr(str(self.root / "models" / "breezeTts2")) + ")"])
        if download and name == "llm":
            self.run_command([python, self.root / spec["downloadScript"]])
        return python

    def start_service(self, name, spec, python):
        """name/spec/python 为服务和解释器；前台启动，模型未准备时明确报错。"""
        env = dict(os.environ)
        env["PYTHONUTF8"] = "1"
        env["HF_HOME"] = str(self.root / "models" / "cache")
        if name == "kugou":
            env["PORT"] = str(spec["port"])
            env["HOST"] = "127.0.0.1"
            self.run_command([shutil.which("node") or "node", "app.js"], self.root / spec["directory"], env)
            return
        args = [python, self.root / spec["script"], "--port", str(spec["port"])]
        if name == "tts":
            ckpt = self.root / "models" / "breezeTts2"
            if not (ckpt / "config.json").is_file():
                raise RuntimeError("TTS 权重未准备，请加 --download-model 安装")
            env["BREEZE_REPO"] = str(self.root / spec["directory"])
            args.extend(["--ckpt", ckpt])
        if name == "llm":
            args.extend(["--bin", self.install_llama(), "--model",
                         self.root / spec["model"], "--ngl", "0", "--min-free-vram", "0"])
        self.run_command(args, env=env)

    def run_cli(self):
        """无参数；默认安装并启动，--plan 仅显示计划，--install-only 不启动服务。"""
        parser = argparse.ArgumentParser(description="SynLove 独立服务")
        parser.add_argument("service", nargs="?", default=self.config.get("defaultService"),
                            choices=list(self.config["services"]))
        parser.add_argument("--plan", action="store_true")
        parser.add_argument("--install-only", action="store_true")
        parser.add_argument("--download-model", action="store_true")
        parser.add_argument("--python", default=sys.executable)
        args = parser.parse_args()
        if not args.service:
            parser.error("请选择服务：asr、tts、llm 或 kugou")
        spec = self.config["services"][args.service]
        if args.plan:
            print(json.dumps({"service": args.service, "installation": spec}, ensure_ascii=False, indent=2))
            return 0
        try:
            python = self.install_service(args.service, spec, args.python, args.download_model)
            if not args.install_only:
                self.start_service(args.service, spec, python)
            return 0
        except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
            print("服务启动失败：" + str(error))
            return 1
