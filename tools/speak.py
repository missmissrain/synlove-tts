"""
TTS 命令行：按 config/serviceConfig.json 调用本机或远程服务，可把语音直接发到 QQ。

用法：
    python tools\\speak.py "你好，我是欣爱。" --instruction "温柔自信的年轻女性" --cfg-scale 4
    ... --send qq                 # 合成后转 mp3 并作为 QQ 语音条发送
    ... --max-chars 80            # 每段更短（更快、更稳）

行为：
- managed 模式按需启动本机服务；external 模式只访问配置的地址；
- 长文本自动按句切段合成再拼接，避免显存/质量一起崩；
- 音频下载到本机 runtime/tts；托管服务空闲退出释放显存。
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from services.ServiceRegistry import ServiceRegistry
from services.ServiceClient import ServiceClient
from services.LocalServiceLifecycle import LocalServiceLifecycle
LOG = ROOT / "runtime" / "tts-service.log"
SECRETS = ROOT / "config" / "secrets.json"
NEWS_CONFIG = ROOT / "newsDigest" / "config" / "newsDigest.json"
FFMPEG = os.environ.get("FFMPEG") or "ffmpeg"

VOICES = {
    "tiantian": "十七八岁的甜妹，声音软糯清甜，语速稍快，尾音轻轻上扬，带一点撒娇感，普通话标准。",
    "tiantian2": "二十岁出头的少女，嗓音清亮甜美，说话轻快俏皮，尾音带笑，像在跟喜欢的人聊天。",
    "yuanqi": "元气满满的少女，声音明亮有活力，语速轻快，情绪饱满，吐字干净利落。",
    "wenrou": "温柔自信的年轻女性，声音清晰，语气亲切，表达轻快而富有感染力。",
    "zhixing": "二十五六岁的知性女性，声音沉稳温柔，语速中慢，吐字清晰，不疾不徐。",
}




def wav_to_mp3(wav_path):
    """wav → mp3（QQ 语音条用 mp3 实测可正常显示）。"""
    mp3_path = str(Path(wav_path).with_suffix(".mp3"))
    result = subprocess.run(
        [FFMPEG, "-y", "-hide_banner", "-loglevel", "error", "-i", wav_path,
         "-codec:a", "libmp3lame", "-b:a", "64k", mp3_path],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if result.returncode != 0 or not Path(mp3_path).is_file():
        raise RuntimeError(f"ffmpeg 转 mp3 失败：{(result.stderr or '')[:200]}")
    return mp3_path


def qq_owner_openid():
    """读 QQ 收件人 openid（优先 newsDigest 配置，其次环境变量）。"""
    try:
        data = json.loads(NEWS_CONFIG.read_text(encoding="utf-8-sig"))
        openid = str((data.get("push") or {}).get("userOpenid") or "").strip()
        if openid:
            return openid
    except (OSError, ValueError):
        pass
    return os.environ.get("QQ_OWNER_OPENID", "").strip()


def send_qq_voice(path):
    """把本地音频作为 QQ 语音条发出去（主动消息）。返回发送结果 dict。"""
    openid = qq_owner_openid()
    if openid == "":
        raise RuntimeError("找不到 QQ 收件人 openid（newsDigest.json 的 push.userOpenid）")
    secrets = json.loads(SECRETS.read_text(encoding="utf-8-sig"))
    qq = secrets.get("qq") or {}
    if not qq.get("appId") or not qq.get("clientSecret"):
        raise RuntimeError("secrets.json 缺 qq.appId / qq.clientSecret")

    token_request = urllib.request.Request(
        "https://api.bot.qq.com/app/getAppAccessToken",
        data=json.dumps({"appId": str(qq["appId"]), "clientSecret": str(qq["clientSecret"])}).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(token_request, timeout=20) as response:
        token = json.loads(response.read().decode("utf-8")).get("access_token")
    if not token:
        raise RuntimeError("获取 QQ access_token 失败")

    import base64

    payload = {
        "file_type": 3,
        "file_data": base64.b64encode(Path(path).read_bytes()).decode("ascii"),
        "srv_send_msg": False,
    }
    upload = urllib.request.Request(
        f"https://api.bot.qq.com/v2/users/{openid}/files",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"QQBot {token}"}, method="POST",
    )
    with urllib.request.urlopen(upload, timeout=60) as response:
        file_info = json.loads(response.read().decode("utf-8")).get("file_info")
    if not file_info:
        raise RuntimeError("QQ 上传没有返回 file_info")

    send = urllib.request.Request(
        f"https://api.bot.qq.com/v2/users/{openid}/messages",
        data=json.dumps({"content": " ", "msg_type": 7, "media": {"file_info": file_info}}).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"QQBot {token}"}, method="POST",
    )
    with urllib.request.urlopen(send, timeout=30) as response:
        return json.loads(response.read().decode("utf-8") or "{}")


def main(argv):
    """入口。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("text")
    parser.add_argument("--voice", default="tiantian2", help="音色预设（默认 tiantian2）：tiantian/tiantian2/yuanqi/wenrou/zhixing")
    parser.add_argument("--instruction", default=None, help="自定义音色描述（优先于 --voice）")
    parser.add_argument("--ref-audio", default=None)
    parser.add_argument("--ref-text", default=None)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-chars", type=int, default=120, help="每段最大字数")
    parser.add_argument("--idle-seconds", type=int, default=300, help="兜底：万一没退出，闲置多久强制退")
    parser.add_argument("--send", choices=["qq"], default=None, help="合成后发到 QQ（语音条）")
    args = parser.parse_args(argv)

    instruction = args.instruction or VOICES.get(str(args.voice or "")) or VOICES["tiantian2"]
    options = {
        "instruction": instruction,
        "refText": args.ref_text,
        "cfgScale": args.cfg_scale,
        "seed": args.seed,
        "maxChars": args.max_chars,
    }
    started = time.time()
    try:
        cfg = ServiceRegistry().get_service("tts")
        if args.instruction and cfg.get("dialect") != "breeze":
            raise ValueError("--instruction 为 Breeze 扩展；标准服务请在 serviceConfig.json 配置 voice")
        if args.ref_audio:
            import base64
            ref = Path(args.ref_audio)
            if ref.stat().st_size > 2 * 1024 * 1024:
                raise ValueError("参考音频不能超过 2 MiB")
            options["refAudioBase64"] = base64.b64encode(ref.read_bytes()).decode("ascii")
        LocalServiceLifecycle("tts", cfg).ensure_service(args.idle_seconds, args.max_chars)
        audio, fmt = ServiceClient(cfg).synthesize(args.text, **options)
        import uuid
        out = ROOT / "runtime" / "tts"
        out.mkdir(parents=True, exist_ok=True)
        if fmt not in ("wav", "mp3", "opus", "aac", "flac", "pcm"):
            raise ValueError("不支持的音频输出格式")
        wav_path = str(out / (uuid.uuid4().hex + "." + fmt))
        Path(wav_path).write_bytes(audio)
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", "replace")
        print(f"合成失败 HTTP {error.code}: {body[:600]}")
        return 1
    except (urllib.error.URLError, TimeoutError, OSError, RuntimeError, ValueError) as error:
        print(f"合成失败：{error}（日志 {LOG}）")
        return 1

    print(f"用时 {time.time() - started:.1f}s  音频 {len(audio)} 字节")
    print(wav_path)

    if args.send == "qq" and wav_path:
        try:
            mp3 = wav_path if Path(wav_path).suffix.lower() == ".mp3" else wav_to_mp3(wav_path)
        except RuntimeError as error:
            print(f"转 mp3 失败，改发 wav：{error}")
            mp3 = wav_path
        try:
            result = send_qq_voice(mp3)
            print("已发送到 QQ：" + str(result.get("id", "")))
        except Exception as error:  # noqa: BLE001
            print(f"发送到 QQ 失败：{type(error).__name__}: {error}")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
