# SynLove TTS

把文字合成为语音的独立服务，基于 Breeze-TTS-2。支持分段合成、OpenAI 风格接口与闲置释放资源，不需要主项目或 Harness。

## 开始使用

准备 **Python 3.12+ 和 Git**，下载本仓库，运行：

```bat
python service.py --download-model
```

首次自动创建独立环境、下载清单固定版本的上游推理代码、安装依赖和模型，然后启动服务。模型较大，安装过程可能需要较长时间；确认 Python 命令指向 3.12+。

已经安装模型后，双击 **StartService.cmd** 或运行 `python service.py` 即可。只看计划加 `--plan`；只安装不启动加 `--install-only`。

默认地址为 `http://127.0.0.1:7862/v1`，合成接口为 `/v1/audio/speech`，健康检查为 `/health`。

接入 SynLove 时，在 `config/serviceConfig.json` 中填写地址，服务归属选择 `external`。服务也支持 `--host` 与 `--api-key-env`；远程监听需要配置鉴权。

## 文件位置

- `config/serviceInstall.json`：依赖、上游提交和端口。
- `vendor/breeze-tts/`：安装时下载的推理代码。
- `models/breezeTts2/`：安装时下载的模型。
- `runtime/tts/`：生成的音频。

运行需要足够的内存与显存；服务会在首次合成时加载模型。模型使用范围以上游许可为准。

上游：[Breeze-TTS](https://github.com/breezeblue-ai/breeze-tts) · [模型](https://huggingface.co/BreezeBlue/breeze-tts-2)。
主项目：[SynLove](https://github.com/missmissrain/SynLove-Personal-Agent)。
