"""可选服务安装与启动入口；参数服务名、--plan、--install-only、--download-model、--python。"""
from services.ServiceInstaller import ServiceInstaller


if __name__ == "__main__":
    raise SystemExit(ServiceInstaller().run_cli())
