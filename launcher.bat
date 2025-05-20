@echo off

echo 正在尝试激活虚拟环境 .venv...

REM 检查虚拟环境是否存在
if not exist .\.venv\Scripts\activate.bat (
    echo 错误：虚拟环境 .venv 未找到或不完整。
    echo 请先运行 install_dependencies.bat 来创建和配置环境。
    pause
    exit /b
)

call .\.venv\Scripts\activate.bat
echo 虚拟环境已激活。

echo.
echo 正在启动应用程序 (main.py)...
echo 如果程序没有立即显示界面，请检查命令行输出是否有错误信息。
echo Gradio 通常会打印一个本地 URL (例如 http://127.0.0.1:7860)。
echo 初次启动耗时可能较长
echo.

python main.py

echo.
echo 应用程序已关闭或已结束。
pause