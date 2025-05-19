@echo off
echo 开始环境配置和依赖安装...

REM 检查 Python 是否安装
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo 错误：未找到 Python。请先安装Python 3.12.2 或更高版本 并将其添加到 PATH。
    echo 您可以从 https://www.python.org/downloads/ 下载。
    pause
    exit /b
)
echo Python 已检测到。

REM 创建虚拟环境
echo 正在创建虚拟环境 .venv...
python -m venv .venv
if %errorlevel% neq 0 (
    echo 错误：创建虚拟环境失败。
    pause
    exit /b
)
echo 虚拟环境 .venv 创建成功。

REM 激活虚拟环境
echo 正在激活虚拟环境...
call .\.venv\Scripts\activate.bat
REM 在同一个脚本中，激活后，后续的 pip 和 python 命令就会在该虚拟环境内执行

REM 提示用户选择 PyTorch 版本
echo.
echo ===============================================================================
echo 关于 PyTorch 安装 (重要！)：
echo ===============================================================================
echo 如果您有 NVIDIA GPU 并希望使用 CUDA 加速，强烈建议您：
echo   1. 访问 PyTorch 官网 (https://pytorch.org/get-started/locally/)
echo   2. 获取适合您 CUDA 版本的 PyTorch 安装命令。
echo   3. 在【新的命令行窗口中先激活此虚拟环境(.venv\Scripts\activate)】，
echo      然后【手动执行】该 PyTorch 安装命令。
echo   4. 手动安装 PyTorch GPU 版本【之后】，再回到【此窗口】按 'n' 跳过自动安装。
echo.
echo 如果您不确定、只想使用 CPU，或者已手动安装 GPU 版 PyTorch，
echo 脚本可以尝试安装一个通用的 PyTorch (通常是CPU版)，或者跳过。
echo ===============================================================================
echo.

:pytorch_choice_prompt
set "install_pytorch_choice="
set /p install_pytorch_choice="您是否已手动安装 PyTorch GPU 版本？(y/n，输入 'y' 跳过自动安装，'n' 让脚本尝试安装): "

if /i "%install_pytorch_choice%"=="y" (
    echo 您选择了已手动安装或跳过 PyTorch 自动安装。
) else if /i "%install_pytorch_choice%"=="n" (
    echo 正在尝试安装 PyTorch (这可能是CPU版本，GPU用户请确保已手动安装GPU版)...
    pip install torch torchvision torchaudio --no-cache-dir
    if %errorlevel% neq 0 (
        echo 警告：PyTorch 自动安装失败或遇到问题。如果需要GPU支持，请务必手动安装。
        pause
    ) else (
        echo PyTorch 自动安装尝试完成。
    )
) else (
    echo 无效输入，请输入 y 或 n。
    goto pytorch_choice_prompt
)

REM 安装其他依赖
echo.
echo 正在安装其他依赖 (来自 requirements.txt)...
pip install -r requirements.txt --no-cache-dir
if %errorlevel% neq 0 (
    echo 错误：从 requirements.txt 安装依赖失败。
    echo 请检查 requirements.txt 文件是否存在且格式正确，以及您的网络连接。
    pause
    exit /b
)
echo 其他依赖安装完成。

echo.
echo ===============================================================================
echo 重要提示：FFmpeg 安装 (必需)
echo ===============================================================================
echo 本项目运行需要 FFmpeg。
echo 请确保您已从 https://ffmpeg.org/download.html (推荐 gyan.dev 构建)
echo 下载 FFmpeg，并将其中的 'bin' 目录路径添加到系统的 PATH 环境变量中。
echo.
echo 如何检查FFmpeg是否配置成功：
echo   打开一个新的命令行窗口，输入 'ffmpeg -version'，如果显示版本信息则表示成功。
echo ===============================================================================
echo.
echo 所有依赖项的安装流程已执行完毕。
echo.
echo 下一步：
echo   1. 如果您尚未配置 FFmpeg，请立即配置。
echo   2. 配置完成后，您可以双击运行 'launcher.bat' 来启动应用程序。
echo.
pause
exit /b