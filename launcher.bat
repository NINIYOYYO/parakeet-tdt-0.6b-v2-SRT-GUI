@echo off

echo ���ڳ��Լ������⻷�� .venv...

REM ������⻷���Ƿ����
if not exist .\.venv\Scripts\activate.bat (
    echo �������⻷�� .venv δ�ҵ���������
    echo �������� install_dependencies.bat �����������û�����
    pause
    exit /b
)

call .\.venv\Scripts\activate.bat
echo ���⻷���Ѽ��

echo.
echo ��������Ӧ�ó��� (main.py)...
echo �������û��������ʾ���棬��������������Ƿ��д�����Ϣ��
echo Gradio ͨ�����ӡһ������ URL (���� http://127.0.0.1:7860)��
echo ����ҳ������http://127.0.0.1:7860���ɽ������
echo ����������ʱ���ܽϳ�
echo.

python main.py

echo.
echo Ӧ�ó����ѹرջ��ѽ�����
pause