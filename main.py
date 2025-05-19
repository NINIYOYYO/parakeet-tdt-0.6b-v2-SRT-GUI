import nemo.collections.asr as nemo_asr
import torch
import os
import subprocess
import time
from pydub import AudioSegment
import gradio as gr
import tempfile
import json

CONFIG_FILENAME = "config.json"
asr_model = None  # 初始化模型变量
device = None     # 初始化设备变量

# --- 配置管理 ---
def get_config_file_path():
    """获取脚本目录中配置文件的绝对路径。"""
    try:
        # 找到脚本目录
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError: # 如果在交互式环境中__file__ 未定义
        script_dir = os.getcwd()
    return os.path.join(script_dir, CONFIG_FILENAME)

def save_config(local_model_path: str, chunk_length: int):
    """将当前配置保存到 config.json。"""
    config = {
        "local_model_path": local_model_path, # NGC 为空字符串，本地为路径，如果从未选择则为 None
        "chunk_length_s": chunk_length
    }
    config_file_path = get_config_file_path()
    try:
        with open(config_file_path, "w", encoding='utf-8') as config_file:
            json.dump(config, config_file, indent=4)
        print(f"配置已保存到 {config_file_path}")
    except Exception as e:
        print(f"错误：保存配置文件 '{config_file_path}' 失败: {e}")

def load_config() -> dict:
    """从 config.json 加载配置。
    返回一个包含 'local_model_path' (可以为 None) 和 'chunk_length_s' 的字典。
    """
    config_file_path = get_config_file_path()
    default_config = {"local_model_path": None, "chunk_length_s": 60} # None 表示尚未做出选择

    if os.path.exists(config_file_path):
        try:
            with open(config_file_path, "r", encoding='utf-8') as config_file:
                loaded_config = json.load(config_file)
                # 确保基本键存在，如果不存在则提供默认值
                loaded_config.setdefault("local_model_path", None)
                loaded_config.setdefault("chunk_length_s", 60)
                print(f"配置已从 {config_file_path} 加载: {loaded_config}")
                return loaded_config
        except json.JSONDecodeError:
            print(f"错误: 配置文件 {config_file_path} 格式错误。使用默认配置。")
            if os.path.exists(config_file_path): # 可选：备份损坏的配置文件
                try:
                    os.rename(config_file_path, config_file_path + ".corrupted")
                    print(f"已备份损坏的配置文件为 {config_file_path}.corrupted")
                except Exception as e_mv:
                    print(f"备份损坏的配置文件失败: {e_mv}")
            return default_config
        except Exception as e:
            print(f"加载配置文件 '{config_file_path}' 时发生未知错误: {e}。使用默认配置。")
            return default_config
    else:
        print(f"配置文件 {config_file_path} 未找到。将使用默认设置 (首次运行)。")
        return default_config

# --- 模型加载 ---
def load_asr_model_globally(
    local_model_path_to_try: str = None,
    load_from_ngc_explicitly: bool = False,
    save_choice_on_success: bool = False,
    current_chunk_value: int = 60
) -> str:
    global asr_model, device

    print("正在尝试加载 ASR 模型...")

    if torch.cuda.is_available():
        current_device = torch.device("cuda")
        print("检测到 CUDA GPU，将在 GPU 上运行。")
    else:
        current_device = torch.device("cpu")
        print("警告：未检测到 CUDA GPU，将在 CPU 上运行，速度可能较慢。")
    device = current_device
    asr_model = None # 在尝试加载前重置模型状态

    # 场景1：明确从NGC加载
    if load_from_ngc_explicitly:
        print("尝试从云端NVIDIA NGC加载模型 'nvidia/parakeet-tdt-0.6b-v2'...")
        try:
            asr_model = nemo_asr.models.ASRModel.from_pretrained(
                model_name="nvidia/parakeet-tdt-0.6b-v2", map_location=device
            )
            status_msg = "云端模型 'nvidia/parakeet-tdt-0.6b-v2' 加载成功。"
            print(status_msg)
            if save_choice_on_success:
                save_config(local_model_path="", chunk_length=current_chunk_value) # NGC使用空路径
            return status_msg
        except Exception as e:
            asr_model = None
            status_msg = f"从NGC加载云端模型失败: {e}"
            print(status_msg)
            return status_msg

    # 场景2：尝试从 local_model_path_to_try 加载
    if local_model_path_to_try and local_model_path_to_try.strip():
        actual_path = local_model_path_to_try.strip()
        if not os.path.exists(actual_path):
            status_msg = f"错误：指定的本地模型路径不存在: {actual_path}"
            print(status_msg)
            return status_msg # asr_model 已经是 None
        if not actual_path.endswith(".nemo"):
            status_msg = f"错误：指定的本地模型路径不是有效的 .nemo 文件: {actual_path}"
            print(status_msg)
            return status_msg # asr_model 已经是 None

        print(f"尝试从本地路径加载模型: {actual_path}...")
        try:
            asr_model = nemo_asr.models.ASRModel.restore_from(
                restore_path=actual_path, map_location=device
            )
            model_name = os.path.basename(actual_path)
            status_msg = f"本地模型 '{model_name}' 加载成功。"
            print(status_msg)
            if save_choice_on_success:
                save_config(local_model_path=actual_path, chunk_length=current_chunk_value)
            return status_msg
        except Exception as e:
            asr_model = None
            status_msg = f"从本地路径 '{actual_path}' 加载模型失败: {e}"
            print(status_msg)
            return status_msg

    # 场景3：没有有效的或采取的特定加载操作
    status_msg = "模型未加载。请提供有效的本地模型路径或选择从云端加载。"
    # 如果既未处理显式NGC加载也未处理本地路径，则此消息更为通用。
    # 如果提供了 local_model_path_to_try 但导致错误，则上面已返回特定错误。
    if not load_from_ngc_explicitly and not (local_model_path_to_try and local_model_path_to_try.strip()):
         print(status_msg) # 记录未采取任何操作的此特定情况

    return status_msg


# --- 辅助函数 (ffmpeg, 音频处理, SRT 生成) ---
def check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("错误：ffmpeg 未检测到或未正确安装。请安装 ffmpeg 并确保其在系统 PATH 中。")
        return False

def extract_audio_from_video(input_video_path: str) -> str:
    if not check_ffmpeg(): return None
    temp_audio_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    output_audio_path = temp_audio_file.name
    temp_audio_file.close()

    ffmpeg_command = [
        'ffmpeg', '-i', input_video_path, '-vn', '-acodec', 'pcm_s16le',
        '-ar', '16000', '-ac', '1', '-y', output_audio_path
    ]
    try:
        process = subprocess.run(ffmpeg_command, check=True, capture_output=True, text=True, errors='ignore')
        print(f"音频提取成功到: {output_audio_path}")
        if os.path.exists(output_audio_path) and os.path.getsize(output_audio_path) > 0:
            return output_audio_path
        else:
            print(f"错误：ffmpeg 命令成功执行，但未生成音频文件：{output_audio_path}")
            if os.path.exists(output_audio_path): os.remove(output_audio_path) # 清理空文件
            return None
    except subprocess.CalledProcessError as e:
        print(f"提取音频时发生 FFmpeg 错误：{e}\n标准输出: {e.stdout}\n标准错误: {e.stderr}")
        if os.path.exists(output_audio_path): os.remove(output_audio_path)
        return None
    except FileNotFoundError:
        print("错误：未找到 FFmpeg 可执行文件。")
        return None


def preprocess_direct_audio(input_audio_path: str) -> str:
    if not check_ffmpeg(): return None
    if not input_audio_path or not os.path.exists(input_audio_path):
        print(f"错误：提供的音频文件路径无效或文件不存在: {input_audio_path}")
        return None

    temp_processed_audio_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    output_wav_path = temp_processed_audio_file.name
    temp_processed_audio_file.close()

    ffmpeg_command = [
        'ffmpeg', '-i', input_audio_path, '-acodec', 'pcm_s16le',
        '-ar', '16000', '-ac', '1', '-y', output_wav_path
    ]
    try:
        process = subprocess.run(ffmpeg_command, check=True, capture_output=True, text=True, errors='ignore')
        print(f"直接音频预处理成功: {input_audio_path} -> {output_wav_path}")
        if os.path.exists(output_wav_path) and os.path.getsize(output_wav_path) > 0:
            return output_wav_path
        else:
            print(f"错误：ffmpeg 命令执行，但未生成有效音频文件：{output_wav_path}")
            if os.path.exists(output_wav_path): os.remove(output_wav_path) # 清理
            return None
    except subprocess.CalledProcessError as e:
        print(f"预处理直接音频 '{input_audio_path}' 时发生 FFmpeg 错误：{e}\n标准输出: {e.stdout}\n标准错误: {e.stderr}")
        if os.path.exists(output_wav_path): os.remove(output_wav_path)
        return None
    return None


def format_srt_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int((seconds * 1000) % 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{milliseconds:03}"

def transcribe_audio_in_chunks(model, audio_path: str, chunk_length_ms: int) -> list:
    if model is None:
        print("模型未加载，无法进行转录。")
        return []
    if not audio_path or not os.path.exists(audio_path):
        print(f"错误: 音频文件路径 '{audio_path}' 无效或文件不存在。")
        return []

    print(f"正在加载音频文件 '{audio_path}' 进行分块处理...")
    try:
        audio = AudioSegment.from_wav(audio_path) # 假设已预处理为 WAV
        audio = audio.set_frame_rate(16000).set_channels(1) # 确保格式
    except Exception as e:
        print(f"加载或处理音频文件 '{audio_path}' 时发生错误 (pydub): {e}")
        return []

    audio_duration_ms = len(audio)
    print(f"音频总时长: {audio_duration_ms / 1000:.2f} 秒")
    all_segment_timestamps = []

    for i in range(0, audio_duration_ms, chunk_length_ms):
        start_time_ms = i
        end_time_ms = min(i + chunk_length_ms, audio_duration_ms)
        chunk = audio[start_time_ms:end_time_ms]
        
        temp_chunk_file_path = "" # 在 try 块外部定义
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_chunk_file:
                temp_chunk_file_path = temp_chunk_file.name
            chunk.export(temp_chunk_file_path, format="wav")
            print(f"处理音频块: {start_time_ms / 1000:.2f}s - {end_time_ms / 1000:.2f}s")

            chunk_output_list = model.transcribe([temp_chunk_file_path], batch_size=1, timestamps=True)

            if chunk_output_list and hasattr(chunk_output_list[0], 'timestamp') and \
               chunk_output_list[0].timestamp and 'segment' in chunk_output_list[0].timestamp:
                current_chunk_segments = chunk_output_list[0].timestamp['segment']
                chunk_global_start_offset_sec = start_time_ms / 1000.0
                for segment_data in current_chunk_segments:
                    local_start_sec = segment_data['start']
                    local_end_sec = segment_data['end']
                    text_content = segment_data.get('segment', segment_data.get('text', ''))
                    global_start_sec = local_start_sec + chunk_global_start_offset_sec
                    global_end_sec = local_end_sec + chunk_global_start_offset_sec
                    if global_end_sec < global_start_sec: # 安全检查
                        global_end_sec = global_start_sec + 0.05
                    all_segment_timestamps.append({
                        'start': global_start_sec, 'end': global_end_sec, 'segment': text_content
                    })
            else:
                full_text = chunk_output_list[0].text if chunk_output_list else "N/A"
                print(f"警告: 音频块未能生成分段时间戳。完整转录: '{full_text}'.")
        except Exception as e:
            print(f"转录音频块 '{temp_chunk_file_path}' 时发生错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if temp_chunk_file_path and os.path.exists(temp_chunk_file_path):
                try:
                    os.remove(temp_chunk_file_path)
                except OSError as e_os:
                    print(f"删除临时音频文件 '{temp_chunk_file_path}' 时发生OS错误: {e_os}")
    
    all_segment_timestamps.sort(key=lambda x: x['start'])
    return all_segment_timestamps

def generate_srt_content(segment_timestamps: list) -> str:
    srt_content = ""
    for i, stamp in enumerate(segment_timestamps):
        subtitle_number = i + 1
        start_time_srt = format_srt_time(stamp['start'])
        end_time_srt = format_srt_time(stamp['end'])
        segment_text = stamp['segment']
        srt_block = f"{subtitle_number}\n{start_time_srt} --> {end_time_srt}\n{segment_text}\n\n"
        srt_content += srt_block
    return srt_content

# --- Gradio 处理函数 ---
def process_video_for_srt(video_file_obj, chunk_length_s: int):
    if asr_model is None:
        yield "错误：ASR 模型未加载。请先加载模型。", None, ""
        return
    if video_file_obj is None:
        yield "请上传一个视频文件。", None, ""
        return
    
    input_video_path = video_file_obj # Gradio Video 对象具有 .name 属性表示路径
    print(f"开始处理视频文件: {input_video_path}")
    start_time_total = time.time()
    extracted_audio_path = None
    output_srt_path_for_download = None # 用于 Gradio File 组件

    try:
        yield "状态：正在提取音频...", None, ""
        extracted_audio_path = extract_audio_from_video(input_video_path)
        if not extracted_audio_path:
            yield "错误：音频提取失败。请检查视频文件或ffmpeg安装。", None, ""
            return

        chunk_length_ms = chunk_length_s * 1000
        yield f"状态：正在转录音频 (分块大小: {chunk_length_s}秒)...", None, ""
        segment_timestamps = transcribe_audio_in_chunks(asr_model, extracted_audio_path, chunk_length_ms)

        if not segment_timestamps:
            yield "错误：转录未生成有效的分段时间戳。", None, ""
            return

        yield "状态：正在生成 SRT 内容...", None, ""
        srt_content = generate_srt_content(segment_timestamps)

        with tempfile.NamedTemporaryFile(mode="w", encoding='utf-8', suffix=".srt", delete=False) as tmp_srt_file:
            tmp_srt_file.write(srt_content)
            output_srt_path_for_download = tmp_srt_file.name
        
        elapsed_time_total = time.time() - start_time_total
        status_message = f"处理完成。总耗时 {elapsed_time_total:.2f} 秒。"
        print(f"{status_message} SRT 文件位于: {output_srt_path_for_download}")
        yield status_message, output_srt_path_for_download, srt_content

    except Exception as e:
        import traceback
        traceback.print_exc()
        yield f"处理过程中发生未知错误: {e}", None, ""
    finally:
        if extracted_audio_path and os.path.exists(extracted_audio_path):
            try: os.remove(extracted_audio_path)
            except OSError as e_clean: print(f"清理临时音频文件 {extracted_audio_path} 时出错: {e_clean}")
        # Gradio 会处理 output_srt_path_for_download（它提供的临时文件）的删除

def process_audio_for_srt(audio_file_path_str, chunk_length_s: int): # 音频输入类型："filepath"
    if asr_model is None:
        yield "错误：ASR 模型未加载。请先加载模型。", None, ""
        return
    if not audio_file_path_str:
        yield "请上传一个音频文件。", None, ""
        return

    print(f"开始处理音频文件: {audio_file_path_str}")
    start_time_total = time.time()
    processed_audio_path = None
    output_srt_path_for_download = None

    try:
        if not check_ffmpeg(): # 再次检查，尽管初始检查是好的
            yield "错误: FFMPEG 未安装或未在 PATH 中，无法处理音频。", None, ""
            return

        yield "状态：正在预处理音频文件...", None, ""
        processed_audio_path = preprocess_direct_audio(audio_file_path_str)
        if not processed_audio_path:
            yield "错误：音频预处理失败。", None, ""
            return
        
        chunk_length_ms = chunk_length_s * 1000
        yield f"状态：准备转录音频 (分块大小: {chunk_length_s}秒)...", None, ""
        segment_timestamps = transcribe_audio_in_chunks(asr_model, processed_audio_path, chunk_length_ms)

        if not segment_timestamps:
            yield "错误：转录未生成有效的分段时间戳。", None, ""
            return

        yield "状态：正在生成 SRT 内容...", None, ""
        srt_content = generate_srt_content(segment_timestamps)

        with tempfile.NamedTemporaryFile(mode="w", encoding='utf-8', suffix=".srt", delete=False) as tmp_srt_file:
            tmp_srt_file.write(srt_content)
            output_srt_path_for_download = tmp_srt_file.name
            
        elapsed_time_total = time.time() - start_time_total
        status_message = f"音频处理完成。总耗时 {elapsed_time_total:.2f} 秒。"
        print(f"{status_message} SRT 文件位于: {output_srt_path_for_download}")
        yield status_message, output_srt_path_for_download, srt_content

    except Exception as e:
        import traceback
        traceback.print_exc()
        yield f"处理音频过程中发生未知错误: {e}", None, ""
    finally:
        if processed_audio_path and os.path.exists(processed_audio_path) and processed_audio_path != audio_file_path_str:
            try: os.remove(processed_audio_path)
            except OSError as e_clean: print(f"清理预处理的音频文件 {processed_audio_path} 时出错: {e_clean}")


# --- Gradio 界面设置 ---
if __name__ == "__main__":
    if not check_ffmpeg():
        print("重要提示: FFMPEG 未找到。视频和音频处理功能将受限或无法工作。")

    # --- 启动时模型加载逻辑 ---
    config = load_config()
    # saved_model_path 可以是：文件路径 (str)，"" (表示NGC)，或 None (之前未选择)
    saved_model_path = config.get("local_model_path")
    initial_chunk_length = config.get("chunk_length_s", 60)
    initial_model_status = "模型未加载。请选择本地模型或从云端加载。" # 真正首次运行时的默认值

    if saved_model_path is not None: # 之前已做出选择
        if saved_model_path == "": # 上次选择的是 NGC
            print("配置中记录上次选择为云端NGC模型，尝试自动重新加载...")
            initial_model_status = load_asr_model_globally(
                load_from_ngc_explicitly=True,
                save_choice_on_success=False, # 自动加载时不重新保存配置
                current_chunk_value=initial_chunk_length
            )
        elif os.path.exists(saved_model_path): # 保存了本地路径且该路径存在
            print(f"配置中记录本地模型路径: {saved_model_path}，尝试自动重新加载...")
            initial_model_status = load_asr_model_globally(
                local_model_path_to_try=saved_model_path,
                save_choice_on_success=False, # 自动加载时不重新保存配置
                current_chunk_value=initial_chunk_length
            )
        else: # 保存了本地路径，但该路径已不存在
            error_msg = f"错误：配置文件中的本地模型路径 '{saved_model_path}' 未找到或无效。模型未加载。"
            print(error_msg)
            initial_model_status = error_msg

    else:
        # 这是真正的首次运行（配置文件不存在或 local_model_path 明确为 None）
        print("首次运行或之前未配置模型。模型不会自动加载，请手动选择。")
        # initial_model_status 保持默认的 "模型未加载..."

    # --- Gradio UI 定义 ---
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 视频/音频自动生成 SRT 字幕工具 (NeMo Parakeet)")
        gr.Markdown("上传视频或音频文件，自动生成SRT字幕。基于nvidia/parakeet-tdt-0.6b-v2模型。")

        with gr.Row():
            with gr.Column(scale=3):
                # 如果 saved_model_path 为 None (首次运行) 或 "" (NGC)，则用 "" 初始化文本框
                local_model_path_input = gr.Textbox(
                    label="本地模型路径 (.nemo 文件)",
                    placeholder="例如: /path/to/your_model.nemo",
                    value=saved_model_path if saved_model_path is not None else ""
                )
            with gr.Column(scale=1, min_width=150):
                load_local_model_button = gr.Button("加载本地模型", variant="secondary")
            with gr.Column(scale=1, min_width=150):
                load_cloud_model_button = gr.Button("加载云端模型", variant="primary")
        
        model_status_output = gr.Textbox(
            label="模型加载状态",
            value=initial_model_status,
            lines=2,
            interactive=False,
            max_lines=3
        )

        chunk_slider = gr.Slider(
            minimum=10, maximum=300, value=initial_chunk_length, step=5,
            label="音频分块长度 (秒)",
            info="推荐60-180秒。更改后，下次点击任一“加载模型”按钮时，此设置会与模型选择一同保存。"
        )
        gr.Markdown("---")

        with gr.Tab("从视频生成字幕"):
            video_input = gr.Video(label="上传视频文件 (例如 MP4, MKV)")
            video_submit_button = gr.Button("开始从视频生成 SRT", variant="primary")
        
        with gr.Tab("从音频生成字幕"):
            # 对于音频，type="filepath" 对于 ffmpeg 处理通常更健壮
            audio_input = gr.Audio(label="上传音频文件 (例如 MP3, WAV, M4A)", type="filepath")
            audio_submit_button = gr.Button("开始从音频生成 SRT", variant="primary")

        status_output = gr.Textbox(label="处理状态", lines=1, interactive=False)
        with gr.Accordion("SRT字幕结果", open=True):
            srt_file_output = gr.File(label="下载 SRT 文件 (.srt)", interactive=False)
            srt_preview_output = gr.Textbox(label="SRT 内容预览", lines=10, max_lines=20, interactive=False)

        # --- 按钮点击处理程序 ---
        def handle_load_local_click(path_from_input_box, chunk_val_from_slider):
            if not path_from_input_box or not path_from_input_box.strip():
                return "错误：请输入有效的本地模型路径后点击“加载本地模型”。若要加载云端模型，请使用对应按钮。"
            return load_asr_model_globally(
                local_model_path_to_try=path_from_input_box,
                load_from_ngc_explicitly=False,
                save_choice_on_success=True,
                current_chunk_value=chunk_val_from_slider
            )

        def handle_load_cloud_click(chunk_val_from_slider):
            return load_asr_model_globally(
                local_model_path_to_try=None,
                load_from_ngc_explicitly=True,
                save_choice_on_success=True,
                current_chunk_value=chunk_val_from_slider
            )

        load_local_model_button.click(
            fn=handle_load_local_click,
            inputs=[local_model_path_input, chunk_slider],
            outputs=[model_status_output]
        )
        load_cloud_model_button.click(
            fn=handle_load_cloud_click,
            inputs=[chunk_slider], # 仅需要 chunk_slider 来保存配置
            outputs=[model_status_output]
        )

        video_submit_button.click(
            fn=process_video_for_srt,
            inputs=[video_input, chunk_slider],
            outputs=[status_output, srt_file_output, srt_preview_output]
        )
        audio_submit_button.click(
            fn=process_audio_for_srt,
            inputs=[audio_input, chunk_slider],
            outputs=[status_output, srt_file_output, srt_preview_output]
        )
        
        gr.Markdown("---")
        gr.Markdown("注意: 处理速度取决于您的硬件 (GPU/CPU) 和文件大小。")
        if device and device.type == 'cpu': # 检查设备是否已初始化
            gr.Markdown("️️️⚠️ **警告：当前正在使用CPU运行，速度会非常慢。建议使用CUDA GPU以获得更好性能。**")
        elif not device and torch.cuda.is_available(): # 模型尚未加载，但 GPU 可用
             gr.Markdown("️️️ℹ️ **提示：检测到CUDA GPU。选择模型后将尝试在GPU上运行。**")
        elif not device: # 模型未加载，且无 GPU
             gr.Markdown("️️️⚠️ **警告：未检测到CUDA GPU。选择模型后将在CPU上运行，速度可能较慢。**")

    print("Gradio 界面即将启动...")
    demo.launch()
    print("Gradio 界面已停止。")