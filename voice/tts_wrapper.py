# -*- coding: utf-8 -*-
"""
TTS包装器 - 解决asyncio事件循环冲突问题
"""

import asyncio
import threading
import logging
import tempfile

from voice.output.tts_handler import generate_speech_async

logger = logging.getLogger(__name__)


class TTSWrapper:
    """TTS包装器，处理asyncio和线程安全问题"""

    def __init__(self):
        self._loop = None
        self._thread = None
        self._start_loop()

    def _start_loop(self):
        """在独立线程中启动事件循环"""

        def run_loop():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_forever()

        self._thread = threading.Thread(target=run_loop, daemon=True)
        self._thread.start()

        # 等待事件循环启动
        while self._loop is None:
            threading.Event().wait(0.01)

    def generate_speech_safe(self, text, voice, response_format="mp3", speed=1.0, model=None):
        """线程安全的TTS生成"""
        try:
            future = asyncio.run_coroutine_threadsafe(
                generate_speech_async(text, voice, response_format, speed, model),
                self._loop,
            )
            result = future.result(timeout=60)
            return result
        except Exception as e:
            logger.error(f"[TTS包装器] 生成语音失败: {e}")
            return self._create_silent_file(response_format)

    def _create_silent_file(self, response_format):
        """创建静音文件作为回退"""
        logger.warning("[TTS包装器] 创建静音文件作为回退")
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{response_format}")
        temp_path = temp_file.name
        temp_file.close()
        return temp_path

    def cleanup(self):
        """清理资源"""
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)
            if self._thread:
                self._thread.join(timeout=1.0)


# 全局TTS包装器实例
_global_tts_wrapper = None


def get_tts_wrapper():
    """获取全局TTS包装器实例"""
    global _global_tts_wrapper
    if _global_tts_wrapper is None:
        _global_tts_wrapper = TTSWrapper()
    return _global_tts_wrapper


def generate_speech_safe(text, voice, response_format="mp3", speed=1.0, model=None):
    """安全的TTS生成函数（供外部调用）"""
    wrapper = get_tts_wrapper()
    return wrapper.generate_speech_safe(text, voice, response_format, speed, model)
