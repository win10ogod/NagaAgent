import sys
import os
import asyncio
import tempfile
from pathlib import Path
from typing import AsyncGenerator, Dict, List, Optional

sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # 加入项目根目录到模块查找路径

from openai import AsyncOpenAI

from system.config import config

_MIN_SPEAKING_RATE = 0.25
_MAX_SPEAKING_RATE = 4.0

# OpenAI TTS voice metadata
_OPENAI_TTS_VOICES: List[Dict[str, str]] = [
    {"id": "alloy", "name": "Alloy", "language": "en", "gender": "neutral"},
    {"id": "ash", "name": "Ash", "language": "en", "gender": "male"},
    {"id": "ballad", "name": "Ballad", "language": "en", "gender": "male"},
    {"id": "coral", "name": "Coral", "language": "en", "gender": "female"},
    {"id": "echo", "name": "Echo", "language": "en", "gender": "male"},
    {"id": "fable", "name": "Fable", "language": "en", "gender": "female"},
    {"id": "marin", "name": "Marin", "language": "en", "gender": "female"},
    {"id": "nova", "name": "Nova", "language": "en", "gender": "female"},
    {"id": "onyx", "name": "Onyx", "language": "en", "gender": "male"},
    {"id": "sage", "name": "Sage", "language": "en", "gender": "neutral"},
    {"id": "shimmer", "name": "Shimmer", "language": "en", "gender": "female"},
    {"id": "verse", "name": "Verse", "language": "en", "gender": "male"},
    {"id": "cedar", "name": "Cedar", "language": "en", "gender": "neutral"},
]

_OPENAI_VOICE_IDS = {voice_info["id"] for voice_info in _OPENAI_TTS_VOICES}

_OPENAI_TTS_MODELS: List[Dict[str, str]] = [
    {"id": "gpt-4o-mini-tts", "name": "GPT-4o mini TTS"},
    {"id": "tts-1", "name": "Text-to-Speech v1"},
    {"id": "tts-1-hd", "name": "Text-to-Speech v1 HD"},
]


def _build_tts_client() -> AsyncOpenAI:
    tts_config = config.tts
    api_key = tts_config.api_key or config.api.api_key
    base_url = getattr(tts_config, "base_url", "") or config.api.base_url
    return AsyncOpenAI(api_key=api_key, base_url=base_url)


_client = _build_tts_client()


def _resolve_model(model: Optional[str]) -> str:
    configured = getattr(config.tts, "model", None)
    fallback = "gpt-4o-mini-tts"
    return model or configured or fallback


def _resolve_voice(voice: Optional[str]) -> str:
    candidate = voice or config.tts.default_voice or "alloy"
    if candidate not in _OPENAI_VOICE_IDS:
        return "alloy"
    return candidate


def _resolve_format(response_format: Optional[str]) -> str:
    return response_format or config.tts.default_format or "mp3"


def _clamp_speaking_rate(speed: Optional[float]) -> float:
    try:
        value = float(speed) if speed is not None else float(config.tts.default_speed)
    except (TypeError, ValueError):
        value = float(config.tts.default_speed)
    return max(_MIN_SPEAKING_RATE, min(_MAX_SPEAKING_RATE, value))


def _prepare_request(
    text: str,
    voice: Optional[str] = None,
    response_format: Optional[str] = None,
    speed: Optional[float] = None,
    model: Optional[str] = None,
) -> tuple[Dict[str, str], Dict[str, Dict[str, float]]]:
    payload: Dict[str, str] = {
        "input": text,
        "model": _resolve_model(model),
        "voice": _resolve_voice(voice),
        "response_format": _resolve_format(response_format),
    }
    voice_settings = {"speaking_rate": _clamp_speaking_rate(speed)}
    return payload, {"voice_settings": voice_settings}


async def _generate_audio_stream(
    text: str,
    voice: Optional[str],
    response_format: Optional[str],
    speed: Optional[float],
    model: Optional[str],
) -> AsyncGenerator[bytes, None]:
    request_kwargs, extra_body = _prepare_request(text, voice, response_format, speed, model)
    async with _client.audio.speech.with_streaming_response.create(
        **request_kwargs,
        stream_format="audio",
        extra_body=extra_body,
    ) as response:
        async for chunk in response.aiter_bytes():
            yield chunk


def generate_speech_stream(
    text: str,
    voice: Optional[str] = None,
    response_format: Optional[str] = None,
    speed: Optional[float] = None,
    model: Optional[str] = None,
) -> AsyncGenerator[bytes, None]:
    return _generate_audio_stream(text, voice, response_format, speed, model)


async def _generate_audio(
    text: str,
    voice: Optional[str],
    response_format: Optional[str],
    speed: Optional[float],
    model: Optional[str],
) -> str:
    request_kwargs, extra_body = _prepare_request(text, voice, response_format, speed, model)
    response = await _client.audio.speech.create(**request_kwargs, extra_body=extra_body)
    output_suffix = f".{request_kwargs['response_format']}"
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=output_suffix)
    temp_path = temp_file.name
    temp_file.close()
    try:
        await response.astream_to_file(temp_path)
    except Exception:
        Path(temp_path).unlink(missing_ok=True)
        raise
    return temp_path


async def generate_speech_async(
    text: str,
    voice: Optional[str] = None,
    response_format: Optional[str] = None,
    speed: Optional[float] = None,
    model: Optional[str] = None,
) -> str:
    return await _generate_audio(text, voice, response_format, speed, model)


def generate_speech(
    text: str,
    voice: Optional[str] = None,
    response_format: Optional[str] = None,
    speed: Optional[float] = None,
    model: Optional[str] = None,
) -> str:
    return asyncio.run(generate_speech_async(text, voice, response_format, speed, model))


def get_models() -> List[Dict[str, str]]:
    return [model.copy() for model in _OPENAI_TTS_MODELS]


def get_models_formatted() -> List[Dict[str, str]]:
    return [{"id": model["id"], "name": model["name"]} for model in _OPENAI_TTS_MODELS]


def get_voices_formatted() -> List[Dict[str, str]]:
    return [{"id": voice["id"], "name": voice["name"]} for voice in _OPENAI_TTS_VOICES]


def get_voices(language: Optional[str] = None) -> List[Dict[str, str]]:
    if language in (None, "all"):
        return [voice.copy() for voice in _OPENAI_TTS_VOICES]
    return [voice.copy() for voice in _OPENAI_TTS_VOICES if voice.get("language") == language]
