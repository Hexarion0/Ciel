"""System telemetry, hardware inspection, and host environment detection for CIEL."""

import os
import platform
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from ciel.config import Settings, get_settings
from ciel.logging import get_logger

logger = get_logger("platform.system_info")


@dataclass
class GpuInfo:
    """NVIDIA GPU hardware status and VRAM metrics."""

    name: str = "Unknown GPU"
    total_vram_mb: int = 0
    free_vram_mb: int = 0
    temperature_c: int = 0
    available: bool = False


@dataclass
class SystemSnapshot:
    """Comprehensive snapshot of CIEL's host environment and runtime capabilities."""

    os_name: str = "Linux"
    os_distro: str = "Arch Linux"
    kernel_version: str = ""
    desktop_environment: str = "Desktop"
    user_name: str = "User"
    hostname: str = "localhost"
    cpu_count: int = 1
    cpu_percent: float = 0.0
    ram_total_gb: float = 0.0
    ram_used_gb: float = 0.0
    ram_percent: float = 0.0
    gpu: GpuInfo = field(default_factory=GpuInfo)
    wake_word: str = "hey ciel"
    stt_model: str = "Not provided"
    tts_engine: str = "Not provided"
    active_providers: list[str] = field(default_factory=list)
    memory_db: str = "data/ciel.db"
    capabilities: list[str] = field(default_factory=list)


def query_gpu_info(*, timeout: float = 1.5) -> GpuInfo:
    """Query NVIDIA GPU metrics via nvidia-smi if available."""
    nvsmi = shutil.which("nvidia-smi")
    if not nvsmi:
        return GpuInfo()

    try:
        out = subprocess.check_output(
            [
                nvsmi,
                "--query-gpu=name,memory.total,memory.free,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=timeout,
            stderr=subprocess.DEVNULL,
        ).strip()
        if out:
            lines = out.splitlines()
            if lines:
                parts = [p.strip() for p in lines[0].split(",")]
                if len(parts) >= 4:
                    return GpuInfo(
                        name=parts[0],
                        total_vram_mb=int(parts[1]),
                        free_vram_mb=int(parts[2]),
                        temperature_c=int(parts[3]),
                        available=True,
                    )
    except Exception as err:
        logger.debug("Could not query GPU stats: %s", err)
    return GpuInfo()


def get_system_snapshot(
    settings: Settings | None = None, *, gpu_timeout: float = 1.5,
) -> SystemSnapshot:
    """Inspect and compile live system telemetry, host info, and CIEL's toolset."""
    cfg = settings or get_settings()

    # 1. Host and OS details
    os_sys = platform.system()
    kernel = platform.release()
    distro = "Linux"
    if hasattr(platform, "freedesktop_os_release"):
        try:
            rel = platform.freedesktop_os_release()
            distro = rel.get("PRETTY_NAME", rel.get("NAME", "Linux"))
        except Exception:
            pass

    desktop = (
        os.environ.get("XDG_CURRENT_DESKTOP")
        or os.environ.get("DESKTOP_SESSION")
        or "Desktop"
    )
    user = os.environ.get("USER") or os.environ.get("USERNAME") or "User"
    hostname = platform.node()

    # 2. CPU & RAM
    cpu_cores = os.cpu_count() or 1
    cpu_pct = 0.0
    ram_tot_gb = 0.0
    ram_used_gb = 0.0
    ram_pct = 0.0

    # Try native Linux /proc/meminfo first
    meminfo_path = Path("/proc/meminfo")
    if meminfo_path.exists():
        try:
            mem_data: dict[str, int] = {}
            for line in meminfo_path.read_text().splitlines():
                parts = line.split(":")
                if len(parts) == 2:
                    k = parts[0].strip()
                    v = parts[1].strip().split()[0]
                    if v.isdigit():
                        mem_data[k] = int(v)
            if "MemTotal" in mem_data and "MemAvailable" in mem_data:
                total_kb = mem_data["MemTotal"]
                avail_kb = mem_data["MemAvailable"]
                used_kb = total_kb - avail_kb
                ram_tot_gb = round(total_kb / (1024 * 1024), 1)
                ram_used_gb = round(used_kb / (1024 * 1024), 1)
                ram_pct = round((used_kb / total_kb) * 100.0, 1)
        except Exception:
            pass

    # Try psutil as fallback
    if ram_tot_gb == 0.0:
        try:
            import psutil

            cpu_pct = psutil.cpu_percent(interval=None)
            vm = psutil.virtual_memory()
            ram_tot_gb = round(vm.total / (1024**3), 1)
            ram_used_gb = round(vm.used / (1024**3), 1)
            ram_pct = round(vm.percent, 1)
        except ImportError:
            pass

    # 3. GPU Info
    gpu_info = query_gpu_info(timeout=gpu_timeout)

    # 4. Providers
    providers = []
    if cfg.providers.nim_api_key:
        providers.append(f"NVIDIA NIM ({cfg.providers.nim_model})")
    if cfg.providers.groq_api_key:
        providers.append("Groq (High-Speed)")
    if cfg.providers.openrouter_api_key:
        providers.append("OpenRouter")
    if cfg.providers.ollama_host:
        providers.append(f"Ollama Local ({cfg.providers.ollama_host})")

    # 5. Core Capabilities
    caps = [
        "Real-time voice dialogue with instant barge-in interruption",
        f"Whisper STT ({cfg.audio.stt_model} on {cfg.audio.stt_device})",
        f"Text-to-Speech ({cfg.audio.tts_engine} / {cfg.audio.tts_voice})",
        f"Multi-tier AI routing ({', '.join([p.split()[0] for p in providers]) or 'Local'})",
        f"Persistent SQLite memory store ({cfg.memory.db_path})",
        "Hardware & system health telemetry inspection",
    ]

    return SystemSnapshot(
        os_name=os_sys,
        os_distro=distro,
        kernel_version=kernel,
        desktop_environment=desktop,
        user_name=user,
        hostname=hostname,
        cpu_count=cpu_cores,
        cpu_percent=cpu_pct,
        ram_total_gb=ram_tot_gb,
        ram_used_gb=ram_used_gb,
        ram_percent=ram_pct,
        gpu=gpu_info,
        wake_word=cfg.audio.wake_word,
        stt_model=f"{cfg.audio.stt_model} ({cfg.audio.stt_device} {cfg.audio.stt_compute_type})",
        tts_engine=f"{cfg.audio.tts_engine} ({cfg.audio.tts_voice})",
        active_providers=providers,
        memory_db=cfg.memory.db_path,
        capabilities=caps,
    )


def generate_system_prompt(
    snapshot: SystemSnapshot | None = None,
    settings: Settings | None = None,
    memories: list[str] | None = None,
) -> str:
    """Generate a rich, context-aware system agenda and advanced persona prompt for CIEL."""
    from ciel.persona import build_advanced_persona, load_persona_preferences

    cfg = settings or get_settings()
    snap = snapshot or get_system_snapshot(settings=cfg)
    now_str = datetime.now().strftime("%A, %B %d, %Y, %I:%M %p")

    gpu_name = snap.gpu.name if snap.gpu.available else "Unknown GPU"

    return build_advanced_persona(
        user_name=snap.user_name,
        time_str=now_str,
        os_distro=snap.os_distro,
        desktop_env=snap.desktop_environment,
        gpu_name=gpu_name,
        gpu_temp_c=snap.gpu.temperature_c,
        gpu_free_mb=snap.gpu.free_vram_mb,
        gpu_total_mb=snap.gpu.total_vram_mb,
        cpu_cores=snap.cpu_count,
        ram_used_gb=snap.ram_used_gb,
        ram_total_gb=snap.ram_total_gb,
        recalled_memories=memories,
        stt_model=snap.stt_model,
        tts_engine=snap.tts_engine,
        memory_db=snap.memory_db,
        preferred_name=cfg.ciel_preferred_name,
        persona_preferences=load_persona_preferences(cfg.ciel_persona_file),
        internet_enabled=cfg.internet.enabled,
    )
