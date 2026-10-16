"""CUDA startup must not install an unrelated process-wide BLAS replacement."""

import ctypes
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ciel.audio.stt import _preload_cuda_libraries


@pytest.fixture
def cuda_libraries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Provide CUDA 12 runtime libraries beside unrelated NVIDIA packages."""
    monkeypatch.setattr(sys, "prefix", str(tmp_path))
    nvidia = (
        tmp_path / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages/nvidia"
    )
    for relative in (
        "cublas/lib/libcublasLt.so.12",
        "cublas/lib/libcublas.so.12",
        "cublas/lib/libnvblas.so.12",
        "cudnn/lib/libcudnn.so.9",
        "cu13/lib/libcublas.so.13",
        "cu13/lib/libnvblas.so.13",
        "cu13/lib/libcudnn.so.9",
        "nvshmem/lib/libnvshmem_host.so.3",
        "nccl/lib/libnccl.so.2",
    ):
        library = nvidia / relative
        library.parent.mkdir(parents=True, exist_ok=True)
        library.touch()
    return nvidia


def test_cuda_preload_excludes_blas_interceptors_and_unrelated_packages(
    cuda_libraries: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep CPU BLAS symbols intact even when CUDA 12 and 13 coexist."""
    loader = MagicMock()
    monkeypatch.setattr(ctypes, "CDLL", loader)

    _preload_cuda_libraries()

    assert [Path(call.args[0]) for call in loader.call_args_list] == [
        cuda_libraries / "cublas/lib/libcublasLt.so.12",
        cuda_libraries / "cublas/lib/libcublas.so.12",
        cuda_libraries / "cudnn/lib/libcudnn.so.9",
    ]
    assert all(call.kwargs["mode"] == ctypes.RTLD_GLOBAL for call in loader.call_args_list)


def test_cuda_preload_tolerates_unavailable_libraries(
    cuda_libraries: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unavailable optional runtime must still allow the CPU fallback."""
    loader = MagicMock(side_effect=OSError("CUDA library unavailable"))
    monkeypatch.setattr(ctypes, "CDLL", loader)

    _preload_cuda_libraries()

    assert loader.call_count == 3


def test_cuda_preload_skips_missing_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CPU-only installations do not attempt to load native CUDA libraries."""
    monkeypatch.setattr(sys, "prefix", str(tmp_path))
    loader = MagicMock()
    monkeypatch.setattr(ctypes, "CDLL", loader)

    _preload_cuda_libraries()

    loader.assert_not_called()
