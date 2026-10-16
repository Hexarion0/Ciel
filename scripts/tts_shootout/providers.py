"""Evaluation adapters around pinned real upstream inference APIs."""

from __future__ import annotations

import contextlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from .common import ROOT, STYLE_REVISION, digest, dump


def cache_key(reference: Path, configuration: dict, inventory: Path) -> str:
    """Cache conditioning by recording, implementation/model, and encoding device."""
    import hashlib

    value = {"reference_sha256": digest(reference), "model_inventory": digest(inventory),
             "configuration": configuration}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


class StyleProvider:
    """Use NeuralVox compute_style/inference unchanged; adapt only offline boot/device."""

    sample_rate = 24000
    streaming = "full utterance waveform"

    def __init__(self, configuration: dict, reference: Path, cache: Path):
        import torch
        from safetensors.torch import load_file, save_file

        vendor = ROOT / "data/voice/vendor/styletts2"
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=vendor, text=True,
        ).strip()
        if revision != STYLE_REVISION:
            raise ValueError("StyleTTS2 implementation is not the pinned evaluation revision")
        model_dir = ROOT / "data/voice/models/styletts2-libritts"
        # The older importable script auto-selects a GPU and downloads at import.
        # Preserve every inference function verbatim; make only boot paths explicit.
        source = (vendor / "msinference.py").read_text()
        source = "import os\n" + source
        source = source.replace("nltk.download('punkt')", "# NLTK assets provisioned offline", 1)
        start = source.index("device = 'cpu'")
        end = source.index("import phonemizer", start)
        source = source[:start] + 'device = os.environ["CIEL_STYLE_DEVICE"]\n\n' + source[end:]
        for filename in ("config.yml", "epochs_2nd_00020.pth"):
            old = f'cached_path("hf://yl4579/StyleTTS2-LibriTTS/Models/LibriTTS/{filename}")'
            new = f'os.path.join(os.environ["CIEL_STYLE_MODEL_DIR"], "Models/LibriTTS/{filename}")'
            if source.count(old) != 1:
                raise ValueError("Unexpected StyleTTS2 boot source; inspect before adapting")
            source = source.replace(old, new)
        cache.mkdir(parents=True, exist_ok=True)
        boot = cache / "styletts2-offline-boot.py"
        boot.write_text(source)
        dump(cache / "styletts2-boot-provenance.json", {
            "upstream_revision": revision, "original_sha256": digest(vendor / "msinference.py"),
            "boot_sha256": digest(boot), "inference_functions_modified": False,
            "changes": ["explicit device", "local pinned model paths", "preprovisioned NLTK"],
        })
        os.environ["CIEL_STYLE_DEVICE"] = configuration["device"]
        os.environ["CIEL_STYLE_MODEL_DIR"] = str(model_dir)
        sys.path.insert(0, str(vendor))
        os.chdir(vendor)
        spec = importlib.util.spec_from_file_location("styletts2_evaluation", boot)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.module = module
        self.steps = configuration["steps"]
        self.device = configuration["device"]
        self.conditioning_loads = 1
        fingerprint = {"implementation": revision, "encoding_device": self.device}
        key = cache_key(reference, fingerprint, model_dir / "inventory.json")
        state = cache / f"style-{key}.safetensors"
        started = time.perf_counter()
        if state.is_file():
            self.state = load_file(state, device=self.device)["style"]
            self.conditioning_encoded = False
        else:
            self.state = module.compute_style(str(reference))
            save_file({"style": self.state.detach().cpu().contiguous()}, str(state))
            self.conditioning_encoded = True
        if not torch.isfinite(self.state).all():
            raise ValueError("StyleTTS2 reference state is non-finite")
        self.conditioning_seconds = time.perf_counter() - started
        self.state_path = str(state)

    def stream(self, text: str):
        yield self.module.inference(text, self.state, alpha=0.3, beta=0.7,
                                    diffusion_steps=self.steps, embedding_scale=1), self.sample_rate

    def close(self):
        self.module = None


class XTTSProvider:
    """Use maintained coqui-tts model streaming, caching both conditioning tensors."""

    sample_rate = 24000
    streaming = "native inference_stream, stream_chunk_size=20"

    def __init__(self, configuration: dict, reference: Path, cache: Path):
        import torch
        from safetensors.torch import load_file, save_file
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts

        accepted = ROOT / "data/voice/shootout-research/xtts-license-acceptance.json"
        if not json.loads(accepted.read_text()).get("accepted"):
            raise ValueError("XTTS model license acceptance has not been recorded")
        directory = Path(configuration.get("model_directory", ROOT / "data/voice/models/xtts-v2"))
        config = XttsConfig()
        config.load_json(str(directory / "config.json"))
        self.model = Xtts.init_from_config(config)
        checkpoint = configuration.get("checkpoint")
        if checkpoint:
            self.model.load_checkpoint(
                config, checkpoint_path=str(checkpoint), vocab_path=str(directory / "vocab.json"),
                speaker_file_path=str(directory / "speakers_xtts.pth"), eval=True,
            )
        else:
            self.model.load_checkpoint(config, checkpoint_dir=str(directory), eval=True)
        self.model.to(configuration["device"]).eval()
        self.device = configuration["device"]
        fingerprint = {"implementation": "coqui-tts-0.27.5", "encoding_device": self.device,
                       "conditioning_precision": "fp32"}
        if checkpoint:
            fingerprint["checkpoint_sha256"] = digest(Path(checkpoint))
        key = cache_key(reference, fingerprint, directory / "inventory.json")
        cache.mkdir(parents=True, exist_ok=True)
        state = cache / f"xtts-{key}.safetensors"
        started = time.perf_counter()
        if state.is_file():
            stored = load_file(state, device=self.device)
            self.latent, self.speaker = stored["gpt_cond_latent"], stored["speaker_embedding"]
            self.conditioning_encoded = False
        else:
            self.latent, self.speaker = self.model.get_conditioning_latents(
                audio_path=[str(reference)],
            )
            save_file({"gpt_cond_latent": self.latent.detach().cpu().contiguous(),
                       "speaker_embedding": self.speaker.detach().cpu().contiguous()}, str(state))
            self.conditioning_encoded = True
        self.conditioning_loads = 1
        self.conditioning_seconds = time.perf_counter() - started
        self.state_path = str(state)
        self.amp = configuration["precision"] == "fp16 AMP"
        self.dtype = str(next(self.model.parameters()).dtype)
        if not torch.isfinite(self.latent).all() or not torch.isfinite(self.speaker).all():
            raise ValueError("XTTS reference state is non-finite")

    def stream(self, text: str):
        import torch

        # XTTS's null-position embedding returns FP32 zeros; model.half() promotes
        # inputs back to FP32 then fails at Half layer norms. AMP preserves the
        # upstream FP32 parameters and safely casts eligible CUDA operations.
        context = (torch.autocast("cuda", dtype=torch.float16)
                   if self.amp else contextlib.nullcontext())
        with context:
            for chunk in self.model.inference_stream(
                text, "en", self.latent, self.speaker,
                stream_chunk_size=20, enable_text_splitting=False,
            ):
                yield chunk.detach().float().cpu().numpy().reshape(-1), self.sample_rate

    def close(self):
        self.model = None


class ChatterboxProvider:
    """Call the existing persistent Turbo worker without changing its inference behavior."""

    streaming = "existing full utterance worker waveform"

    def __init__(self, configuration: dict, reference: Path, cache: Path):
        from ciel.audio.chatterbox_worker import ChatterboxTurboWorker

        # The existing client validates a nonempty transcript but sends only audio
        # to the actual Turbo conditioning API. Use the actual baseline transcript.
        transcript = (ROOT / "data/voice/references/ciel/reference-1.txt").read_text()
        self.worker = ChatterboxTurboWorker(
            ROOT / "data/voice/envs/clone/bin/python",
            ROOT / "scripts/chatterbox_turbo_worker.py",
            ROOT / "data/voice/models/chatterbox-turbo", reference, transcript,
            synthesis_timeout=180,
            event_reference=ROOT / "data/voice/references/ciel/event-style.wav",
            cue_audio={key: ROOT / f"data/voice/references/ciel/{key}.wav"
                       for key in ("sigh", "groan")},
        )
        self.startup = self.worker.start()
        self.conditioning_loads = 1
        self.conditioning_encoded = True
        self.conditioning_seconds = self.startup["conditioning_seconds"]
        self.state_path = "existing persistent worker conditionals"

    def stream(self, text: str):
        result = self.worker.synthesize(text)
        if result is None:
            raise RuntimeError("Existing Chatterbox worker returned no audio")
        yield result

    def close(self):
        self.worker.close()


class NeuTTSProvider:
    """Evaluate native CPU GGUF streaming; encode the identical reference only once."""

    sample_rate = 24000
    streaming = "native infer_stream, upstream 25-frame chunks"

    def __init__(self, configuration: dict, reference: Path, cache: Path):
        import torch
        from neutts import NeuTTS
        from safetensors.torch import load_file, save_file

        if configuration["device"] != "cpu":
            raise ValueError("This optional candidate is evaluated on CPU only")
        inventory = ROOT / "data/voice/models/neutts-air/inventory.json"
        metadata = json.loads(inventory.read_text())
        assets = metadata["assets"]
        backbone = Path(assets["neuphonic/neutts-air-q8-gguf"]["directory"])
        decoder = Path(assets["neuphonic/neucodec-onnx-decoder"]["directory"])
        self.transcript = (ROOT / "data/voice/references/ciel/reference-1.txt").read_text().strip()
        self.conditioning_loads = 1
        key = cache_key(reference, {"implementation": "neutts-1.4.1", "device": "cpu"}, inventory)
        state = cache / f"neutts-{key}.safetensors"
        cache.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        if state.is_file():
            self.codes = load_file(state)["codes"]
            self.conditioning_encoded = False
        else:
            from .neutts_reference import encode_reference_cpu

            self.codes = encode_reference_cpu(reference, metadata, state.with_suffix(".json"))
            save_file({"codes": self.codes}, str(state))
            self.conditioning_encoded = True
        if not self.codes.numel() or not torch.isfinite(self.codes).all():
            raise ValueError("NeuTTS reference codes are empty or non-finite")
        self.conditioning_seconds = time.perf_counter() - started
        self.state_path = str(state)
        self.model = NeuTTS(
            backbone_repo=str(backbone / "neutts-air-Q8_0.gguf"), backbone_device="cpu",
            codec_repo=str(decoder / "model.onnx"), codec_device="cpu", language="en-us",
        )
        self.runtime_details = {
            "onnx_providers": self.model.codec.session.get_providers(),
            "watermark_enabled": self.model.watermarker is not None,
            "llama_threads": self.model.backbone.n_threads,
            "llama_batch_threads": self.model.backbone.n_threads_batch,
            "gpu_layers": 0,
        }
        if "CUDAExecutionProvider" in self.runtime_details["onnx_providers"]:
            raise ValueError("Unexpected CUDA decoder in a CPU-only evaluation")

    def stream(self, text: str):
        # Sampling defaults and native watermarking remain unchanged.
        for wave in self.model.infer_stream(text, self.codes, self.transcript):
            yield wave, self.sample_rate

    def close(self):
        if self.model is not None:
            self.model.backbone.close()
        self.model = None


def load(configuration: dict, reference: Path, cache: Path):
    """Choose only the requested engine; fallback must never impersonate a result."""
    classes = {"styletts2": StyleProvider, "xtts": XTTSProvider,
               "chatterbox": ChatterboxProvider, "neutts": NeuTTSProvider}
    return classes[configuration["engine"]](configuration, reference, cache)
