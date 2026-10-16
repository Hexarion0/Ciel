"""Stream outputs supplied by the official Kaggle API without whole-file RAM buffering."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

FLOOR = 3 * 1024**3


def output_path(root: Path, name: str) -> Path:
    path = (root / name).resolve()
    if Path(name).is_absolute() or "\\" in name or not path.is_relative_to(root.resolve()):
        raise ValueError("Kaggle output path leaves its download directory")
    return path


def download(kernel: str, destination: Path) -> None:
    import requests
    from kaggle import api
    from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest

    owner, slug, version = api.parse_kernel_string(kernel)
    if api.get_config_value("username") != owner:
        raise ValueError("Authenticated owner differs from the requested private kernel")
    destination.mkdir(parents=True, exist_ok=True)
    token = None
    inventory = []
    with api.build_kaggle_client() as client:
        while True:
            request = ApiListKernelSessionOutputRequest()
            request.user_name, request.kernel_slug = owner, slug
            request.page_size = 200
            # Match the official CLI: this endpoint selects the latest saved output.
            # Passing a numeric version_label is not accepted by the production API.
            if token:
                request.page_token = token
            response = client.kernels.kernels_api_client.list_kernel_session_output(request)
            for item in response.files or []:
                path = output_path(destination, item.file_name)
                path.parent.mkdir(parents=True, exist_ok=True)
                partial = path.with_name(path.name + ".part")
                if partial.is_symlink():
                    raise ValueError("Refusing to write through a partial download symlink")
                try:
                    # The authenticated official API supplies this URL; never log it.
                    with requests.get(item.url, stream=True, timeout=(30, 120)) as stream:
                        stream.raise_for_status()
                        size = int(stream.headers.get("Content-Length", 0))
                        if path.is_file() and size and path.stat().st_size == size:
                            inventory.append({"file": item.file_name, "bytes": size,
                                              "existing": True})
                            continue
                        if shutil.disk_usage(destination).free < FLOOR + size:
                            raise RuntimeError("Local disk reserve would fall below 3 GiB")
                        received, last_print = 0, time.monotonic()
                        digest = hashlib.sha256()
                        print(f"Downloading {item.file_name} ({size / 1024**3:.2f} GiB)",
                              flush=True)
                        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0)
                        with os.fdopen(os.open(partial, flags, 0o600), "wb") as handle:
                            for chunk in stream.iter_content(chunk_size=4 * 1024**2):
                                if not chunk:
                                    continue
                                if shutil.disk_usage(destination).free < FLOOR + len(chunk):
                                    raise RuntimeError("Local disk free fell to the 3 GiB reserve")
                                handle.write(chunk)
                                digest.update(chunk)
                                received += len(chunk)
                                if time.monotonic() - last_print >= 10:
                                    print(f"  {item.file_name}: {received / 1024**3:.2f} GiB",
                                          flush=True)
                                    last_print = time.monotonic()
                            handle.flush()
                            os.fsync(handle.fileno())
                        if size and received != size:
                            raise RuntimeError("Incomplete download; retain partial for review")
                        partial.replace(path)
                        inventory.append({"file": item.file_name, "bytes": received,
                                          "sha256": digest.hexdigest()})
                except requests.RequestException as error:
                    # Signed output URLs are private. Do not print their query strings.
                    raise RuntimeError(f"Output download failed for {item.file_name}: "
                                       f"{type(error).__name__}") from None
            if response.log:
                (destination / f"{slug}.log").write_text(response.log)
            token = response.next_page_token
            if not token:
                break
    (destination.parent / "streamed-download-inventory.json").write_text(json.dumps({
        "kernel": kernel, "method": "official_Kaggle_API_output_URLs_streamed",
        "credentials_logged": False, "files": inventory}, indent=2) + "\n")
    print(f"Downloaded {len(inventory)} output files with bounded memory", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kernel")
    parser.add_argument("-p", "--path", type=Path, required=True)
    args = parser.parse_args()
    download(args.kernel, args.path.resolve())
