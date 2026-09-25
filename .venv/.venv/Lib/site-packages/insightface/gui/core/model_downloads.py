"""Manual model download catalog and helpers.

The GUI never downloads models automatically. Users must open Model Downloads,
refresh URLs, and explicitly start a download.
"""

from __future__ import annotations

import json
import os
import shutil
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional

from .utils import safe_json_dumps, utc_now_iso

GITHUB_RELEASES_URL = "https://github.com/deepinsight/insightface/releases"
GITHUB_LATEST_RELEASE_API = "https://api.github.com/repos/deepinsight/insightface/releases/latest"
FALLBACK_RELEASE_TAG = "v0.7"
FALLBACK_RELEASE_NAME = "insightface v0.7 model packages"
GFPGAN_MODEL_NAME = "GFPGANv1.4.onnx"
GFPGAN_DOWNLOAD_URL = (
    "https://github.com/harisreedhar/Face-Upscalers-ONNX/releases/download/"
    "Models/GFPGANv1.4.onnx"
)


@dataclass
class ModelAsset:
    name: str
    browser_download_url: str
    tag_name: str = FALLBACK_RELEASE_TAG
    release_name: str = FALLBACK_RELEASE_NAME
    size: int = 0
    content_type: str = ""
    updated_at: str = ""
    source: str = "InsightFace"

    @property
    def stem(self) -> str:
        if self.name.endswith(".zip"):
            return self.name[:-4]
        if self.name.endswith(".onnx"):
            return self.name[:-5]
        return Path(self.name).stem

    @property
    def kind(self) -> str:
        if self.source.lower() != "insightface":
            return "third-party restore model"
        if self.name.endswith(".zip"):
            return "model package"
        if self.name.endswith(".onnx"):
            return "onnx model"
        return "asset"


def fallback_model_assets() -> List[ModelAsset]:
    # Latest GitHub release is v0.7. GitHub's HTML page reports nine assets,
    # and the SourceForge mirror lists the uploaded model files. The GitHub
    # download URLs follow the release asset URL pattern.
    names = [
        "antelopev2.zip",
        "buffalo_l.zip",
        "buffalo_m.zip",
        "buffalo_s.zip",
        "buffalo_sc.zip",
        "inswapper_128.onnx",
        "scrfd_person_2.5g.onnx",
    ]
    assets = [
        ModelAsset(
            name=name,
            browser_download_url=f"https://github.com/deepinsight/insightface/releases/download/{FALLBACK_RELEASE_TAG}/{name}",
            tag_name=FALLBACK_RELEASE_TAG,
            release_name=FALLBACK_RELEASE_NAME,
        )
        for name in names
    ]
    return merge_required_assets(assets)


def third_party_model_assets() -> List[ModelAsset]:
    return [
        ModelAsset(
            name=GFPGAN_MODEL_NAME,
            browser_download_url=GFPGAN_DOWNLOAD_URL,
            tag_name="third-party",
            release_name="harisreedhar/Face-Upscalers-ONNX GFPGAN v1.4",
            size=340 * 1024 * 1024,
            content_type="application/octet-stream",
            source="third party",
        )
    ]


def merge_required_assets(assets: Iterable[ModelAsset]) -> List[ModelAsset]:
    merged = list(assets)
    seen = {asset.name for asset in merged}
    for asset in third_party_model_assets():
        if asset.name not in seen:
            merged.append(asset)
            seen.add(asset.name)
    return merged


def cache_file(gui_cache_dir: str | os.PathLike[str]) -> Path:
    path = Path(gui_cache_dir).expanduser() / "model_download_urls.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def asset_from_dict(data: dict) -> ModelAsset:
    return ModelAsset(
        name=str(data.get("name", "")),
        browser_download_url=str(data.get("browser_download_url", "")),
        tag_name=str(data.get("tag_name", data.get("release_tag", FALLBACK_RELEASE_TAG))),
        release_name=str(data.get("release_name", FALLBACK_RELEASE_NAME)),
        size=int(data.get("size") or 0),
        content_type=str(data.get("content_type", "")),
        updated_at=str(data.get("updated_at", "")),
        source=str(data.get("source", "InsightFace")),
    )


def load_cached_assets(gui_cache_dir: str | os.PathLike[str]) -> List[ModelAsset]:
    path = cache_file(gui_cache_dir)
    if not path.exists():
        return fallback_model_assets()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        assets = [asset_from_dict(item) for item in payload.get("assets", []) if item.get("browser_download_url")]
        return merge_required_assets(assets)
    except Exception:
        return fallback_model_assets()


def save_cached_assets(gui_cache_dir: str | os.PathLike[str], assets: Iterable[ModelAsset], source: str = "") -> Path:
    path = cache_file(gui_cache_dir)
    payload = {
        "source": source or GITHUB_LATEST_RELEASE_API,
        "refreshed_at": utc_now_iso(),
        "assets": [asdict(asset) for asset in assets],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def refresh_model_assets(gui_cache_dir: str | os.PathLike[str], timeout: int = 20) -> tuple[List[ModelAsset], str]:
    request = urllib.request.Request(
        GITHUB_LATEST_RELEASE_API,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "InsightFace-Evaluation-Studio/1.0.1",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        assets = fallback_model_assets()
        save_cached_assets(gui_cache_dir, assets, source=f"fallback after refresh error: {exc}")
        return assets, f"Refresh failed; using bundled {FALLBACK_RELEASE_TAG} URLs. Error: {exc}"

    tag = payload.get("tag_name") or FALLBACK_RELEASE_TAG
    release_name = payload.get("name") or tag
    assets = []
    for item in payload.get("assets", []):
        name = item.get("name") or ""
        url = item.get("browser_download_url") or ""
        if not url or not (name.endswith(".zip") or name.endswith(".onnx")):
            continue
        assets.append(
            ModelAsset(
                name=name,
                browser_download_url=url,
                tag_name=tag,
                release_name=release_name,
                size=int(item.get("size") or 0),
                content_type=item.get("content_type") or "",
                updated_at=item.get("updated_at") or "",
            )
        )
    if not assets:
        assets = fallback_model_assets()
        message = f"Latest release returned no model assets; using bundled {FALLBACK_RELEASE_TAG} URLs."
    else:
        assets = merge_required_assets(assets)
        message = f"Refreshed {len(assets)} asset(s), including third-party restore models."
    save_cached_assets(gui_cache_dir, assets, source=GITHUB_LATEST_RELEASE_API)
    return assets, message


def local_model_status(asset: ModelAsset, model_root: str | os.PathLike[str]) -> str:
    root = Path(model_root).expanduser()
    if asset.name.endswith(".zip"):
        target = root / "models" / asset.stem
        if target.exists() and any(target.glob("*.onnx")):
            return f"installed: {target}"
        if target.exists():
            return f"folder exists: {target}"
        return "not installed"
    target_file = root / "models" / asset.stem / asset.name
    if target_file.exists():
        return f"installed: {target_file}"
    legacy_file = root / "models" / asset.name
    if legacy_file.exists():
        return f"installed: {legacy_file}"
    return "not installed"


def is_model_package_installed(model_name: str, model_root: str | os.PathLike[str]) -> bool:
    target = Path(model_root).expanduser() / "models" / model_name
    return target.exists() and any(target.glob("*.onnx"))


def list_installed_swap_models(model_root: str | os.PathLike[str]) -> list[Path]:
    root = Path(model_root).expanduser() / "models"
    if not root.exists():
        return []
    paths = []
    for path in root.rglob("*.onnx"):
        name = path.name.lower()
        parent = path.parent.name.lower()
        if "swap" in name or "inswapper" in name or "swap" in parent or "inswapper" in parent:
            paths.append(path)
    return sorted(paths)


def list_installed_gfpgan_models(model_root: str | os.PathLike[str]) -> list[Path]:
    root = Path(model_root).expanduser() / "models"
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*.onnx") if "gfpgan" in path.name.lower())


def _content_range_total(value: str | None) -> int:
    if not value or "/" not in value:
        return 0
    total = value.rsplit("/", 1)[-1].strip()
    if not total or total == "*":
        return 0
    try:
        return int(total)
    except ValueError:
        return 0


def _download_with_retries(
    url: str,
    destination: Path,
    asset_name: str,
    expected_size: int = 0,
    progress: Optional[Callable[[int, int, str], None]] = None,
    retries: int = 4,
) -> None:
    partial = destination.with_suffix(destination.suffix + ".part")
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        existing = partial.stat().st_size if partial.exists() else 0
        headers = {
            "Accept": "application/octet-stream,*/*",
            "User-Agent": "Mozilla/5.0 InsightFace-Evaluation-Studio/1.0.1",
        }
        if existing:
            headers["Range"] = f"bytes={existing}-"
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                status = getattr(response, "status", response.getcode())
                content_length = int(response.headers.get("Content-Length") or 0)
                content_range_total = _content_range_total(response.headers.get("Content-Range"))
                if existing and status != 206:
                    existing = 0
                    mode = "wb"
                elif existing:
                    mode = "ab"
                else:
                    mode = "wb"
                if content_range_total:
                    total = content_range_total
                elif status == 206:
                    total = existing + content_length
                else:
                    total = content_length or expected_size
                downloaded = existing if mode == "ab" else 0
                with partial.open(mode) as handle:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        handle.write(chunk)
                        downloaded += len(chunk)
                        if progress:
                            progress(downloaded, total or downloaded, f"Downloading {asset_name}")
            partial.replace(destination)
            return
        except (urllib.error.URLError, TimeoutError, ConnectionResetError, OSError) as exc:
            if isinstance(exc, urllib.error.HTTPError) and exc.code == 416 and partial.exists():
                partial.replace(destination)
                return
            last_error = exc
            if progress:
                progress(
                    existing,
                    expected_size or existing or 1,
                    f"Download interrupted; retrying {asset_name} ({attempt}/{retries})",
                )
            if attempt < retries:
                time.sleep(min(2 * attempt, 8))
    raise RuntimeError(f"Download failed for {asset_name} after {retries} attempt(s): {last_error}")


def download_model_asset(
    asset: ModelAsset,
    model_root: str | os.PathLike[str],
    gui_cache_dir: str | os.PathLike[str],
    progress: Optional[Callable[[int, int, str], None]] = None,
) -> Path:
    cache_dir = Path(gui_cache_dir).expanduser() / "models"
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_root_path = Path(model_root).expanduser() / "models"
    model_root_path.mkdir(parents=True, exist_ok=True)

    archive_path = cache_dir / asset.name
    _download_with_retries(
        asset.browser_download_url,
        archive_path,
        asset.name,
        expected_size=asset.size,
        progress=progress,
    )

    if asset.name.endswith(".zip"):
        target_dir = model_root_path / asset.stem
        if target_dir.exists():
            shutil.rmtree(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(target_dir)
        if progress:
            progress(1, 1, f"Extracted to {target_dir}")
        return target_dir

    target_dir = model_root_path / asset.stem
    target_dir.mkdir(parents=True, exist_ok=True)
    target_file = target_dir / asset.name
    shutil.copy2(archive_path, target_file)
    if progress:
        progress(1, 1, f"Saved to {target_file}")
    return target_file
