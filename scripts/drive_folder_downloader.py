"""Download a public Google Drive folder without the gdown 50-file/folder limit.

gdown caps each folder listing at 50 entries. This script bypasses that by
parsing Google's `embeddedfolderview?id=<FOLDER_ID>` HTML, which lists every
child item in a public folder. We then download each file by ID using
`gdown --id`, with retries, throttling, and resume support.

Usage:
    python scripts/drive_folder_downloader.py \\
        --folder-id 1lg4dfa6hSnqkAHkg1Y3BpuukMN9ZCpDv \\
        --out-dir drive_data/chatterbox/output-audios

You can call it once per audio subfolder, or use the helper main that walks
a hierarchy from the parent folder (1SGEGaUai2UqOMbwXx447yZeY-6gCU0F_).
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import requests

EMBED_URL = "https://drive.google.com/embeddedfolderview?id={folder_id}#list"
DIRECT_DOWNLOAD = "https://drive.google.com/uc?export=download&id={file_id}"

UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


@dataclass
class DriveItem:
    id: str
    name: str
    is_folder: bool


def list_public_folder(folder_id: str, session: requests.Session | None = None) -> list[DriveItem]:
    """List children of a public Google Drive folder via embeddedfolderview.

    Pages through Google's HTML response which lists ALL items (no 50 cap)
    for public folders.
    """
    session = session or requests.Session()
    session.headers.update({"User-Agent": UA})
    url = EMBED_URL.format(folder_id=folder_id)
    resp = session.get(url, timeout=30)
    resp.raise_for_status()
    html = resp.text

    items: dict[str, DriveItem] = {}
    entry_starts = [
        (m.group(1), m.start(), m.end())
        for m in re.finditer(r'id="entry-([A-Za-z0-9_\-]+)"', html)
    ]
    href_re = re.compile(
        r'href="https://drive\.google\.com/(file|drive/folders)/(?:d/)?([A-Za-z0-9_\-]+)'
    )
    title_re = re.compile(r'class="flip-entry-title"[^>]*>([^<]+)</div>')

    for i, (item_id, _, end) in enumerate(entry_starts):
        next_start = entry_starts[i + 1][1] if i + 1 < len(entry_starts) else len(html)
        body = html[end:next_start]
        href_m = href_re.search(body)
        title_m = title_re.search(body)
        if not href_m or not title_m:
            continue
        is_folder = href_m.group(1) == "drive/folders"
        name = title_m.group(1).strip()
        if not name:
            continue
        items[item_id] = DriveItem(id=item_id, name=name, is_folder=is_folder)

    if not items:
        title_match = re.search(r"<title>([^<]+) - Google Drive</title>", html)
        if title_match:
            print(
                f"[warn] no items parsed for folder {folder_id} (title: {title_match.group(1)!r}). "
                "It may be empty, private, or use a different layout."
            )
    return list(items.values())


def download_one(file_id: str, dest_path: str, session: requests.Session, max_retries: int = 5) -> bool:
    """Download one file via the public uc endpoint. Resumes if dest exists with size > 0."""
    if os.path.isfile(dest_path) and os.path.getsize(dest_path) > 0:
        return True

    tmp = dest_path + ".part"
    backoff = 2.0
    for attempt in range(1, max_retries + 1):
        try:
            with session.get(
                DIRECT_DOWNLOAD.format(file_id=file_id),
                stream=True,
                timeout=60,
                allow_redirects=True,
            ) as r:
                if r.status_code == 429 or "Too Many Requests" in r.text[:500]:
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 60)
                    continue
                r.raise_for_status()
                content_type = r.headers.get("content-type", "")
                if "text/html" in content_type:
                    confirm_token = None
                    for k, v in r.cookies.items():
                        if k.startswith("download_warning"):
                            confirm_token = v
                            break
                    if confirm_token:
                        with session.get(
                            DIRECT_DOWNLOAD.format(file_id=file_id) + f"&confirm={confirm_token}",
                            stream=True,
                            timeout=120,
                        ) as r2:
                            r2.raise_for_status()
                            _stream_to_file(r2, tmp)
                    else:
                        snippet = r.text[:300].replace("\n", " ")
                        raise RuntimeError(f"Got HTML response (likely quota) for {file_id}: {snippet}")
                else:
                    _stream_to_file(r, tmp)

            if os.path.getsize(tmp) == 0:
                os.remove(tmp)
                raise RuntimeError("Downloaded 0 bytes")
            os.replace(tmp, dest_path)
            return True
        except Exception as e:
            print(f"[retry {attempt}/{max_retries}] {file_id}: {e}")
            time.sleep(backoff)
            backoff = min(backoff * 2, 60)
    return False


def _stream_to_file(response, dest_path: str) -> None:
    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=64 * 1024):
            if chunk:
                f.write(chunk)


def download_folder(
    folder_id: str,
    out_dir: str,
    parallel: int = 4,
    only_extensions: tuple[str, ...] | None = None,
) -> tuple[int, int]:
    """Download all files in a public Drive folder. Returns (n_done, n_total)."""
    os.makedirs(out_dir, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": UA})

    items = list_public_folder(folder_id, session=session)
    files = [i for i in items if not i.is_folder]
    if only_extensions:
        files = [i for i in files if i.name.lower().endswith(only_extensions)]

    if not files:
        print(f"[empty] {folder_id} -> no files matched")
        return 0, 0

    results: dict[str, bool] = {}
    print(f"[start] {folder_id}: {len(files)} file(s) -> {out_dir}")
    with ThreadPoolExecutor(max_workers=parallel) as ex:
        futs = {
            ex.submit(download_one, f.id, os.path.join(out_dir, f.name), session): f
            for f in files
        }
        for i, fut in enumerate(as_completed(futs), 1):
            f = futs[fut]
            try:
                ok = fut.result()
            except Exception as e:
                print(f"[fail ] {f.name}: {e}")
                ok = False
            results[f.id] = ok
            if i % 50 == 0 or i == len(files):
                done = sum(1 for v in results.values() if v)
                print(f"[progress] {done}/{len(files)} ok in {out_dir}")

    n_done = sum(1 for v in results.values() if v)
    return n_done, len(files)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--folder-id", required=True, help="Google Drive folder ID")
    parser.add_argument("--out-dir", required=True, help="Local output directory")
    parser.add_argument("--parallel", type=int, default=4, help="Concurrent downloads (be gentle)")
    parser.add_argument("--ext", nargs="*", default=None, help="Restrict to file extensions (e.g. .wav .jsonl)")
    parser.add_argument("--list-only", action="store_true", help="Print file listing and exit")
    args = parser.parse_args()

    session = requests.Session()
    session.headers.update({"User-Agent": UA})
    items = list_public_folder(args.folder_id, session=session)
    print(f"Found {len(items)} item(s) in folder {args.folder_id}:")
    for it in items:
        kind = "DIR " if it.is_folder else "FILE"
        print(f"  [{kind}] {it.id}  {it.name}")
    if args.list_only:
        return

    only_ext = tuple(e.lower() for e in args.ext) if args.ext else None
    n_done, n_total = download_folder(args.folder_id, args.out_dir, args.parallel, only_ext)
    print(f"Done: {n_done}/{n_total}")
    if n_done != n_total:
        sys.exit(1)


if __name__ == "__main__":
    main()
