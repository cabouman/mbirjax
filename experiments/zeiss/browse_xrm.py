"""
browse_xrm.py — Interactive tkinter tree browser for .xrm / .txrm OLE files.

All top-level storage nodes start collapsed; click the arrow to expand.
Double-click an image stream to display it with mbirjax.slice_viewer.

Usage: edit FILE_PATH below (optional), then run:
    python zeiss/browse_xrm.py
"""

import math
import re
import struct
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import olefile
from pathlib import Path

from browse_shared import ScanBrowser, _natural_key, _launch_slice_viewer

# ── Default file (leave empty string "" to always start with the file picker) ─
FILE_PATH = "../data/Zeiss/SiC-SiC_CompositeFFOV_tomo-A_Drift.txrm"
# ──────────────────────────────────────────────────────────────────────────────

ARRAY_THRESHOLD = 256  # streams larger than this are treated as blobs

# Zeiss DataType codes → (numpy dtype name, bytes per element)
_ZEISS_DTYPES = {
    5:  ("uint16",  2),
    10: ("float32", 4),
}

# Patterns for image-bearing streams
_RE_IMAGEDATA = re.compile(r"^imagedata\d+$",      re.IGNORECASE)
_RE_IMAGE_NUM = re.compile(r"^image\d+$",           re.IGNORECASE)
_RE_MULTIREF  = re.compile(r"^multireferencedata$", re.IGNORECASE)


def _is_ref_image(path_list):
    """Return True if path_list points to a reference image stream (either layout)."""
    if len(path_list) < 2:
        return False
    parent, name = path_list[-2], path_list[-1]
    return bool((parent.lower() == "referencedata" and name.lower() == "image") or
                (_RE_MULTIREF.match(parent) and _RE_IMAGE_NUM.match(name)))


# ── Metadata reader ───────────────────────────────────────────────────────────

def _read_metadata(ole):
    """
    Read the small set of structurally important fields whose formats are known.
    Returns a dict; any field that cannot be read is set to None.
    """
    def _val(label, fmt):
        try:
            if ole.exists(label):
                raw = ole.openstream(label).read(struct.calcsize(fmt))
                return struct.unpack(fmt, raw)[0]
        except Exception:
            pass
        return None

    n_images       = _val("ImageInfo/NoOfImages",      "<I")
    dtype_code     = _val("ImageInfo/DataType",         "<I")
    ref_dtype_code = (_val("referencedata/DataType",         "<I") or
                      _val("MultiReferenceData/DataType",    "<I"))
    n_ref_images   = (_val("ReferenceData/ImageInfo/NoOfImages",     "<I") or
                      _val("MultiReferenceData/ImageInfo/NoOfImages", "<I") or 1)

    dtype_name, dtype_bytes   = _ZEISS_DTYPES.get(dtype_code,     ("unknown", None))
    ref_dtype_name, _         = _ZEISS_DTYPES.get(ref_dtype_code, ("unknown", None))

    return {
        "image_width":    _val("ImageInfo/ImageWidth",  "<I"),
        "image_height":   _val("ImageInfo/ImageHeight", "<I"),
        "dtype_code":     dtype_code,
        "dtype_name":     dtype_name,
        "dtype_bytes":    dtype_bytes,
        "n_images":       n_images,
        "n_ref_images":   n_ref_images,
        "pixel_size":     _val("ImageInfo/pixelsize",   "<f"),
        "ref_dtype_name": ref_dtype_name,
    }


# ── Hint logic ────────────────────────────────────────────────────────────────

def _peek_hint(ole, path_list, size, metadata=None):
    """Return a short human-readable description of a stream's content."""
    if size == 0:
        return "<empty>"

    # ── Use known metadata for image streams ──────────────────────────────────
    if metadata and len(path_list) >= 2:
        parent, name = path_list[-2], path_list[-1]

        if _RE_IMAGEDATA.match(parent) and _RE_IMAGE_NUM.match(name):
            w  = metadata.get("image_width")
            h  = metadata.get("image_height")
            dt = metadata.get("dtype_name", "?")
            dims = f"{w}\u00d7{h}" if (w and h) else f"{size} bytes"
            return f"<{dt} image, {dims}>  — double-click to display"

        if _is_ref_image(path_list):
            w  = metadata.get("image_width")
            h  = metadata.get("image_height")
            dt = metadata.get("ref_dtype_name", "?")
            n  = metadata.get("n_ref_images", 1)
            dims   = f"{w}\u00d7{h}" if (w and h) else f"{size} bytes"
            plural = f", {n} images" if n > 1 else ""
            return f"<{dt} ref image{plural}, {dims}>  — double-click to display"

    # ── Use known metadata to label per-image scalar arrays ──────────────────
    if metadata:
        n  = metadata.get("n_images")
        db = metadata.get("dtype_bytes")
        if n and n > 1:
            if size == n * 4:
                try:
                    first = struct.unpack("<f", ole.openstream(path_list).read(4))[0]
                    return f"<f32 per-image array, n={n}, first={first:.6g}>"
                except Exception:
                    pass
            if db and size == n * db:
                try:
                    fmt = "<f" if db == 4 else "<H"
                    first = struct.unpack(fmt, ole.openstream(path_list).read(db))[0]
                    return f"<{metadata['dtype_name']} per-image array, n={n}, first={first:.6g}>"
                except Exception:
                    pass

    # ── Generic fallback hints ────────────────────────────────────────────────
    try:
        data = ole.openstream(path_list).read(min(size, max(8, ARRAY_THRESHOLD)))
    except Exception as exc:
        return f"<unreadable: {exc}>"

    # Printable ASCII string (null-terminated).
    # latin-1 is too permissive: any float32 bytes pass as latin-1.
    try:
        s = data.split(b"\x00")[0].decode("ascii")
        if s and all(c.isprintable() or c in "\t\r\n" for c in s):
            snippet = s[:80]
            suffix = "…" if len(s) > 80 else ""
            return f'str: "{snippet}{suffix}"'
    except Exception:
        pass

    # Small scalars: try to pick one unambiguous interpretation; fall back to all.
    if size <= 8:
        cands = {}
        for fmt, label in [("<i", "i32"), ("<I", "u32"), ("<f", "f32"), ("<d", "f64")]:
            if size >= struct.calcsize(fmt):
                try:
                    cands[label] = struct.unpack(fmt, data[:struct.calcsize(fmt)])[0]
                except Exception:
                    pass

        i32 = cands.get("i32")
        f32 = cands.get("f32")
        f64 = cands.get("f64")

        _f32_is_denormal = f32 is not None and 0 < abs(f32) < 1.2e-38

        # Prefer integer when i32 is small, or when f32 would be a denormal.
        if i32 is not None and (-8 <= i32 <= 8 or (i32 > 8 and _f32_is_denormal)):
            return f"i32={i32}"

        # Plausible f32: finite, not a denormal, within a human-readable range.
        if (f32 is not None
                and math.isfinite(f32)
                and not _f32_is_denormal
                and abs(f32) <= 1000):
            return f"f32={f32:.6g}"

        # Plausible f64 (8-byte streams only).
        if (f64 is not None and size == 8
                and math.isfinite(f64)
                and not (0 < abs(f64) < 2.3e-308)  # exclude f64 denormals
                and abs(f64) <= 1000):
            return f"f64={f64:.6g}"

        # Fall back: show all interpretations.
        parts = []
        for label in ("i32", "u32", "f32", "f64"):
            val = cands.get(label)
            if val is not None:
                parts.append(f"{label}={val:.6g}" if label[0] == "f" else f"{label}={val}")
        return " | ".join(parts) if parts else f"<{size}B: {data.hex()}>"

    # Larger generic streams: guess element type from divisibility.
    if size > ARRAY_THRESHOLD:
        for fmt, label, nbytes in [("<f", "f32", 4), ("<d", "f64", 8), ("<I", "u32", 4), ("<i", "i32", 4)]:
            if size % nbytes == 0:
                n = size // nbytes
                try:
                    first = struct.unpack(fmt, data[:nbytes])[0]
                    return (f"<{label} array, n={n}, first={first:.6g}>"
                            if label.startswith("f") else
                            f"<{label} array, n={n}, first={first}>")
                except Exception:
                    pass
        return f"<binary blob, {size} bytes>"

    # Medium: hex snippet.
    hex_str = data.hex()
    display = (hex_str[:48] + "…") if len(hex_str) > 48 else hex_str
    return f"<{size}B: {display}>"


# ── Image I/O ─────────────────────────────────────────────────────────────────

def _read_image_stream(ole, path_list, metadata):
    """Read one OLE image stream and return a 2-D numpy array."""
    w  = metadata.get("image_width")
    h  = metadata.get("image_height")
    is_ref  = _is_ref_image(path_list)
    dt_name = metadata.get("ref_dtype_name" if is_ref else "dtype_name", "uint16")
    if not (w and h):
        raise RuntimeError("Image dimensions not found in metadata.")
    try:
        raw   = ole.openstream(path_list).read()
        dtype = np.dtype(dt_name).newbyteorder("<")
        return np.frombuffer(raw, dtype=dtype).reshape(h, w)
    except Exception as exc:
        raise RuntimeError(f"Could not read image stream: {exc}") from exc


def _all_image_paths(ole):
    """Return naturally-sorted list of all projection image stream paths."""
    return [p for p in sorted(ole.listdir(streams=True, storages=False), key=_natural_key)
            if len(p) >= 2 and _RE_IMAGEDATA.match(p[-2]) and _RE_IMAGE_NUM.match(p[-1])]


def _all_ref_image_paths(ole):
    """Return naturally-sorted list of all reference image stream paths (either layout)."""
    return [p for p in sorted(ole.listdir(streams=True, storages=False), key=_natural_key)
            if _is_ref_image(p)]


# ── Tree builder ──────────────────────────────────────────────────────────────

def _populate_tree(tree, ole, metadata=None):
    """Insert all OLE streams into the Treeview as a nested hierarchy."""
    node_ids = {}

    for path_components in sorted(ole.listdir(streams=True, storages=False),
                                  key=_natural_key):
        size = ole.get_size(path_components)
        hint = _peek_hint(ole, path_components, size, metadata)

        for depth in range(len(path_components) - 1):
            key = tuple(path_components[: depth + 1])
            if key not in node_ids:
                parent_id = node_ids.get(tuple(path_components[:depth]), "")
                node_ids[key] = tree.insert(
                    parent_id, "end",
                    text=path_components[depth],
                    open=False,
                    tags=("storage",),
                )

        parent_id = node_ids.get(tuple(path_components[:-1]), "")
        tree.insert(
            parent_id, "end",
            text=path_components[-1],
            values=(size, hint),
            tags=("stream",),
        )


# ── Browser UI ────────────────────────────────────────────────────────────────

class OLEBrowser(ScanBrowser):

    _WINDOW_TITLE = "OLE Browser"
    _COL0_HEADING = "Stream / Storage"

    def __init__(self, root):
        self.ole          = None
        self.metadata     = None
        self.current_path = None
        super().__init__(root)

    # ── Toolbar ───────────────────────────────────────────────────────────────

    def _add_toolbar_buttons(self, toolbar):
        ttk.Button(toolbar, text="Load File…", command=self._pick_file).pack(side="left", padx=(0, 4))

    # ── Default load ──────────────────────────────────────────────────────────

    def _auto_load(self):
        p = Path(FILE_PATH) if FILE_PATH else None
        if p and p.exists():
            self._load(p)
        else:
            msg = (f"File not found: {FILE_PATH} — opening file picker."
                   if FILE_PATH else "No default file — opening file picker.")
            self.status.set(msg)
            self.root.after(100, self._pick_file)

    # ── Actions ───────────────────────────────────────────────────────────────

    def _pick_file(self):
        initialdir = self.current_path.parent if self.current_path else Path.cwd()
        path = filedialog.askopenfilename(
            title="Open xrm/txrm file",
            initialdir=initialdir,
            filetypes=[("Zeiss XRM/TXRM files", "*.xrm *.txrm"), ("All files", "*.*")],
        )
        if path:
            self._load(Path(path))

    def _load(self, path):
        try:
            new_ole = olefile.OleFileIO(str(path))
        except Exception as exc:
            messagebox.showerror("Load error", f"Could not open file:\n{exc}")
            return

        if self.ole:
            self.ole.close()
        self.ole          = new_ole
        self.current_path = path
        self.metadata     = _read_metadata(self.ole)

        self.tree.delete(*self.tree.get_children())
        self._search_var.set("")
        self._search_hits = []
        self._search_idx  = -1
        self._match_label_var.set("")
        self.status.set(f"Loading {path.name}\u2026")
        self.root.update()
        _populate_tree(self.tree, self.ole, self.metadata)

        n_streams = len(self.ole.listdir(streams=True, storages=False))
        w  = self.metadata.get("image_width")
        h  = self.metadata.get("image_height")
        n  = self.metadata.get("n_images")
        dt = self.metadata.get("dtype_name", "")
        dim_info = f"  |  {n} \u00d7 {w}\u00d7{h} {dt}" if all((w, h, n)) else ""
        self.root.title(f"XRM Browser — {path.name}")
        self.status.set(f"{path.name}  —  {n_streams} streams{dim_info}")

    def _on_double_click(self, _event):
        sel = self.tree.selection()
        if not sel or not self.ole or not self.metadata:
            return
        path_list = self._item_path(sel[0])
        if len(path_list) < 2:
            return

        parent, name = path_list[-2], path_list[-1]
        is_proj_image = _RE_IMAGEDATA.match(parent) and _RE_IMAGE_NUM.match(name)
        is_ref_image  = _is_ref_image(path_list)
        if not (is_proj_image or is_ref_image):
            return

        # ── Decide which images to load ───────────────────────────────────────
        if is_ref_image:
            n_ref = self.metadata.get("n_ref_images", 1)
            if n_ref <= 1:
                paths_to_load = [path_list]
                title = f"{self.current_path.name} — ref image"
            else:
                all_ref = _all_ref_image_paths(self.ole)
                load_all = messagebox.askyesno(
                    "Load reference images",
                    f"This file contains {n_ref} reference images.\n\n"
                    f"Load all {n_ref}?  (may take a moment)\n\n"
                    f"Yes = all     No = selected only",
                )
                paths_to_load = all_ref if load_all else [path_list]
                title = (f"{self.current_path.name} — all {n_ref} ref images"
                         if load_all else "/".join(path_list))
        else:
            all_paths = _all_image_paths(self.ole)
            n_total   = len(all_paths)

            if n_total <= 1:
                paths_to_load = all_paths or [path_list]
                title = "/".join(path_list)
            else:
                load_all = messagebox.askyesno(
                    "Load images",
                    f"This file contains {n_total} projection images.\n\n"
                    f"Load all {n_total}?  (may take a moment)\n\n"
                    f"Yes = all images     No = selected image only",
                )
                if load_all:
                    paths_to_load = all_paths
                    title = f"{self.current_path.name} — all {n_total} images"
                else:
                    paths_to_load = [path_list]
                    title = "/".join(path_list)

        # ── Load and launch ───────────────────────────────────────────────────
        try:
            n = len(paths_to_load)
            self.status.set(f"Loading {n} image{'s' if n > 1 else ''}…")
            self.root.update()
            arrays = [_read_image_stream(self.ole, p, self.metadata) for p in paths_to_load]
            volume = np.stack(arrays, axis=0)          # shape (N, H, W)
        except RuntimeError as exc:
            messagebox.showerror("Image error", str(exc))
            return

        self.status.set(f"Launching slice_viewer: {title}  shape={volume.shape}  dtype={volume.dtype}")
        proc = _launch_slice_viewer(volume, title)
        self._viewers.append(proc)

    def _on_close(self):
        if self.ole:
            self.ole.close()
        super()._on_close()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    root = tk.Tk()
    OLEBrowser(root)
    root.mainloop()


if __name__ == "__main__":
    main()
