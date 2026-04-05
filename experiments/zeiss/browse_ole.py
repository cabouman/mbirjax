"""
browse_ole.py — Interactive tkinter tree browser for .xrm / .txrm OLE files.

All top-level storage nodes start collapsed; click the arrow to expand.
Double-click an image stream to display it with mbirjax.slice_viewer.

Usage: edit FILE_PATH below (optional), then run:
    python zeiss/browse_ole.py
"""

import math
import re
import struct
import subprocess
import sys
import tempfile
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import olefile
from pathlib import Path

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
_RE_IMAGEDATA = re.compile(r"^imagedata\d+$",    re.IGNORECASE)
_RE_IMAGE_NUM = re.compile(r"^image\d+$",         re.IGNORECASE)
_RE_MULTIREF  = re.compile(r"^multireferencedata$", re.IGNORECASE)


def _is_ref_image(path_list):
    """Return True if path_list points to a reference image stream (either layout)."""
    if len(path_list) < 2:
        return False
    parent, name = path_list[-2], path_list[-1]
    return bool((parent.lower() == "referencedata" and name.lower() == "image") or
                (_RE_MULTIREF.match(parent) and _RE_IMAGE_NUM.match(name)))


# ── Natural sort ──────────────────────────────────────────────────────────────

def _natural_key(path_components):
    """Sort key that orders embedded integers numerically (Image2 before Image10)."""
    def _tokenize(s):
        return [int(tok) if tok.isdigit() else tok.lower()
                for tok in re.split(r"(\d+)", s)]
    return [_tokenize(c) for c in path_components]


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
        # Decode every candidate that fits in the available bytes.
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

        # Prefer integer when:
        #   • i32 in [-8, 8] (small signed value; f32 of such bytes is always
        #     a denormal, but we also want to catch negative values whose f32
        #     is NaN/Inf and therefore never a useful interpretation), OR
        #   • i32 > 8 and f32 IS a denormal — any positive integer < 2^24
        #     produces a denormal f32, so the float reading is meaningless.
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


def _launch_slice_viewer(volume, title):
    """
    Save *volume* (shape N×H×W) to a temp .npy file and open slice_viewer in a
    subprocess so the browser window stays responsive.  The subprocess deletes
    the temp file when the viewer closes.  Returns the Popen object so the
    caller can track and terminate it later.
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".npy", delete=False)
    tmp.close()
    np.save(tmp.name, volume)

    script = (
        "import numpy as np, mbirjax as mj, os; "
        f"arr = np.load({repr(tmp.name)}); "
        f"mj.slice_viewer(arr, title={repr(title)}, slice_axis=0); "
        f"os.unlink({repr(tmp.name)})"
    )
    return subprocess.Popen([sys.executable, "-c", script])


# ── Tree builder ─────────────────────────────────────────────────────────────

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


# ── Browser UI ───────────────────────────────────────────────────────────────

class OLEBrowser:
    def __init__(self, tk_root):
        self.root     = tk_root
        self.ole      = None
        self.metadata = None
        self.current_path = None
        self._viewers = []   # Popen objects for open slice_viewer subprocesses
        self._search_var       = tk.StringVar()
        self._match_label_var  = tk.StringVar()
        self._search_hits: list = []
        self._search_idx: int   = -1
        self._build_ui()

        p = Path(FILE_PATH) if FILE_PATH else None
        if p and p.exists():
            self._load(p)
        else:
            msg = (f"File not found: {FILE_PATH} — opening file picker."
                   if FILE_PATH else "No default file — opening file picker.")
            self.status.set(msg)
            self.root.after(100, self._pick_file)

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        self.root.title("OLE Browser")
        self.root.geometry("1100x750")

        toolbar = ttk.Frame(self.root, padding=4)
        toolbar.pack(side="top", fill="x")
        ttk.Button(toolbar, text="Load File…",  command=self._pick_file).pack(side="left", padx=(0, 4))
        ttk.Button(toolbar, text="Collapse All", command=self._collapse_all).pack(side="left")

        searchbar = ttk.Frame(self.root, padding=(4, 2))
        searchbar.pack(side="top", fill="x")
        tk.Label(searchbar, text="Search:").pack(side="left", padx=(0, 4))
        self._search_entry = ttk.Entry(searchbar, textvariable=self._search_var, width=30)
        self._search_entry.pack(side="left")
        self._search_entry.bind("<Return>",       lambda _e: self._search_next())
        self._search_entry.bind("<Shift-Return>", lambda _e: self._search_prev())
        self._search_entry.bind("<Escape>",       lambda _e: self._search_clear())
        self._search_var.trace_add("write", lambda *_: self._search_run())
        ttk.Button(searchbar, text="✕", width=2, command=self._search_clear).pack(side="left", padx=(2, 0))
        ttk.Button(searchbar, text="◀", width=2, command=self._search_prev).pack(side="left", padx=(4, 0))
        ttk.Button(searchbar, text="▶", width=2, command=self._search_next).pack(side="left", padx=(2, 4))
        tk.Label(searchbar, textvariable=self._match_label_var, anchor="w").pack(side="left")

        frame = ttk.Frame(self.root)
        frame.pack(fill="both", expand=True, padx=4, pady=4)

        vsb = ttk.Scrollbar(frame, orient="vertical")
        hsb = ttk.Scrollbar(frame, orient="horizontal")

        self.tree = ttk.Treeview(
            frame,
            columns=("size", "hint"),
            show="tree headings",
            yscrollcommand=vsb.set,
            xscrollcommand=hsb.set,
        )
        vsb.config(command=self.tree.yview)
        hsb.config(command=self.tree.xview)

        self.tree.heading("#0", text="Stream / Storage", anchor="w")
        self.tree.heading("size", text="Bytes",          anchor="e")
        self.tree.heading("hint", text="Hint",           anchor="w")
        self.tree.column("#0", width=320, stretch=False)
        self.tree.column("size", width=75, anchor="e", stretch=False)
        self.tree.column("hint", width=680, stretch=True)

        self.tree.tag_configure("storage",    font=("TkDefaultFont", 10, "bold"))
        self.tree.tag_configure("search_hit", background="#c8c8c8")

        vsb.pack(side="right", fill="y")
        hsb.pack(side="bottom", fill="x")
        self.tree.pack(fill="both", expand=True)

        self.tree.bind("<<TreeviewSelect>>", self._on_select)
        self.tree.bind("<Double-1>",         self._on_double_click)

        self.status = tk.StringVar(value="Ready.")
        tk.Label(self.root, textvariable=self.status, anchor="w", relief="sunken").pack(
            fill="x", side="bottom", padx=2, pady=1
        )
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── Actions ───────────────────────────────────────────────────────────────

    def _pick_file(self):
        initialdir = self.current_path.parent if self.current_path else Path.cwd()
        path = filedialog.askopenfilename(
            title="Open OLE file",
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
        _populate_tree(self.tree, self.ole, self.metadata)

        n_streams = len(self.ole.listdir(streams=True, storages=False))
        w  = self.metadata.get("image_width")
        h  = self.metadata.get("image_height")
        n  = self.metadata.get("n_images")
        dt = self.metadata.get("dtype_name", "")
        dim_info = f"  |  {n} \u00d7 {w}\u00d7{h} {dt}" if all((w, h, n)) else ""
        self.root.title(f"OLE Browser — {path.name}")
        self.status.set(f"{path.name}  —  {n_streams} streams{dim_info}")

    def _item_path(self, item):
        """Return the full OLE path as a list of strings for a treeview item."""
        parts, cur = [], item
        while cur:
            parts.insert(0, self.tree.item(cur, "text"))
            cur = self.tree.parent(cur)
        return parts

    def _on_select(self, _event):
        sel = self.tree.selection()
        if not sel:
            return
        item = sel[0]
        vals = self.tree.item(item, "values")
        path_str = "/".join(self._item_path(item))
        self.status.set(path_str + (f"  —  {vals[1]}" if vals else ""))

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

    # ── Search ────────────────────────────────────────────────────────────────

    def _all_items(self, parent="") -> list:
        result = []
        for child in self.tree.get_children(parent):
            result.append(child)
            result.extend(self._all_items(child))
        return result

    def _search_run(self):
        query = self._search_var.get().strip().lower()

        # Clear previous highlights (re-tag with original tag only)
        for item in self._search_hits:
            current_tags = [t for t in self.tree.item(item, "tags") if t != "search_hit"]
            self.tree.item(item, tags=current_tags)
        self._search_hits = []
        self._search_idx  = -1

        if not query:
            self._match_label_var.set("")
            return

        hits = []
        for item in self._all_items():
            text = self.tree.item(item, "text").lower()
            if query in text:
                hits.append(item)
                current_tags = list(self.tree.item(item, "tags"))
                if "search_hit" not in current_tags:
                    current_tags.append("search_hit")
                self.tree.item(item, tags=current_tags)

        self._search_hits = hits
        if hits:
            self._search_idx = 0
            self._scroll_to_hit(0)
        else:
            self._match_label_var.set("No matches")

    def _scroll_to_hit(self, idx):
        item = self._search_hits[idx]
        # Expand all ancestor storages
        parent = self.tree.parent(item)
        while parent:
            self.tree.item(parent, open=True)
            parent = self.tree.parent(parent)
        self.tree.selection_set(item)
        self.tree.see(item)
        self._match_label_var.set(f"{idx + 1} / {len(self._search_hits)}")

    def _search_clear(self):
        self._search_var.set("")
        self._search_entry.focus_set()

    def _search_next(self):
        if not self._search_hits:
            return
        self._search_idx = (self._search_idx + 1) % len(self._search_hits)
        self._scroll_to_hit(self._search_idx)

    def _search_prev(self):
        if not self._search_hits:
            return
        self._search_idx = (self._search_idx - 1) % len(self._search_hits)
        self._scroll_to_hit(self._search_idx)

    def _collapse_all(self):
        for item in self.tree.get_children():
            self.tree.item(item, open=False)

    def _on_close(self):
        for proc in self._viewers:
            if proc.poll() is None:   # still running
                proc.terminate()
        if self.ole:
            self.ole.close()
        self.root.destroy()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    root = tk.Tk()
    OLEBrowser(root)
    root.mainloop()


if __name__ == "__main__":
    main()
