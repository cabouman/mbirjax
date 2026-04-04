"""
browse_ole.py — Interactive tkinter tree browser for .xrm / .txrm OLE files.

All top-level storage nodes start collapsed; click the arrow to expand.
Use the toolbar to load a different file or export the tree to YAML.

Usage: edit FILE_PATH below (optional), then run:
    python zeiss/browse_ole.py
"""

import struct
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import olefile
from pathlib import Path
from ruamel.yaml import YAML

# ── Default file (leave empty string "" to always start with the file picker) ─
FILE_PATH = "../data/Zeiss/SiC-SiC_CompositeFFOV_tomo-A_Drift.txrm"
# ──────────────────────────────────────────────────────────────────────────────

ARRAY_THRESHOLD = 256  # streams larger than this are treated as blobs


# ── Hint logic ────────────────────────────────────────────────────────────────

def _peek_hint(ole, path_list, size):
    """Return a short human-readable description of a stream's content."""
    if size == 0:
        return "<empty>"
    try:
        stream = ole.openstream(path_list)
        data = stream.read(min(size, max(8, ARRAY_THRESHOLD)))
    except Exception as exc:
        return f"<unreadable: {exc}>"

    # --- try printable ASCII string (null-terminated) -----------------------
    # Use ascii-only (bytes < 128); latin-1 is too permissive and misidentifies
    # binary numeric data (e.g. float32 bytes) as strings.
    try:
        s = data.split(b"\x00")[0].decode("ascii")
        if s and all(c.isprintable() or c in "\t\r\n" for c in s):
            snippet = s[:80]
            suffix = "…" if len(s) > 80 else ""
            return f'str: "{snippet}{suffix}"'
    except Exception:
        pass

    # --- small scalars: show multiple numeric interpretations ---------------
    if size <= 8:
        parts = []
        for fmt, name in [("<i", "i32"), ("<I", "u32"), ("<f", "f32"), ("<d", "f64")]:
            sz = struct.calcsize(fmt)
            if size >= sz:
                try:
                    val = struct.unpack(fmt, data[:sz])[0]
                    parts.append(f"{name}={val:.6g}" if name.startswith("f") else f"{name}={val}")
                except Exception:
                    pass
        return " | ".join(parts) if parts else f"<{size}B: {data.hex()}>"

    # --- larger streams: treat as typed arrays, show element count + first --
    if size > ARRAY_THRESHOLD:
        for fmt, name, nbytes in [("<f", "f32", 4), ("<d", "f64", 8), ("<I", "u32", 4), ("<i", "i32", 4)]:
            if size % nbytes == 0:
                n = size // nbytes
                try:
                    first = struct.unpack(fmt, data[:nbytes])[0]
                    return (f"<{name} array, n={n}, first={first:.6g}>"
                            if name.startswith("f") else
                            f"<{name} array, n={n}, first={first}>")
                except Exception:
                    pass
        return f"<binary blob, {size} bytes>"

    # --- medium: show as hex snippet ----------------------------------------
    hex_str = data.hex()
    display = (hex_str[:48] + "…") if len(hex_str) > 48 else hex_str
    return f"<{size}B: {display}>"


# ── Tree / YAML builders ──────────────────────────────────────────────────────

def _populate_tree(tree, ole):
    """Insert all OLE streams into the Treeview as a nested hierarchy."""
    node_ids = {}  # tuple(path_components) -> treeview item id

    entries = sorted(ole.listdir(streams=True, storages=False))

    for path_components in entries:
        size = ole.get_size(path_components)
        hint = _peek_hint(ole, path_components, size)

        # Ensure every ancestor storage node exists (collapsed by default)
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

        # Insert the leaf stream
        parent_id = node_ids.get(tuple(path_components[:-1]), "")
        tree.insert(
            parent_id, "end",
            text=path_components[-1],
            values=(size, hint),
            tags=("stream",),
        )


def _yaml_insert(node, path_components, value):
    """Insert a value into a nested dict at the given path."""
    for key in path_components[:-1]:
        node = node.setdefault(key, {})
    leaf_key = path_components[-1]
    if leaf_key in node and isinstance(node[leaf_key], dict):
        node[leaf_key]["_stream_info"] = value
    else:
        node[leaf_key] = value


def _build_yaml_tree(ole):
    """Build a nested dict representing the full OLE directory tree."""
    root = {}
    for path_components in ole.listdir(streams=True, storages=False):
        size = ole.get_size(path_components)
        hint = _peek_hint(ole, path_components, size)
        _yaml_insert(root, path_components, {"size_bytes": size, "hint": hint})
    return root


# ── Browser UI ───────────────────────────────────────────────────────────────

class OLEBrowser:
    def __init__(self, tk_root):
        self.root = tk_root
        self.ole = None
        self.current_path = None
        self._build_ui()

        p = Path(FILE_PATH) if FILE_PATH else None
        if p and p.exists():
            self._load(p)
        else:
            msg = f"File not found: {FILE_PATH} — opening file picker." if FILE_PATH else "No default file — opening file picker."
            self.status.set(msg)
            self.root.after(100, self._pick_file)

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        self.root.title("OLE Browser")
        self.root.geometry("1100x750")

        # Toolbar
        toolbar = ttk.Frame(self.root, padding=4)
        toolbar.pack(side="top", fill="x")
        ttk.Button(toolbar, text="Load File…", command=self._pick_file).pack(side="left", padx=(0, 4))
        ttk.Button(toolbar, text="Export YAML…", command=self._export_yaml).pack(side="left")

        # Treeview + scrollbars
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
        self.tree.heading("size", text="Bytes", anchor="e")
        self.tree.heading("hint", text="Hint", anchor="w")
        self.tree.column("#0", width=320, stretch=False)
        self.tree.column("size", width=75, anchor="e", stretch=False)
        self.tree.column("hint", width=680, stretch=True)

        self.tree.tag_configure("storage", font=("TkDefaultFont", 10, "bold"))

        vsb.pack(side="right", fill="y")
        hsb.pack(side="bottom", fill="x")
        self.tree.pack(fill="both", expand=True)

        self.tree.bind("<<TreeviewSelect>>", self._on_select)

        # Status bar
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
        self.ole = new_ole
        self.current_path = path

        self.tree.delete(*self.tree.get_children())
        _populate_tree(self.tree, self.ole)

        n_streams = len(self.ole.listdir(streams=True, storages=False))
        self.root.title(f"OLE Browser — {path.name}")
        self.status.set(f"{path.name}  —  {n_streams} streams")

    def _export_yaml(self):
        if not self.ole:
            messagebox.showinfo("No file loaded", "Load a file before exporting.")
            return

        initialdir = self.current_path.parent if self.current_path else Path.cwd()
        default_name = self.current_path.name + ".yaml" if self.current_path else "output.yaml"
        out_path = filedialog.asksaveasfilename(
            title="Save YAML",
            defaultextension=".yaml",
            filetypes=[("YAML files", "*.yaml"), ("All files", "*.*")],
            initialdir=initialdir,
            initialfile=default_name,
        )
        if not out_path:
            return

        try:
            data = _build_yaml_tree(self.ole)
            yaml = YAML()
            yaml.default_flow_style = False
            yaml.width = 120
            with open(out_path, "w", encoding="utf-8") as f:
                yaml.dump(data, f)
            self.status.set(f"YAML written to {out_path}")
        except Exception as exc:
            messagebox.showerror("Export error", f"Could not write YAML:\n{exc}")

    def _on_select(self, _event):
        sel = self.tree.selection()
        if not sel:
            return
        item = sel[0]
        vals = self.tree.item(item, "values")
        path_parts, cur = [], item
        while cur:
            path_parts.insert(0, self.tree.item(cur, "text"))
            cur = self.tree.parent(cur)
        self.status.set("/".join(path_parts) + (f"  —  {vals[1]}" if vals else ""))

    def _on_close(self):
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
