"""
browse_shared.py — Base class and shared utilities for scan-data tree browsers.

Provides:
  • _natural_key()         — natural sort utility
  • _launch_slice_viewer() — open mbirjax.slice_viewer in a subprocess
  • ScanBrowser            — abstract base class with all shared UI and search logic

To create a new format browser, subclass ScanBrowser and override:
  _WINDOW_TITLE            class var   window title string
  _COL0_HEADING            class var   column-0 heading text
  _add_toolbar_buttons(toolbar)        add format-specific load button(s)
  _auto_load()                         load default dataset or open a picker
  _on_double_click(event)              handle image display on double-click
  _on_close()                          clean up format resources, then call super()._on_close()
"""

import re
import subprocess
import sys
import tempfile
import tkinter as tk
from tkinter import ttk
import numpy as np


# ── Shared utilities ──────────────────────────────────────────────────────────

def _natural_key(path_components):
    """Sort key that orders embedded integers numerically (Image2 before Image10)."""
    def _tokenize(s):
        return [int(tok) if tok.isdigit() else tok.lower()
                for tok in re.split(r"(\d+)", s)]
    return [_tokenize(c) for c in path_components]


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


# ── Base browser class ────────────────────────────────────────────────────────

class ScanBrowser:
    """
    Abstract base class for scan-data tree browsers.

    Subclasses must override _add_toolbar_buttons, _auto_load, and
    _on_double_click at minimum.  See module docstring for full list.
    """

    _WINDOW_TITLE = "Scan Browser"
    _COL0_HEADING = "Name"

    def __init__(self, root):
        self.root = root
        self._viewers: list = []   # Popen objects for open slice_viewer subprocesses
        self._search_var      = tk.StringVar()
        self._match_label_var = tk.StringVar()
        self._search_hits: list = []
        self._search_idx: int   = -1
        self._build_ui()
        self._auto_load()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        self.root.title(self._WINDOW_TITLE)
        self.root.geometry("1100x750")

        toolbar = ttk.Frame(self.root, padding=4)
        toolbar.pack(side="top", fill="x")
        self._add_toolbar_buttons(toolbar)
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

        self.tree.heading("#0", text=self._COL0_HEADING, anchor="w")
        self.tree.heading("size", text="Bytes",          anchor="e")
        self.tree.heading("hint", text="Hint",           anchor="w")
        self.tree.column("#0", width=320, stretch=False)
        self.tree.column("size", width=75,   anchor="e", stretch=False)
        self.tree.column("hint", width=1500, stretch=False)  # wide + h-scroll

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

    def _add_toolbar_buttons(self, toolbar):
        """Override in subclasses to add format-specific load button(s)."""
        pass

    def _auto_load(self):
        """Override in subclasses to load a default dataset or open a picker."""
        pass

    # ── Generic tree helpers ──────────────────────────────────────────────────

    def _item_path(self, item):
        """Return the tree path as a list of label strings, from root to item."""
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
        """Override in subclasses to handle image display."""
        pass

    def _collapse_all(self):
        for item in self.tree.get_children():
            self.tree.item(item, open=False)

    def _on_close(self):
        for proc in self._viewers:
            if proc.poll() is None:   # still running
                proc.terminate()
        self.root.destroy()

    # ── Search ────────────────────────────────────────────────────────────────

    def _all_items(self, parent="") -> list:
        result = []
        for child in self.tree.get_children(parent):
            result.append(child)
            result.extend(self._all_items(child))
        return result

    def _search_run(self):
        query = self._search_var.get().strip().lower()

        # Clear previous highlights
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
            if query in self.tree.item(item, "text").lower():
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
