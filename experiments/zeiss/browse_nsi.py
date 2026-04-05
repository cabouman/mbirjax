"""
browse_nsi.py — Interactive tkinter tree browser for North Star Imaging (NSI) CT datasets.

An NSI dataset is a folder containing:
  *.nsipro               — XML-ish project config (expanded inline in the tree)
  Geometry*.rtf          — Geometry report (key/value table, expanded inline)
  Corrections/           — gain*.tif, offset.tif, defective_pixels.defect, correction.cfg
  Radiographs-*/         — projection TIFFs + optional positions.txt
  Geometry/              — geometry calibration TIFFs + optional positions.txt
  (any other folders)    — shown as generic file-list nodes

All top-level nodes start collapsed.  Double-click a TIFF to open slice_viewer;
double-click a text file to open a scrollable text viewer.

Usage: edit DATASET_DIR below (optional), then run:
    python zeiss/browse_nsi.py
"""

import os
import re
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import tifffile

from browse_shared import ScanBrowser, _natural_key, _launch_slice_viewer

# ── Default dataset folder (empty string → always open folder picker) ─────────
DATASET_DIR = "../data/demo_data_nsi"
# ──────────────────────────────────────────────────────────────────────────────

# Files/folders to silently ignore when browsing
_SKIP_NAMES = {".ds_store", "thumbs.db", "desktop.ini"}

def _should_skip(name):
    """Return True for system/lock files that should never appear in the tree."""
    nl = name.lower()
    return nl in _SKIP_NAMES or nl.startswith("~$")

# Folder name patterns (case-insensitive prefix)
_CORRECTIONS_RE   = re.compile(r"^corrections$",        re.IGNORECASE)
_RADIOGRAPHS_RE   = re.compile(r"^radiographs",         re.IGNORECASE)
_GEOMETRY_RE      = re.compile(r"^geometry$",           re.IGNORECASE)


# ── Utility functions ─────────────────────────────────────────────────────────

def _format_size(n):
    """Return a human-readable file size string."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f} MB"
    if n >= 1_000:
        return f"{n / 1_000:.1f} KB"
    return f"{n} B"


def _tiff_shape_dtype(path):
    """Return (shape, dtype_str) from TIFF metadata without loading pixels."""
    try:
        with tifffile.TiffFile(str(path)) as tif:
            page = tif.pages[0]
            return page.shape, str(page.dtype)
    except Exception:
        return None, None


def _tiff_hint(path, role="image"):
    """Return a display hint string for a TIFF file."""
    shape, dtype = _tiff_shape_dtype(path)
    if shape and dtype:
        h, w = shape[0], shape[1]
        return f"<{dtype} {role}, {h}\u00d7{w}> \u2014 double-click to display"
    return f"<TIFF {role}> \u2014 double-click to display"


# ── nsipro parser ─────────────────────────────────────────────────────────────

def _parse_nsipro(path):
    """
    Parse a .nsipro file into a nested dict {section: {field: value}}.

    The format is NOT standard XML.  Each line is one of:
      <SectionName>          ← section open (nothing after >)
      </SectionName>         ← section close
      <field name>value      ← leaf field (no closing tag)
    Section names can contain spaces.  Duplicate keys get a _N suffix.
    """
    stack   = []          # list of (name, dict) pairs
    root    = {}
    current = root

    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
    except Exception:
        return root

    tag_re = re.compile(r"^<([^/][^>]*)>(.*)")

    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        if line.startswith("</"):
            # Section close — pop stack; tuple holds the parent dict
            if stack:
                _, current = stack.pop()
            continue

        m = tag_re.match(line)
        if not m:
            continue

        tag   = m.group(1)
        value = m.group(2).strip()

        if value == "":
            # Section open — push
            new_dict = {}
            # Deduplicate key if needed
            key = tag
            if key in current:
                idx = 1
                while f"{key}_{idx}" in current:
                    idx += 1
                key = f"{key}_{idx}"
            current[key] = new_dict
            stack.append((key, current))
            current = new_dict
        else:
            # Leaf field — normalise Windows path separators
            key = tag
            if key in current:
                idx = 1
                while f"{key}_{idx}" in current:
                    idx += 1
                key = f"{key}_{idx}"
            current[key] = value.replace("\\", "/")

    return root


# ── RTF parser ────────────────────────────────────────────────────────────────

# Top-level destination groups to skip entirely (contain binary/font data)
_RTF_SKIP_WORDS = frozenset({
    "pict", "objdata", "fonttbl", "colortbl", "stylesheet",
    "info", "rsidtbl", "xmlnstbl", "themedata", "colorschememapping",
    "wgrffmtfilter", "latentstyles",
})

def _parse_rtf_lines(path):
    """
    Parse a .rtf file and return a list of non-empty text lines,
    skipping all binary/image/font data.
    """
    try:
        with open(path, "rb") as fh:
            raw = fh.read().decode("latin-1", errors="replace")
    except Exception:
        return []

    result     = []
    depth      = 0
    skip_depth = None
    i          = 0
    n          = len(raw)
    buf        = []

    def flush():
        text = "".join(buf).strip()
        buf.clear()
        if text:
            result.append(text)

    while i < n:
        c = raw[i]

        if c == "{":
            depth += 1
            i += 1
        elif c == "}":
            if skip_depth is not None and depth == skip_depth:
                skip_depth = None
            depth -= 1
            i += 1
        elif skip_depth is not None:
            i += 1
        elif c == "\\":
            if i + 1 >= n:
                i += 1
                continue
            nc = raw[i + 1]
            if nc == "\n":
                flush()
                i += 2
            elif nc == "'":
                # \'xx — escaped character
                if i + 3 < n:
                    try:
                        buf.append(chr(int(raw[i + 2:i + 4], 16)))
                    except ValueError:
                        pass
                i += 4
            elif nc.isalpha():
                # Control word
                j = i + 1
                while j < n and raw[j].isalpha():
                    j += 1
                word = raw[i + 1:j]
                # Skip optional number
                if j < n and (raw[j].isdigit() or raw[j] == "-"):
                    while j < n and (raw[j].isdigit() or raw[j] == "-"):
                        j += 1
                # Skip one trailing space
                if j < n and raw[j] == " ":
                    j += 1

                if word in ("par", "line", "row", "cell", "sect", "page"):
                    flush()
                elif word in _RTF_SKIP_WORDS:
                    skip_depth = depth
                i = j
            else:
                # Control symbol.  RTF-spec escapes: \\ → /, \{ → {, \} → }
                # NSI geometry reports embed raw Windows paths (e.g. \DR, \21…)
                # which are technically invalid RTF.  Treat any unrecognised
                # \<non-alpha> as a path separator (/) followed by the character.
                if nc == "\\":
                    buf.append("/")
                elif nc in ("{", "}"):
                    buf.append(nc)
                elif nc in ("~", "-", "_", "|", ":"):
                    pass  # RTF special symbols — discard
                else:
                    # Likely an unescaped Windows path backslash
                    buf.append("/")
                    buf.append(nc)
                i += 2
        else:
            # Raw \r and \n in RTF source are whitespace, not paragraph breaks
            if skip_depth is None and c not in ("\r", "\n"):
                buf.append(c)
            i += 1

    flush()
    # Join any lines that look like split Windows drive paths (e.g. "E:/" + "rest")
    merged = []
    i = 0
    while i < len(result):
        line = result[i]
        if (re.match(r"^[A-Za-z]:/?$", line) and
                i + 1 < len(result) and not result[i + 1].endswith(":")):
            merged.append(line.rstrip("/") + "/" + result[i + 1])
            i += 2
        else:
            merged.append(line)
            i += 1
    return merged


def _rtf_to_sections(lines):
    """
    Convert RTF text lines (from _parse_rtf_lines) into a nested dict
    suitable for inline tree display.

    Structure detection heuristic:
      - Lines ending with ':' whose *next* non-empty line also ends with ':'
        are treated as section headers (the table's left-column bold rows).
      - Lines ending with ':' followed by a value line are field keys.
      - Garbage lines before the first recognisable content are discarded.

    Returns OrderedDict-like plain dict preserving insertion order (Python 3.7+).
    """
    # Discard preamble noise — keep only lines that look like label or value text
    clean = []
    for line in lines:
        # Skip very short junk lines composed only of punctuation/parens
        if re.fullmatch(r"[\s().;,'/\\-]*", line):
            continue
        clean.append(line)

    sections = {}
    current_section = None
    current_dict    = {}

    i = 0
    while i < len(clean):
        line = clean[i]
        is_label = line.endswith(":")

        if is_label:
            # Peek at next line
            next_line = clean[i + 1] if i + 1 < len(clean) else ""
            next_is_label = next_line.endswith(":")

            if next_is_label:
                # Section header — save previous section, start new one
                if current_section is not None:
                    sections[current_section] = current_dict
                current_section = line[:-1]   # strip trailing ':'
                current_dict    = {}
            else:
                # Field key — grab value on next line
                key   = line[:-1]
                value = next_line
                if current_section is None:
                    current_section = "Header"
                    current_dict    = {}
                current_dict[key] = value
                i += 1   # skip value line
        # non-label lines that aren't consumed as values are ignored
        i += 1

    # Flush last section
    if current_section is not None:
        sections[current_section] = current_dict

    return sections


# ── positions.txt parser ──────────────────────────────────────────────────────

def _parse_positions_txt(path):
    """
    Parse a positions.txt file.  Returns a dict mapping tif basename → angle (float).
    Lines starting with '#' are skipped.
    """
    result = {}
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(None, 1)
                if len(parts) == 2:
                    try:
                        angle    = float(parts[0])
                        basename = parts[1].replace("\\", "/").split("/")[-1]
                        result[basename] = angle
                    except ValueError:
                        pass
    except Exception:
        pass
    return result


# ── Defect counter ────────────────────────────────────────────────────────────

def _count_defects(path):
    """Return the number of defective pixels listed in a .defect XML file."""
    try:
        tree = ET.parse(str(path))
        return len(tree.findall(".//Defect"))
    except Exception:
        pass
    # Fallback: count lines containing '<Defect>'
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return sum(1 for ln in fh if "<Defect>" in ln)
    except Exception:
        return 0


# ── Text viewer ───────────────────────────────────────────────────────────────

def _open_text_viewer(parent_root, title, path):
    """Open a simple scrollable text viewer window for a file."""
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            content = fh.read()
    except Exception as exc:
        messagebox.showerror("Read error", f"Could not read file:\n{exc}")
        return

    win = tk.Toplevel(parent_root)
    win.title(title)
    win.geometry("900x600")

    frame = ttk.Frame(win)
    frame.pack(fill="both", expand=True)

    vsb = ttk.Scrollbar(frame, orient="vertical")
    hsb = ttk.Scrollbar(frame, orient="horizontal")
    txt = tk.Text(frame, wrap="none", font=("Courier", 10),
                  yscrollcommand=vsb.set, xscrollcommand=hsb.set)
    vsb.config(command=txt.yview)
    hsb.config(command=txt.xview)

    vsb.pack(side="right",  fill="y")
    hsb.pack(side="bottom", fill="x")
    txt.pack(fill="both", expand=True)

    txt.insert("1.0", content)
    txt.config(state="disabled")

    ttk.Button(win, text="Close", command=win.destroy).pack(pady=4)


# ── NSI Browser ───────────────────────────────────────────────────────────────

class NSIBrowser(ScanBrowser):

    _WINDOW_TITLE = "NSI Dataset Browser"
    _COL0_HEADING = "File / Section"

    def __init__(self, root):
        self.dataset_dir   = None
        self._item_action  = {}   # item_id → ("tiff"|"text", Path)
        super().__init__(root)

    # ── Toolbar ───────────────────────────────────────────────────────────────

    def _add_toolbar_buttons(self, toolbar):
        ttk.Button(toolbar, text="Load Folder\u2026",
                   command=self._pick_folder).pack(side="left", padx=(0, 4))

    # ── Default load ──────────────────────────────────────────────────────────

    def _auto_load(self):
        if DATASET_DIR:
            p = (Path(__file__).parent / DATASET_DIR).resolve()
            if p.is_dir():
                self._load(p)
                return
            self.status.set(f"Default folder not found: {DATASET_DIR} \u2014 opening picker.")
        else:
            self.status.set("No default folder \u2014 opening folder picker.")
        self.root.after(100, self._pick_folder)

    # ── Actions ───────────────────────────────────────────────────────────────

    def _pick_folder(self):
        start = str(self.dataset_dir.parent) if self.dataset_dir else str(Path.cwd())
        folder = filedialog.askdirectory(title="Open NSI Dataset Folder",
                                         initialdir=start)
        if folder:
            self._load(Path(folder))

    def _load(self, dataset_dir):
        self.dataset_dir  = dataset_dir
        self._item_action = {}

        # Reset tree and search state
        self.tree.delete(*self.tree.get_children())
        self._search_var.set("")
        self._search_hits = []
        self._search_idx  = -1
        self._match_label_var.set("")

        self._populate_tree()

        self.root.title(f"NSI Dataset Browser \u2014 {dataset_dir.name}")
        self.status.set(str(dataset_dir))

    # ── Tree population ───────────────────────────────────────────────────────

    def _populate_tree(self):
        d = self.dataset_dir
        recognised_names = set()  # track what we handle specially

        # ── .nsipro file(s) ──────────────────────────────────────────────────
        for nsi_path in sorted(d.glob("*.nsipro"), key=lambda p: _natural_key([p.name])):
            if _should_skip(nsi_path.name):
                continue
            recognised_names.add(nsi_path.name.lower())
            self._insert_nsipro_node(nsi_path)

        # ── Geometry report(s) ───────────────────────────────────────────────
        for rtf_path in sorted(d.glob("*.rtf"), key=lambda p: _natural_key([p.name])):
            if _should_skip(rtf_path.name):
                continue
            recognised_names.add(rtf_path.name.lower())
            self._insert_rtf_node(rtf_path)

        # ── Known sub-folders and any other items ─────────────────────────────
        items = sorted(d.iterdir(), key=lambda p: _natural_key([p.name]))
        for item in items:
            if _should_skip(item.name):
                continue
            if item.name.lower() in recognised_names:
                continue
            if item.is_dir():
                name_lower = item.name.lower()
                if _CORRECTIONS_RE.match(item.name):
                    self._insert_corrections_folder(item)
                elif _RADIOGRAPHS_RE.match(item.name):
                    self._insert_tiff_folder(item, label=item.name,
                                             tiff_role="radiograph",
                                             has_positions=True)
                elif _GEOMETRY_RE.match(item.name):
                    self._insert_tiff_folder(item, label=item.name,
                                             tiff_role="geometry radiograph",
                                             has_positions=True)
                else:
                    self._insert_generic_folder(item)
            else:
                # Root-level file we don't recognise specially
                recognised_names.add(item.name.lower())
                self._insert_generic_file(item, parent_id="")

    # ── nsipro node ───────────────────────────────────────────────────────────

    def _insert_nsipro_node(self, path):
        size = path.stat().st_size
        data = _parse_nsipro(path)

        # Unwrap the single outer envelope (e.g. "NSI Reconstruction Project")
        # so all sections appear as direct children of the nsipro tree node.
        if len(data) == 1 and isinstance(list(data.values())[0], dict):
            data = list(data.values())[0]

        hint = self._nsipro_summary(data)

        node_id = self.tree.insert(
            "", "end",
            text=path.name,
            values=(_format_size(size), hint),
            open=False,
            tags=("storage",),
        )
        self._insert_dict_subtree(node_id, data)

    def _nsipro_summary(self, d):
        """Extract a one-line summary from the (unwrapped) nsipro dict."""
        try:
            n    = d.get("Object Radiograph", {}).get("number", "?")
            step = d.get("Object Radiograph", {}).get("angleStep", "?")
            det  = (d.get("CT Project Configuration", {})
                     .get("Technique Configuration", {})
                     .get("Detector", {}))
            w    = det.get("width pixels", "?")
            h    = det.get("height pixels", "?")
            return f"{n} views, {step}\u00b0 step, {w}\u00d7{h} px"
        except Exception:
            return ""

    def _insert_dict_subtree(self, parent_id, d):
        """Recursively insert a nested dict as tree nodes under parent_id."""
        for key, value in d.items():
            if isinstance(value, dict):
                child_id = self.tree.insert(
                    parent_id, "end",
                    text=key,
                    values=("", ""),
                    open=False,
                    tags=("storage",),
                )
                self._insert_dict_subtree(child_id, value)
            else:
                self.tree.insert(
                    parent_id, "end",
                    text=key,
                    values=("", str(value)),
                    tags=("stream",),
                )

    # ── RTF node ──────────────────────────────────────────────────────────────

    def _insert_rtf_node(self, path):
        size     = path.stat().st_size
        rtf_lines  = _parse_rtf_lines(path)
        sections = _rtf_to_sections(rtf_lines)

        if sections:
            # Build a hint from the first few interesting fields found anywhere
            hint = self._rtf_summary(sections)
            node_id = self.tree.insert(
                "", "end",
                text=path.name,
                values=(_format_size(size), hint),
                open=False,
                tags=("storage",),
            )
            self._insert_dict_subtree(node_id, sections)
        else:
            # Fallback: show as plain text-viewable file
            node_id = self.tree.insert(
                "", "end",
                text=path.name,
                values=(_format_size(size), "Geometry report \u2014 double-click to view"),
                tags=("stream",),
            )
            self._item_action[node_id] = ("text", path)

    def _rtf_summary(self, sections):
        """Pull a short summary from the RTF geometry report sections."""
        try:
            dist = sections.get("Distances", {})
            src_det = dist.get("Source to detector", "")
            mag     = sections.get("", {})
            # grab magnification from any section
            for s in sections.values():
                if "Magnification" in s:
                    return f"SDD: {src_det},  Mag: {s['Magnification']}"
            if src_det:
                return f"Source to detector: {src_det}"
        except Exception:
            pass
        return "Geometry report"

    # ── Corrections/ folder ───────────────────────────────────────────────────

    def _insert_corrections_folder(self, folder_path):
        node_id = self.tree.insert(
            "", "end",
            text=folder_path.name + "/",
            values=("", ""),
            open=False,
            tags=("storage",),
        )

        items = sorted(folder_path.iterdir(), key=lambda p: _natural_key([p.name]))
        for item in items:
            if _should_skip(item.name):
                continue
            name_lower = item.name.lower()
            size = item.stat().st_size

            if name_lower.endswith(".tif") or name_lower.endswith(".tiff"):
                if re.match(r"^gain\d*", name_lower):
                    role = "flat-field"
                elif name_lower.startswith("offset"):
                    role = "dark frame"
                else:
                    role = "correction image"
                hint = _tiff_hint(item, role)
                child_id = self.tree.insert(
                    node_id, "end",
                    text=item.name,
                    values=(_format_size(size), hint),
                    tags=("stream",),
                )
                self._item_action[child_id] = ("tiff", item)

            elif name_lower.endswith(".defect"):
                n    = _count_defects(item)
                hint = f"{n} defective pixels"
                child_id = self.tree.insert(
                    node_id, "end",
                    text=item.name,
                    values=(_format_size(size), hint),
                    tags=("stream",),
                )
                self._item_action[child_id] = ("text", item)

            elif name_lower.endswith(".cfg"):
                hint = "Correction configuration"
                child_id = self.tree.insert(
                    node_id, "end",
                    text=item.name,
                    values=(_format_size(size), hint),
                    tags=("stream",),
                )
                self._item_action[child_id] = ("text", item)

            else:
                self._insert_generic_file(item, parent_id=node_id)

    # ── TIFF folder (Radiographs-* or Geometry/) ─────────────────────────────

    def _insert_tiff_folder(self, folder_path, label, tiff_role, has_positions):
        node_id = self.tree.insert(
            "", "end",
            text=label + "/",
            values=("", ""),
            open=False,
            tags=("storage",),
        )

        items = sorted(folder_path.iterdir(), key=lambda p: _natural_key([p.name]))

        # Load positions.txt if present
        pos_map   = {}
        pos_path  = folder_path / "positions.txt"
        if has_positions and pos_path.exists():
            pos_map = _parse_positions_txt(pos_path)

        # Separate out known special files from TIFFs
        tif_files   = []
        other_files = []

        for item in items:
            if _should_skip(item.name):
                continue
            nl = item.name.lower()
            if nl == "positions.txt":
                continue  # inserted separately below
            if nl.endswith(".tif") or nl.endswith(".tiff"):
                tif_files.append(item)
            else:
                other_files.append(item)

        # Insert positions.txt first if it exists
        if pos_path.exists():
            angles  = list(pos_map.values())
            n_local = len(tif_files)
            n_log   = len(pos_map)
            if angles:
                rng = f"{min(angles):.2f}\u00b0 \u2013 {max(angles):.2f}\u00b0"
                hint = f"{n_log} entries in log, {n_local} local TIFs, {rng}"
            else:
                hint = f"{n_log} entries in log, {n_local} local TIFs"
            size = pos_path.stat().st_size
            child_id = self.tree.insert(
                node_id, "end",
                text="positions.txt",
                values=(_format_size(size), hint),
                tags=("stream",),
            )
            self._item_action[child_id] = ("text", pos_path)

        # Insert other non-TIFF files
        for item in other_files:
            self._insert_generic_file(item, parent_id=node_id)

        # Insert zero_pos TIFs first
        zero_tifs  = [f for f in tif_files if f.name.lower().startswith("zero_pos")]
        radio_tifs = [f for f in tif_files if not f.name.lower().startswith("zero_pos")]

        for item in zero_tifs:
            size     = item.stat().st_size
            hint     = _tiff_hint(item, "reference radiograph")
            child_id = self.tree.insert(
                node_id, "end",
                text=item.name,
                values=(_format_size(size), hint),
                tags=("stream",),
            )
            self._item_action[child_id] = ("tiff", item)

        # Store the full sorted list of projection TIFs for "load all" option
        self._radio_paths_for_folder = radio_tifs  # temporary; overwritten per folder

        for item in radio_tifs:
            size  = item.stat().st_size
            angle = pos_map.get(item.name)
            hint  = f"angle: {angle:.2f}\u00b0" if angle is not None else "angle: unknown"
            child_id = self.tree.insert(
                node_id, "end",
                text=item.name,
                values=(_format_size(size), hint),
                tags=("stream",),
            )
            self._item_action[child_id] = ("tiff", item)

    # ── Generic folder ────────────────────────────────────────────────────────

    def _insert_generic_folder(self, folder_path):
        node_id = self.tree.insert(
            "", "end",
            text=folder_path.name + "/",
            values=("", ""),
            open=False,
            tags=("storage",),
        )
        for item in sorted(folder_path.iterdir(),
                           key=lambda p: _natural_key([p.name])):
            if _should_skip(item.name):
                continue
            self._insert_generic_file(item, parent_id=node_id)

    def _insert_generic_file(self, path, parent_id):
        size = path.stat().st_size
        nl   = path.name.lower()

        if nl.endswith(".tif") or nl.endswith(".tiff"):
            hint     = _tiff_hint(path)
            child_id = self.tree.insert(
                parent_id, "end",
                text=path.name,
                values=(_format_size(size), hint),
                tags=("stream",),
            )
            self._item_action[child_id] = ("tiff", path)
        else:
            # Guess text vs binary by extension
            text_exts = {".txt", ".cfg", ".xml", ".json", ".csv",
                         ".log", ".ini", ".rtf", ".defect", ".nsipro"}
            is_text = any(nl.endswith(e) for e in text_exts)
            hint     = "text file" if is_text else f"{path.suffix or 'binary'} file"
            child_id = self.tree.insert(
                parent_id, "end",
                text=path.name,
                values=(_format_size(size), hint),
                tags=("stream",),
            )
            if is_text:
                self._item_action[child_id] = ("text", path)

    # ── Double-click ──────────────────────────────────────────────────────────

    def _on_double_click(self, _event):
        sel = self.tree.selection()
        if not sel:
            return
        item = sel[0]
        action = self._item_action.get(item)
        if action is None:
            return

        kind, path = action

        if kind == "text":
            _open_text_viewer(self.root, path.name, path)

        elif kind == "tiff":
            self._handle_tiff_click(path)

    def _handle_tiff_click(self, path):
        """Load a TIFF (or all TIFFs in its folder) and launch slice_viewer."""
        parent_dir = path.parent

        # Gather all projection TIFs in the same folder (excluding zero_pos)
        all_tifs = sorted(
            [f for f in parent_dir.iterdir()
             if f.suffix.lower() in (".tif", ".tiff")
             and not f.name.lower().startswith("zero_pos")],
            key=lambda p: _natural_key([p.name]),
        )
        n_total = len(all_tifs)

        if path.name.lower().startswith("zero_pos") or n_total <= 1:
            paths_to_load = [path]
            title = path.name
        else:
            load_all = messagebox.askyesno(
                "Load images",
                f"This folder contains {n_total} projection images.\n\n"
                f"Load all {n_total}?  (may take a moment)\n\n"
                f"Yes = all images     No = selected image only",
            )
            if load_all:
                paths_to_load = all_tifs
                title = f"{parent_dir.name} \u2014 all {n_total} images"
            else:
                paths_to_load = [path]
                title = path.name

        try:
            n = len(paths_to_load)
            self.status.set(f"Loading {n} image{'s' if n > 1 else ''}\u2026")
            self.root.update()
            arrays = [tifffile.imread(str(p)) for p in paths_to_load]
            volume = np.stack(arrays, axis=0)   # shape (N, H, W)
        except Exception as exc:
            messagebox.showerror("Image error", f"Could not load image(s):\n{exc}")
            return

        self.status.set(
            f"Launching slice_viewer: {title}  "
            f"shape={volume.shape}  dtype={volume.dtype}"
        )
        proc = _launch_slice_viewer(volume, title)
        self._viewers.append(proc)

    # ── Close ─────────────────────────────────────────────────────────────────

    def _on_close(self):
        super()._on_close()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    root = tk.Tk()
    NSIBrowser(root)
    root.mainloop()


if __name__ == "__main__":
    main()
