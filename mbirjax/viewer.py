import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import gridspec
from matplotlib.widgets import RangeSlider, Slider, RadioButtons, CheckButtons
import warnings
import easygui
import time


# === CONSTANTS ===

# --- File picker launcher for easygui ---
def launch_file_picker():
    file_path = easygui.fileopenbox(default='*.npy', filetypes=["*.npy"])
    return file_path


TOOLTIP_FONT_SIZE = 9
TOOLTIP_BOX_ALPHA = 0.9
TOOLTIP_OFFSET = (10, 10)
TOOLTIP_TEXT = (
        "Click and drag to move" + chr(10) +
        "Click edge to resize" + chr(10) +
        "Press Esc to remove"
)

CIRCLE_COLOR = 'red'
CIRCLE_LINEWIDTH = 2
CIRCLE_ALPHA = 1.0
CIRCLE_FILL = False

SLICE_AXIS_FONT_SIZE = 9
SLICE_AXIS_LABEL_FONT_SIZE = 8
SLICE_AXIS_RADIO_SIZE = 30

TKAGG = True
Y_SKIP = 28
LOAD_LABEL = 'Load'
if matplotlib.get_backend() != 'TkAgg':
    TKAGG = False
    Y_SKIP = 50
    LOAD_LABEL = 'Load disabled - requires TkAgg'

def multiline(*lines):
    return chr(10).join(lines)


class SliceViewer:

    def __init__(self, *datasets, title='', vmin=None, vmax=None, slice_label=None,
                 slice_axis=None, cmap='gray', show_instructions=True):
        self.datasets = datasets
        self.n_volumes = len(datasets)
        self.title = title
        self.vmin = vmin
        self.vmax = vmax
        self.slice_label = slice_label
        self.slice_axis = slice_axis
        self.cmap = cmap
        self.show_instructions = show_instructions

        self.original_data = []
        self.data = []
        self.axes_perms = np.zeros(0).astype(int)
        self.labels = []
        self.cur_slices = []
        self.axes = [None] * self.n_volumes
        self.caxes = [None] * self.n_volumes
        self.images = [None] * self.n_volumes
        self.circles = []
        self.text_boxes = []
        self.axis_buttons = []

        self._normalize_inputs()
        self._prepare_data()
        self._build_figure()

        self._syncing_limits = True
        self._syncing_axes = False
        self._tk_button_pressed = False

    @staticmethod
    def _get_perm_from_slice_ind(s):
        return list({0, 1, 2} - {s}) + [s]

    def _normalize_inputs(self):
        if isinstance(self.slice_axis, int) or self.slice_axis is None:
            slice_axes = [2 if self.slice_axis is None else self.slice_axis] * self.n_volumes
        else:
            slice_axes = list(self.slice_axis)
        self._syncing_axes = all(slice_axis == slice_axes[0] for slice_axis in slice_axes)
        self.axes_perms = np.array([self._get_perm_from_slice_ind(s) for s in slice_axes]).astype(int)

        if isinstance(self.slice_label, str) or self.slice_label is None:
            self.labels = [f"Slice" if self.slice_label is None else self.slice_label
                           for i in range(self.n_volumes)]
        else:
            self.labels = list(self.slice_label)

    def _prepare_data(self):
        self.original_data = list(self.datasets)
        for i, data in enumerate(self.datasets):
            if data.ndim == 2:
                data = data[..., np.newaxis]
                self.original_data[i] = data
            elif data.ndim != 3:
                raise ValueError("Each input data must be a 2D or 3D array")
            data = np.transpose(data, self.axes_perms[i])
            self.data.append(data)

        self.cur_slices = [d.shape[2] // 2 for d in self.data]
        if self.vmin is None:
            self.vmin = np.min([np.min(d) for d in self.data])
        if self.vmax is None:
            self.vmax = np.max([np.max(d) for d in self.data])
        if self.vmin == self.vmax:
            eps = 1e-6
            scale = np.clip(eps * np.abs(self.vmax), a_min=eps, a_max=None)
            self.vmin -= scale
            self.vmax += scale

    def _build_figure(self):
        self._last_display_time = 0
        self._meshgrids = {}
        figwidth = 6 * self.n_volumes
        self.fig = plt.figure(figsize=(figwidth, 8))
        self.fig.suptitle(self.title)
        self.gs = gridspec.GridSpec(nrows=4, ncols=self.n_volumes, height_ratios=[15, 1, 1, 1])

        self._draw_images()
        self._add_slice_slider()
        self._add_intensity_slider()
        self._add_axis_controls()
        self._connect_events()
        self._resize_index = None
        self._resize_anchor = None

        if self.show_instructions:
            self.fig.text(0.01, 0.25, multiline('Press h', 'for help'), fontdict={'color': 'red'})

    def _draw_images(self, image_index=None):

        def on_xlim_changed(ax):
            def callback(lim):
                if not self._syncing_limits:
                    return
                self._syncing_limits = False
                try:
                    for other_ax in self.axes:
                        if other_ax != ax:
                            other_ax.set_xlim(ax.get_xlim())
                finally:
                    self._syncing_limits = True
                self.fig.canvas.draw_idle()

            return callback

        def on_ylim_changed(ax):
            def callback(lim):
                if not self._syncing_limits:
                    return
                self._syncing_limits = False
                try:
                    for other_ax in self.axes:
                        if other_ax != ax:
                            other_ax.set_ylim(ax.get_ylim())
                finally:
                    self._syncing_limits = True
                self.fig.canvas.draw_idle()

            return callback

        self.circles = [None] * self.n_volumes
        self.text_boxes = [None] * self.n_volumes
        indices = [image_index] if image_index is not None else range(len(self.data))
        cur_data = [self.data[image_index]] if image_index is not None else self.data
        for i, d in zip(indices, cur_data):
            if len(self.axes) > i and self.axes[i]:
                self.axes[i].remove()
                self.caxes[i].remove()
            ax = self.fig.add_subplot(self.gs[0, i])
            img = ax.imshow(d[:, :, self.cur_slices[i]], cmap=self.cmap,
                            aspect='equal',
                            vmin=self.vmin, vmax=self.vmax)
            ax.set_title(multiline(f"{self.labels[i]} {self.cur_slices[i]}",
                                   f"Shape: {self.original_data[i].shape}, Axes: {self.axes_perms[i]}"))
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            self.fig.colorbar(img, cax=cax, orientation='vertical')

            self.axes[i] = ax
            self.caxes[i] = cax
            self.images[i] = img

            # Sync zoom/pan
            ax.callbacks.connect('xlim_changed', on_xlim_changed(ax))
            ax.callbacks.connect('ylim_changed', on_ylim_changed(ax))

        # Rebuild tooltips to match updated axes
        self.tooltips = [
            ax.annotate(
                TOOLTIP_TEXT,
                xy=(0, 0), xytext=TOOLTIP_OFFSET, textcoords='offset points',
                ha='left', fontsize=TOOLTIP_FONT_SIZE,
                bbox=dict(boxstyle='round', fc='w', alpha=TOOLTIP_BOX_ALPHA),
                arrowprops=dict(arrowstyle='->'),
                visible=False
            ) for ax in self.axes
        ]

    def _add_slice_slider(self):
        ax = self.fig.add_subplot(self.gs[2, :])
        self.slice_slider = Slider(ax, label="Slice", valmin=0,
                                   valmax=max(d.shape[2] for d in self.data) - 1,
                                   valinit=self.cur_slices[0], valfmt='%0.0f')

        self.slice_slider.on_changed(self._update_slice)

    def _add_intensity_slider(self):
        ax = self.fig.add_subplot(self.gs[3, :])
        log_range = np.log10(self.vmax - self.vmin)
        digits = max(-int(np.round(log_range)) + 2, 0)
        valfmt = '%0.' + str(digits) + 'f'
        self.intensity_slider = RangeSlider(ax=ax, label="Intensity range",
                                            valmin=self.vmin, valmax=self.vmax,
                                            valinit=(self.vmin, self.vmax), valfmt=valfmt)
        self.intensity_slider.on_changed(self._update_intensity)

        def on_right_click(event):
            if event.button == 3 and event.inaxes == ax:
                title = "Adjust intensity slider range"
                msg = "Set Intensity Range"
                field_names = ["Min", "Max"]
                field_values = [f"{self.vmin:.3g}", f"{self.vmax:.3g}"]
                if TKAGG:
                    self._tk_button_pressed = True
                    inputs = easygui.multenterbox(msg + " (Cancel=previous values; leave blank=determine from data)", title, field_names, field_values)
                else:
                    warnings.warn("Right-click on intensity slider requires TkAgg: use matplotlib.use('TkAgg')")
                    inputs = None
                if inputs is None:
                    return
                self._is_drawing = False
                if not inputs[0]:
                    inputs[0] = f"{np.min([np.min(d) for d in self.data]):.3g}"
                if not inputs[1]:
                    inputs[1] = f"{np.max([np.max(d) for d in self.data]):.3g}"
                try:
                    new_min, new_max = map(float, inputs)
                    if new_min > new_max:
                        raise ValueError("Minimum must be less than maximum")
                    if new_min == new_max:
                        eps = 1e-6
                        scale = np.clip(eps * np.abs(new_max), a_min=eps, a_max=None)
                        new_min -= scale
                        new_max += scale
                    self.vmin, self.vmax = new_min, new_max
                    self.intensity_slider.valmin = self.vmin
                    self.intensity_slider.valmax = self.vmax
                    self.intensity_slider.ax.set_xlim(self.vmin, self.vmax)
                    self.intensity_slider.set_val((self.vmin, self.vmax))
                    self._update_intensity((self.vmin, self.vmax))
                    self.fig.canvas.draw_idle()
                except Exception as e:
                    easygui.msgbox(str(e), title="Invalid Input")

        self.fig.canvas.mpl_connect('button_press_event', on_right_click)

    def _add_axis_controls(self):
        self.axis_buttons = []
        all_same = all(ax_ind == self.axes_perms[0, -1] for ax_ind in self.axes_perms[:, -1])

        if self.n_volumes == 1:
            self._create_axis_buttons(False)
        else:

            self._create_axis_buttons(not all_same)

    def _create_axis_buttons(self, decoupled):
        for b in self.axis_buttons:
            b.ax.remove()
        self.axis_buttons.clear()

        if not decoupled:
            ax = self.fig.add_subplot(self.gs[1, 0])
            ax.set_title("Slice axis", loc='left', fontsize=SLICE_AXIS_FONT_SIZE)
            btns = RadioButtons(ax, labels=["0", "1", "2"], radio_props={'s': [SLICE_AXIS_RADIO_SIZE]})
            for lbl in btns.labels: lbl.set_fontsize(SLICE_AXIS_LABEL_FONT_SIZE)
            btns.set_active(int(self.axes_perms[0, -1]))
            for i in range(self.n_volumes):
                self._update_axis(i, int(self.axes_perms[0, -1]))
            btns.on_clicked(lambda label: [self._update_axis(i, int(label)) for i in range(self.n_volumes)])
            self.axis_buttons.append(btns)
        else:
            for i in range(self.n_volumes):
                ax = self.fig.add_subplot(self.gs[1, i])
                ax.set_title("Slice axis", loc='left', fontsize=SLICE_AXIS_FONT_SIZE)
                btns = RadioButtons(ax, labels=["0", "1", "2"], radio_props={'s': [30]})
                for lbl in btns.labels: lbl.set_fontsize(8)
                btns.set_active(int(self.axes_perms[i, -1]))
                btns.on_clicked(lambda label, index=i: self._update_axis(index, int(label)))
                self.axis_buttons.append(btns)

    def _toggle_decouple_slice_axes(self):
        self._syncing_axes = not self._syncing_axes
        self._create_axis_buttons(self._syncing_axes)
        self._update_slice_slider()
        self.fig.canvas.draw_idle()

    def _toggle_sync_limits(self):
        self._syncing_limits = not self._syncing_limits
        self.fig.canvas.draw_idle()

    def _update_slice_slider(self):
        # Update slice slider max if any volume changed depth
        new_max = max(d.shape[2] for d in self.data) - 1
        self.slice_slider.valmax = new_max
        self.slice_slider.ax.set_xlim(self.slice_slider.valmin, new_max)
        self.slice_slider.set_val(self.cur_slices[0])
        self.fig.canvas.draw_idle()

    def _update_axis(self, i, new_perm):
        if isinstance(new_perm, (list, tuple, np.ndarray)):
            new_perm = list(new_perm)
        else:
            new_perm = self._get_perm_from_slice_ind(new_perm)

        if new_perm != list(self.axes_perms[i]):
            orig = self.original_data[i]
            prev_slice_axis = self.axes_perms[i, -1]
            prev_fraction = self.cur_slices[i] / orig.shape[prev_slice_axis]
            new_slice_axis = new_perm[-1]
            self.axes_perms[i] = np.array(new_perm)
            new_data = np.transpose(orig, new_perm)
            self.data[i] = new_data
            self.cur_slices[i] = int(np.round(prev_fraction * orig.shape[new_slice_axis]))
            self.images[i].set_data(new_data[:, :, self.cur_slices[i]])
            self.axes[i].set_xlim(0, new_data.shape[1])
            self.axes[i].set_ylim(new_data.shape[0], 0)
            self.axes[i].set_aspect('equal')
            self.axes[i].set_title(multiline(f"{self.labels[i]} {self.cur_slices[i]}",
                                             f"Shape: {orig.shape}, Axes: {self.axes_perms[0]}"))  # f"Shape: {orig.shape}"))
            self._draw_images()
            self._update_slice_slider()
            plt.tight_layout()
            self.fig.canvas.draw_idle()

    def _update_slice(self, val):
        new_slice = int(round(val))
        for i, d in enumerate(self.data):
            frac = new_slice / d.shape[2]
            idx = int(round(frac * d.shape[2]))
            idx = np.clip(idx, 0, d.shape[2] - 1)
            self.cur_slices[i] = idx
            self.images[i].set_data(d[:, :, idx])
            self.axes[i].set_title(multiline(f"{self.labels[i]} {idx}",
                                             f"Shape: {self.original_data[i].shape}, Axes: {self.axes_perms[i]}"))  #
        self._display_mean()
        self.fig.canvas.draw_idle()

    def _update_intensity(self, val):
        for img in self.images:
            img.set_clim(val[0], val[1])
        self.fig.canvas.draw_idle()

    def _get_mask(self, data, x, y, r):
        shape = data.shape
        if shape not in self._meshgrids:
            ny, nx = shape[:2]
            self._meshgrids[shape] = np.meshgrid(np.arange(nx), np.arange(ny))
        xv, yv = self._meshgrids[shape]
        return (xv - x) ** 2 + (yv - y) ** 2 <= r ** 2

    def _display_mean(self):
        if all(c is None for c in self.circles):
            return
        now = time.time()
        if now - self._last_display_time < 0.3:
            return
        self._last_display_time = now

        for i, circle in enumerate(self.circles):
            if circle is None or self._is_drawing or self._is_moving:
                self.tooltips[i].set_visible(False)
            x, y = circle.center
            r = circle.get_radius()
            mask = self._get_mask(self.data[i][:, :, self.cur_slices[i]], x, y, r)
            values = self.data[i][:, :, self.cur_slices[i]][mask]
            if values.size:
                mean, std = np.mean(values), np.std(values)
                cur_min, cur_max = np.min(values), np.max(values)
                if self.text_boxes[i]:
                    self.text_boxes[i].remove()
                self.text_boxes[i] = self.axes[i].text(0.05, 0.95,
                                                       multiline(f"(µ, σ)=({mean:.3g}, {std:.3g})",
                                                                 f"(min, max)=({cur_min:.3g}, {cur_max:.3g})"),
                                                       transform=self.axes[i].transAxes,
                                                       fontsize=12, va='top', bbox=dict(facecolor='white', alpha=1.0))
        self.fig.canvas.draw_idle()

    def _remove_graphics(self):
        for item in self.circles + self.text_boxes:
            if item is not None:
                item.remove()
        for tip in getattr(self, 'tooltips', []):
            tip.set_visible(False)
        self.circles = [None] * self.n_volumes
        self.text_boxes = [None] * self.n_volumes

    def _create_circle(self, center, radius):
        return plt.Circle(center, radius, color=CIRCLE_COLOR, lw=CIRCLE_LINEWIDTH, fill=CIRCLE_FILL, alpha=CIRCLE_ALPHA)

    def _connect_events(self):
        self._menu_texts = []

        def show_help(show):
            if show:
                help_items = ["Right-click an image for menu", 'Click and drag for ROI']
                if TKAGG:
                    help_items += ['Right-click intensity slider to adjust range']
                help_items += ['Press [esc] to remove ROI/menu/help']
                self.help_overlay = self.fig.text(0.25, 0.5,
                                                  multiline(*help_items),
                                                  ha='left', va='center', fontsize=12,
                                                  bbox=dict(facecolor='white', alpha=0.9), zorder=10)
            else:
                if hasattr(self, 'help_overlay') and self.help_overlay:
                    self.help_overlay.remove()
                    self.help_overlay = None
            self.fig.canvas.draw_idle()

        self._is_moving = False
        self._move_offset = None
        self.tooltips = [
            ax.annotate(
                TOOLTIP_TEXT,
                xy=(0, 0), xytext=TOOLTIP_OFFSET, textcoords='offset points',
                ha='left', fontsize=TOOLTIP_FONT_SIZE,
                bbox=dict(boxstyle='round', fc='w', alpha=TOOLTIP_BOX_ALPHA),
                arrowprops=dict(arrowstyle='->'),
                visible=False
            ) for ax in self.axes
        ]
        self._drag_start = None
        self._is_drawing = False  # Add a flag to track drawing state

        def on_press(event):
            show_help(False)
            if event.button != 1: return
            if event.inaxes not in self.axes: return
            if hasattr(event,
                       'menu_option_selected') and event.menu_option_selected: return  # This is set to true when the user selects a menu option.
            if hasattr(self.fig.canvas, 'toolbar') and getattr(self.fig.canvas.toolbar, 'mode', ''): return
            if self._tk_button_pressed:
                if event.button == 1:  # Consume only a left-click
                    self._tk_button_pressed = False
                    return

            self._resize_index = None

            for i, circle in enumerate(self.circles):
                if circle is None: continue
                x, y = circle.center
                r = circle.get_radius()
                if event.xdata is not None and event.ydata is not None:
                    dx = event.xdata - x
                    dy = event.ydata - y
                    dist = np.sqrt(dx ** 2 + dy ** 2)
                    if abs(dist - r) <= 0.1 * r:
                        self._resize_index = i
                        self._resize_anchor = (x, y)
                        return
                    elif dx ** 2 + dy ** 2 <= r ** 2:
                        self._is_moving = True
                        self._move_offset = (dx, dy)
                        return

            self._remove_graphics()
            self._is_drawing = True
            self._drag_start = (event.xdata, event.ydata)
            for i, ax in enumerate(self.axes):
                circ = self._create_circle((event.xdata, event.ydata), 0)
                self.circles[i] = circ
                ax.add_patch(circ)
            self.fig.canvas.draw_idle()

        def on_motion(event):
            if event.inaxes not in self.axes: return
            if hasattr(self.fig.canvas, 'toolbar') and getattr(self.fig.canvas.toolbar, 'mode', ''): return
            if not self._is_drawing: return

            for i, circle in enumerate(self.circles):
                if circle is None: continue
                x, y = circle.center
                r = circle.get_radius()
                if event.xdata is not None and event.ydata is not None:
                    dx = event.xdata - x
                    dy = event.ydata - y
                    dist2 = dx ** 2 + dy ** 2
                    if not (self._is_drawing or self._is_moving or self._resize_index is not None):
                        if dist2 <= (1.1 * r) ** 2:
                            self.tooltips[i].xy = self.circles[i].center
                            self.tooltips[i].set_visible(True)
                        else:
                            self.tooltips[i].set_visible(False)
                    else:
                        self.tooltips[i].set_visible(False)

            if self._is_drawing and self._drag_start is not None:
                x0, y0 = self._drag_start
                r = np.sqrt((event.xdata - x0) ** 2 + (event.ydata - y0) ** 2)
                for i, ax in enumerate(self.axes):
                    if self.circles[i]:
                        self.circles[i].remove()
                    circ = self._create_circle((x0, y0), r)
                    self.circles[i] = circ
                    ax.add_patch(circ)
                self._display_mean()

            elif getattr(self, '_is_moving', False):
                dx, dy = self._move_offset
                new_center = (event.xdata - dx, event.ydata - dy)
                for j in range(self.n_volumes):
                    self.circles[j].center = new_center
                self._display_mean()

            elif self._resize_index is not None and self._resize_anchor is not None:
                x0, y0 = self._resize_anchor
                new_radius = np.sqrt((event.xdata - x0) ** 2 + (event.ydata - y0) ** 2)
                for j in range(self.n_volumes):
                    self.circles[j].set_radius(new_radius)
                self._display_mean()

            self.fig.canvas.draw_idle()

        def on_release(event):
            self._is_moving = False
            self._move_offset = None
            self._resize_index = None
            self._resize_anchor = None
            self._is_drawing = False
            self._drag_start = None
            self._display_mean()

        def on_key(event):
            if event.key == 'h':
                show_help(True)
            elif event.key == 'escape':
                show_help(False)
                self._remove_menu()
                self._is_drawing = False
                for i in range(self.n_volumes):
                    if self.text_boxes[i] is not None:
                        self.text_boxes[i].remove()
                self.text_boxes = [None] * self.n_volumes
                self._remove_graphics()
                self.circles = [None] * self.n_volumes
                self._display_mean()
                self.fig.canvas.draw_idle()

        def on_context_menu(event):
            if event.button == 3 and event.inaxes in self.axes:
                i = self.axes.index(event.inaxes)
                # Remove any existing menu items
                self._remove_menu()

                def make_option(label, y_offset, callback):
                    bounds = self.fig.bbox.bounds
                    x_frac = (event.x - bounds[0]) / (bounds[2] - bounds[0])
                    y_frac = (event.y + y_offset - bounds[1]) / (bounds[3] - bounds[1])
                    txt = self.fig.text(
                        x_frac, y_frac,
                        label,
                        ha='left', va='bottom', fontsize=10, color='white',
                        bbox=dict(facecolor='black', edgecolor='white')
                    )
                    self._menu_texts.append(txt)
                    txt._viewer_callback = callback

                options = self._get_context_menu_options(i)
                y_offset = 0
                y_skip = Y_SKIP
                for option in options:
                    make_option(option[0], y_offset, option[1])
                    y_offset -= y_skip
                self.fig.canvas.draw_idle()

        def on_context_select(event):
            if event.button != 1:
                return
            clicked = False
            for txt in self._menu_texts:
                bbox = txt.get_window_extent()
                if bbox.contains(event.x, event.y):
                    clicked = True
                    event.menu_option_selected = True
                    if hasattr(txt, '_viewer_callback'):
                        txt._viewer_callback()
            self._remove_menu()
            return clicked

        self.fig.canvas.mpl_connect('button_press_event', on_context_menu)
        self.fig.canvas.mpl_connect('button_press_event', on_context_select)

        self.fig.canvas.mpl_connect('button_press_event', on_press)
        self.fig.canvas.mpl_connect('motion_notify_event', on_motion)
        self.fig.canvas.mpl_connect('button_release_event', on_release)
        self.fig.canvas.mpl_connect('key_press_event', on_key)

    def _get_context_menu_options(self, i):
        options = []
        if self.n_volumes > 1:
            options.append([
                "{} slice axes".format("Decouple" if all(
                    ax == self.axes_perms[0, -1] for ax in self.axes_perms[:, -1]) else "Couple"),
                self._toggle_decouple_slice_axes])
            options.append([
                "{} pan/zoom".format("Decouple" if self._syncing_limits else "Couple"),
                self._toggle_sync_limits])
        options += [["Transpose image", lambda: self._on_transpose(i)],
                    [LOAD_LABEL, lambda: self._on_load(i)],
                    ["Reset", lambda: self._on_reset(i)],
                    ["Cancel", self._remove_menu]]
        return options

    def _on_transpose(self, i):
        perm = self.axes_perms[i].copy()
        perm[0], perm[1] = perm[1], perm[0]
        self._update_axis(i, perm)
        self._remove_menu()

    def _on_load(self, i):
        if not hasattr(self.fig.canvas.manager, 'window'):
            warnings.warn("Load disabled: matplotlib backend is not TkAgg")
            return

        self._remove_menu()

        def deferred_load():
            file_path = launch_file_picker()
            if not file_path:
                return
            try:
                new_array = np.load(file_path)
                if new_array.ndim == 2:
                    new_array = new_array[..., np.newaxis]
                if new_array.ndim != 3:
                    raise ValueError("Loaded array must be 2D or 3D")

                self.original_data[i] = new_array
                self.axes_perms[i] = self._get_perm_from_slice_ind(self.axes_perms[i][-1])
                transposed = np.transpose(new_array, self.axes_perms[i])
                self.data[i] = transposed
                self.cur_slices[i] = transposed.shape[2] // 2
                self._draw_images()
                self._update_slice_slider()
                self.fig.canvas.draw_idle()
            except Exception as e:
                print(f"Failed to load array: {e}")

        # Use TkAgg-safe scheduling
        try:
            self.fig.canvas.manager.window.after(10, deferred_load)
        except Exception as e:
            warnings.warn("Unable to load file.  Use matplotlib.use('TkAgg') to enable file load.")

    def _on_reset(self, i):
        self._draw_images(i)
        self._update_slice_slider()
        plt.tight_layout()
        self.fig.canvas.draw_idle()

        self._remove_menu()
    def _remove_menu(self):
        if self._menu_texts:
            for txt in self._menu_texts:
                txt.remove()
            self._menu_texts.clear()
            self.fig.canvas.draw_idle()

    def show(self):
        plt.tight_layout()
        plt.show()


def slice_viewer(*datasets, **kwargs):
    text = ['Line {}'.format(j) for j in range(100)]
    text = multiline(*text)
    easygui.textbox(msg='msg', title='title', text=text, codebox=False, callback=None, run=True)
    viewer = SliceViewer(*datasets, **kwargs)
    viewer.show()
