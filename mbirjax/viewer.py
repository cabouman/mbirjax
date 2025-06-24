import os
import warnings
import matplotlib

import mbirjax as mj

# Set backend
if os.environ.get("READTHEDOCS") == "True":
    matplotlib.use('Agg')
else:
    try:
        matplotlib.use('TkAgg')
    except ImportError:
        matplotlib.use('Agg')  # Fallback
        warnings.warn("TkAgg not available. Falling back to Agg.")

# Now it's safe to import pyplot
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import gridspec
from matplotlib.widgets import RangeSlider, Slider, RadioButtons, CheckButtons
import time
import h5py
import easygui

# === CONSTANTS ===
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

VALMAX_EPS = 1e-6

TKAGG = True
Y_SKIP = 28
LOAD_LABEL = 'Load'
SAVE_LABEL = 'Save data to h5'
if matplotlib.get_backend() != 'TkAgg':
    TKAGG = False
    Y_SKIP = 50
    LOAD_LABEL = 'Load disabled - requires TkAgg'
    SAVE_LABEL = 'Save disabled - requires TkAgg'


def multiline(*lines):
    return chr(10).join(lines)


class SliceViewer:
    """
    Interactive multi-volume slice viewer for 2D and 3D NumPy arrays using matplotlib.

    This class provides a graphical interface for exploring one or more 3D volumes or 2D slices.
    Features include synchronized slice navigation, ROI statistics, axis transposition, file loading,
    dynamic intensity range adjustment, and interactive GUI tools for zooming and panning.

    Designed primarily for inspecting CT or other volumetric reconstructions in research workflows.

    Args:
        *datasets (ndarray or None): One or more 2D or 3D NumPy arrays to display.
            - 2D arrays are automatically promoted to 3D via a singleton axis.
            - `None` values are replaced with placeholder zero arrays.

        data_dicts (None or dict or list of None or dicts, optional): Dictionary of string entries to associated with the data (e.g., from :meth:`get_recon_dict`)
        title (str, optional): Window title. Defaults to an empty string.
        vmin (float, optional): Minimum intensity value for display. Defaults to the global minimum across all datasets.
        vmax (float, optional): Maximum intensity value for display. Defaults to the global maximum across all datasets.
        slice_label (str or list of str, optional): Label(s) for the current slice. Defaults to "Slice".
        slice_axis (int or list of int, optional): Axis along which to slice (0, 1, or 2). Defaults to the last axis (2).
        cmap (str, optional): Colormap to use. Defaults to "gray".
        show_instructions (bool, optional): Whether to display usage instructions in the figure. Defaults to True.

    Notes:
        - Right-click an image to access a context menu with options such as axis transposition and file loading.
        - Right-click the intensity slider (if using TkAgg backend) to manually set display range bounds.
        - Press 'h' to show help overlay. Press 'Esc' to clear overlays or reset ROI selections.
    """

    def __init__(self, *datasets, data_dicts=None, title='', vmin=None, vmax=None, slice_label=None,
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
        if data_dicts is None:
            self.data_dicts = [None] * self.n_volumes
        else:
            if isinstance(data_dicts, dict):
                data_dicts = [data_dicts]
            if len(data_dicts) == self.n_volumes and all([isinstance(d, dict) or d is None for d in data_dicts]):
                self.data_dicts = [mj.TomographyModel.convert_subdicts_to_strings(d) for d in data_dicts]
            else:
                raise ValueError('data_dicts must be single dict or a list of dicts of the same length as the number of datasets')

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
        self._image_selection_dict = {}
        self._clear_image_selection()
        self._difference_image_dicts = [None] * self.n_volumes  # Entries:  comparison_index, use_abs, prev_label
        easygui.boxes.global_state.prop_font_line_length = 80
        self.show()

    def _clear_image_selection(self):
        self._image_selection_dict = {'selecting_image': False, 'baseline_index': -1}

    @staticmethod
    def _get_perm_from_slice_ind(s):
        return list({0, 1, 2} - {s}) + [s]

    def _normalize_inputs(self):
        # Set up the slice axes and labels
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
        # Promote arrays to 3D if needed and find max and min
        self.original_data = list(self.datasets)
        for i, data in enumerate(self.datasets):
            if data is None:
                data = np.zeros((20, 20, 20))
                self.original_data[i] = data
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
        # Construct the figure to display the images and tools
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
        # Set up the components to display and some of the functionality

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

        self._remove_graphics()
        indices = [image_index] if image_index is not None else range(len(self.data))
        cur_data = [self.data[image_index]] if image_index is not None else self.data

        # Draw the images, titles, colorbars
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
            ax.zorder = 2
            cax.zorder = 1
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
        valmax = max(d.shape[2] for d in self.data) - 1
        valmax = max(valmax, VALMAX_EPS)
        self.slice_slider = Slider(ax, label="Slice", valmin=0,
                                   valmax=valmax,
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
            # Handle a right-click on the intensity slider to change the upper and lower bounds.
            # This is enabled only when TkAgg is available.
            if event.button == 3 and event.inaxes == ax:
                title = "Adjust intensity slider range"
                msg = "Set intensity range:"
                field_names = ["Min", "Max"]
                field_values = [f"{self.vmin:.3g}", f"{self.vmax:.3g}"]
                if TKAGG:
                    self._tk_button_pressed = True  # This is designed to avoid interpreting closing this window for starting a new ROI
                    inputs = easygui.multenterbox(
                        multiline(msg, "  Cancel=use previous values", "  Leave blank=determine from data"),
                        title, field_names, field_values)
                else:
                    warnings.warn("Right-click on intensity slider requires TkAgg: use matplotlib.use('TkAgg')")
                    inputs = None
                if inputs is None:
                    return
                self._is_drawing = False
                # If the inputs are blank, then determine the min and max from data.
                if not inputs[0]:
                    inputs[0] = f"{np.min([np.min(d) for d in self.data]):.3g}"
                if not inputs[1]:
                    inputs[1] = f"{np.max([np.max(d) for d in self.data]):.3g}"
                try:
                    # Set the new min and max from inputs
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
        # This is the radio button control to select the slice axis
        self.axis_buttons = []
        all_same = all(ax_ind == self.axes_perms[0, -1] for ax_ind in self.axes_perms[:, -1])

        if self.n_volumes == 1:
            self._create_axis_buttons(False)
        else:

            self._create_axis_buttons(not all_same)

    def _create_axis_buttons(self, decoupled):
        # If the image axes are coupled, then we have only one radio button, otherwise one for each.
        for b in self.axis_buttons:
            b.ax.remove()
        self.axis_buttons.clear()

        if not decoupled:
            ax = self.fig.add_subplot(self.gs[1, 0])
            ax.set_title("Slice axis", loc='left', fontsize=SLICE_AXIS_FONT_SIZE)
            btns = RadioButtons(ax, labels=["0", "1", "2"], radio_props={'s': [SLICE_AXIS_RADIO_SIZE]})
            for lbl in btns.labels:
                lbl.set_fontsize(SLICE_AXIS_LABEL_FONT_SIZE)
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
                for lbl in btns.labels:
                    lbl.set_fontsize(8)
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
        new_max = max(new_max, VALMAX_EPS)
        self.slice_slider.valmax = new_max
        self.slice_slider.ax.set_xlim(self.slice_slider.valmin, new_max)
        self.slice_slider.set_val(self.cur_slices[0])
        self.fig.canvas.draw_idle()

    def _update_axis(self, i, new_perm):
        # If the order of axes changed for this image, then update the data and redraw.
        if isinstance(new_perm, (list, tuple, np.ndarray)):
            new_perm = list(new_perm)
        else:
            new_perm = self._get_perm_from_slice_ind(new_perm)

        if new_perm != list(self.axes_perms[i]):
            new_data = self.data[i]
            prev_fraction = self.slice_slider.val / (self.slice_slider.valmax + VALMAX_EPS)
            new_slice_axis = new_perm[-1]
            inverse_perm = np.argsort(self.axes_perms[i])
            new_data = np.transpose(new_data, inverse_perm)
            new_num_slices = new_data.shape[new_slice_axis]
            self.axes_perms[i] = np.array(new_perm)
            new_data = np.transpose(new_data, new_perm)
            self.data[i] = new_data
            self.cur_slices[i] = int(np.round(prev_fraction * (new_num_slices - 1)))
            self.images[i].set_data(new_data[:, :, self.cur_slices[i]])
            self.axes[i].set_xlim(0, new_data.shape[1])
            self.axes[i].set_ylim(new_data.shape[0], 0)
            self.axes[i].set_aspect('equal')
            self.axes[i].set_title(multiline(f"{self.labels[i]} {self.cur_slices[i]}",
                                             f"Shape: {new_data.shape}, Axes: {self.axes_perms[0]}"))  # f"Shape: {orig.shape}"))
            self._draw_images()
            self._update_slice_slider()
            self._update_intensity(self.intensity_slider.val)
            plt.tight_layout()
            self.fig.canvas.draw_idle()

    def _update_slice(self, val):
        # Update the image to display the chosen slice.
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
        # Update the intensity range.
        for img in self.images:
            img.set_clim(val[0], val[1])
        self.fig.canvas.draw_idle()

    def _get_mask(self, data, x, y, r):
        # Get a mask for the ROI
        shape = data.shape
        if shape not in self._meshgrids:
            ny, nx = shape[:2]
            self._meshgrids[shape] = np.meshgrid(np.arange(nx), np.arange(ny))
        xv, yv = self._meshgrids[shape]
        return (xv - x) ** 2 + (yv - y) ** 2 <= r ** 2

    def _display_mean(self):
        # Show the stats associated with an ROI
        if all(c is None for c in self.circles):
            return
        now = time.time()
        if now - self._last_display_time < 0.3:  # Delay each update to avoid lag for large volumes
            return
        self._last_display_time = now

        # Get the stats for the circle in each image.
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
        # Remove the ROI, tooltips, etc.
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
        # Set up the main event handling routines.
        self._menu_texts = []
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
            if event.button != 1:
                return
            if event.inaxes not in self.axes:
                return
            if hasattr(event, 'menu_option_selected') and event.menu_option_selected:
                return  # This is set to true when the user selects a menu option.
            if hasattr(self.fig.canvas, 'toolbar') and getattr(self.fig.canvas.toolbar, 'mode', ''):
                return
            if self._tk_button_pressed:  # Skip a press if the user just chose a menu item
                if event.button == 1:  # Consume only a left-click
                    self._tk_button_pressed = False
                    return
            if self._image_selection_dict['selecting_image']:
                image_index = self.axes.index(event.inaxes)
                self._on_difference(image_index)
                return

            self._resize_index = None

            # Handle the case of resizing or moving an ROI circle
            for i, circle in enumerate(self.circles):
                if circle is None:
                    continue
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

            # Otherwise start drawing an ROI circle
            self._remove_graphics()
            self._is_drawing = True
            self._drag_start = (event.xdata, event.ydata)
            for i, ax in enumerate(self.axes):
                circ = self._create_circle((event.xdata, event.ydata), 0)
                self.circles[i] = circ
                ax.add_patch(circ)
            self.fig.canvas.draw_idle()

        def on_motion(event):
            if event.inaxes not in self.axes:
                return
            if hasattr(self.fig.canvas, 'toolbar') and getattr(self.fig.canvas.toolbar, 'mode', ''):
                return

            # Check for motion in or out of an ROI and display a tooltip as needed.
            for i, circle in enumerate(self.circles):
                if circle is None:
                    continue
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

            # Update the ROI circle and stats if needed.
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
            # Reset any items from a mouse drag
            self._is_moving = False
            self._move_offset = None
            self._resize_index = None
            self._resize_anchor = None
            self._is_drawing = False
            self._drag_start = None
            self._display_mean()

        def on_key(event):
            # Handle any key press
            if event.key == 'h':
                self._show_message(True, message_type='help')
            elif event.key == 'escape':
                # Remove and reset as needed
                self._show_message(False)
                self._remove_menu()
                self._is_drawing = False
                if self._image_selection_dict['selecting_image']:
                    image_index = self._image_selection_dict['baseline_index']
                    self._difference_image_dicts[image_index] = None
                    self._clear_image_selection()
                for i in range(self.n_volumes):
                    if self.text_boxes[i] is not None:
                        self.text_boxes[i].remove()
                self.text_boxes = [None] * self.n_volumes
                self._remove_graphics()
                self.circles = [None] * self.n_volumes
                self._display_mean()
                self.fig.canvas.draw_idle()

        def on_context_menu(event):
            if self._image_selection_dict['selecting_image']:
                return
            # Start the menu for an image
            if event.button == 3 and event.inaxes in self.axes:
                image_index = self.axes.index(event.inaxes)
                # Remove any existing menu items
                self._remove_menu()
                self._render_menu(event, image_index)

        def on_context_select(event):
            # Handle the image menu selection
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

    def _get_context_menu_options(self, image_index):
        # Define the image menu options, taking care that some are available only with TkAgg
        options = []
        if self.n_volumes > 1:
            options.append([
                "{} slice axes".format("Decouple" if all(
                    ax == self.axes_perms[0, -1] for ax in self.axes_perms[:, -1]) else "Couple"),
                self._toggle_decouple_slice_axes])
            options.append([
                "{} pan/zoom".format("Decouple" if self._syncing_limits else "Couple"),
                self._toggle_sync_limits])
            if self._difference_image_dicts[image_index] is None:
                options.append(["Replace with difference image", lambda: self._on_difference(image_index)])
                options.append(["Replace with error image", lambda: self._on_difference(image_index, use_abs=True)])
            else:
                options.append(['Restore original image', lambda: self._on_restore(image_index)])
        if TKAGG:
            options += [["Show data dict", lambda: self._on_show_data_dict(image_index)]]
        options += [
            ["Transpose image", lambda: self._on_transpose(image_index)],
            [LOAD_LABEL, lambda: self._on_load(image_index)],
            [SAVE_LABEL, lambda: self._on_save(image_index)],
            ["Reset", lambda: self._on_reset(image_index)],
            ["Cancel", self._remove_menu]
        ]
        return options

    def _on_show_data_dict(self, image_index):
        # Handle the display of image data_dict
        data_dict = self.data_dicts[image_index]
        if not data_dict:
            easygui.textbox(
                msg='No data dict available',
                title='No data dict',
                text='No data dict included with this image',
                codebox=False,
                callback=None,
                run=True
            )
            return

        names = list(data_dict.keys())
        if len(data_dict) > 1:
            choices = names + ['Cancel']
            msg = ['Choose an entry to display:', ' '] + names
            choice = easygui.buttonbox(
                msg=multiline(*msg),
                title='Choose an entry to display',
                choices=choices
            )
            if choice == 'Cancel':
                return
        else:
            choice = names[0]

        index = names.index(choice)
        dict_entry = data_dict[names[index]]
        easygui.textbox(
            msg=f'Dict entry = {choice}',
            title=choice,
            text=dict_entry,
            codebox=False,
            callback=None,
            run=True
        )

    def _on_transpose(self, image_index):
        # Transpose the image
        perm = self.axes_perms[image_index].copy()
        perm[0], perm[1] = perm[1], perm[0]
        self._update_axis(image_index, perm)
        self._update_intensity(self.intensity_slider.val)
        self._remove_menu()

    def _on_load(self, image_index):
        # Handle loading a file.  The uses a delay in launching the gui to avoid oddities with tk.
        if not hasattr(self.fig.canvas.manager, 'window'):
            warnings.warn("Load disabled: matplotlib backend is not TkAgg")
            return

        self._remove_menu()

        def deferred_load():
            file_path = easygui.fileopenbox(default='*.h5', filetypes=["*.npy", "*.npz", "*.h5", "*.hdf5"])
            if not file_path:
                return
            try:
                self._load_file(file_path, image_index)
            except Exception as e:
                self._show_message(True, message=f"Failed to load file: {e}. Press Esc to exit.")

        # Use TkAgg-safe scheduling to delay the gui launch.
        try:
            self.fig.canvas.manager.window.after(10, deferred_load)
        except Exception as e:
            warnings.warn("Unable to load file.  Use matplotlib.use('TkAgg') to enable file load.")

    def _on_save(self, image_index):
        # Handle saving the full 3D volume and optional data_dict
        if not hasattr(self.fig.canvas.manager, 'window'):
            warnings.warn("Save disabled: matplotlib backend is not TkAgg")
            return

        self._remove_menu()

        def deferred_save():
            # Prompt for filename
            file_path = easygui.filesavebox(
                msg="Choose filename (must end in .h5)",
                default="volume",
                # filetypes=["*.h5"]
            )
            if not file_path:
                return
            if not file_path.lower().endswith(".h5"):
                file_path += ".h5"

            # Gather allowable data_dict with multi-line textboxes and prepopulate existing values
            # Existing data_dict loaded earlier (if any)
            if self.data_dicts[image_index] is not None:
                data_dict = self.data_dicts[image_index]
            else:
                data_dict = {}
            for key in {"model_params", "notes", "recon_log", "recon_params"}.union(data_dict.keys()):
                default_val = data_dict.get(key, "")
                val = easygui.textbox(
                    msg=f"Enter value for {key} (leave blank for none):",
                    title=key,
                    text=default_val
                )
                # If user cancelled, keep existing text; if cleared, treat as empty
                if val is None:
                    val = default_val
                data_dict[key] = val

            self.data_dicts[image_index] = data_dict

            # Save full 3D volume
            data = self.original_data[image_index]
            mj.save_data_hdf5(file_path, data, 'volume', data_dict)

            easygui.msgbox(f"Saved to {file_path}", title="Save complete")

        # Schedule the save dialog after a short delay (TkAgg-safe)
        try:
            self.fig.canvas.manager.window.after(10, deferred_save)
        except Exception:
            warnings.warn("Unable to schedule save dialog. Requires TkAgg backend.")

    def _on_difference(self, image_index: int, use_abs: bool = False):

        self._remove_menu()
        # Start the selection process for this baseline
        if not self._image_selection_dict['selecting_image']:
            self._image_selection_dict['selecting_image'] = True
            self._image_selection_dict['baseline_index'] = image_index
            self._difference_image_dicts[image_index] = {'use_abs': use_abs}
            if self.n_volumes > 2:
                # Show instructions
                self._show_message(True, message_type='difference')
                # Return to await selection or Esc
                return
            else:
                image_index = 1 - image_index  # There are only 2 images, so choose the other, then continue below with setting the data

        # Save this index as the comparison and set up the plot
        if self._image_selection_dict['selecting_image']:
            # Check the shape and axes perm of the two images
            baseline_index = self._image_selection_dict['baseline_index']
            comparison_index = image_index
            if any(a != b for a, b in
                   zip(self.original_data[baseline_index].shape, self.original_data[comparison_index].shape)) or \
                    any(a != b for a, b in zip(self.axes_perms[baseline_index], self.axes_perms[comparison_index])) or \
                    baseline_index == comparison_index:
                # Show instructions
                self._show_message(True, message_type='difference')
                return  # The difference is not valid, so exit

            # Otherwise exit the selection process and set the current index as the comparison
            self._show_message(False)
            self._clear_image_selection()
            self._difference_image_dicts[baseline_index]['comparison_index'] = comparison_index

            # Set up the difference image
            difference = self.data[comparison_index] - self.data[baseline_index]
            if self._difference_image_dicts[baseline_index]['use_abs']:
                difference = np.abs(difference)
            self.data[baseline_index] = difference
            self.images[baseline_index].set_data(difference[:, :, self.cur_slices[baseline_index]])

            # Adjust the labels and refresh the display
            self._difference_image_dicts[baseline_index]['prev_label'] = self.labels[baseline_index]
            if self._difference_image_dicts[baseline_index]['use_abs']:
                label_prepend = 'abs(Image {} minus current): '.format(comparison_index)
            else:
                label_prepend = 'Image {} minus current: '.format(comparison_index)
            self.labels[baseline_index] = label_prepend + self.labels[baseline_index]
            self._update_slice_slider()
            self.fig.canvas.draw_idle()

    def _on_restore(self, image_index):
        # Restore the original image
        original_data = self.original_data[image_index]
        self.data[image_index] = np.transpose(original_data, self.axes_perms[image_index])
        self.images[image_index].set_data(self.data[image_index][:, :, self.cur_slices[image_index]])
        if self._difference_image_dicts[image_index] is not None:
            self.labels[image_index] = self._difference_image_dicts[image_index]['prev_label']
        self._difference_image_dicts[image_index] = None
        self._update_slice_slider()
        self.fig.canvas.draw_idle()

    def _on_reset(self, image_index):
        # Do a hard reset
        if self._syncing_limits:
            for i in range(self.n_volumes):
                self._draw_images(i)
        else:
            self._draw_images(image_index)
        self._update_slice_slider()
        self._update_intensity(self.intensity_slider.val)
        plt.tight_layout()
        self.fig.canvas.draw_idle()
        self._clear_image_selection()
        self._show_message(False)
        self._remove_menu()

    def _show_message(self, show, message_type=None, message=None):
        # Clear any existing message
        if hasattr(self, 'help_overlay') and self.help_overlay:
            self.help_overlay.remove()
            self.help_overlay = None
        if show:
            if message_type == 'help':
                message = ['Left-click and drag for ROI', 'Right-click an image for menu']
                if TKAGG:
                    message += ['Right-click intensity slider to adjust range']
                message += ['Press [esc] to remove ROI/menu/help', 'Close image to quit']
                message = multiline(*message)

            elif message_type == 'difference':
                message = ['Select another image of the same shape and axes permutation', 'or press [esc] to exit']
                message = multiline(*message)

            if message is None:
                return

            self.help_overlay = self.fig.text(0.25, 0.5, message,
                                              ha='left', va='center', fontsize=12,
                                              bbox=dict(facecolor='white', alpha=0.9))

        self.fig.canvas.draw_idle()

    def _load_file(self, file_path, image_index):
        # Do the actual file load, with different behavior for npy, npz, and h5/hdf5.
        ext = os.path.splitext(file_path)[-1].lower()
        data_dict = ()

        if ext == ".npy":
            new_array = np.load(file_path)

        elif ext == ".npz":
            # Check the available arrays and have the user choose.
            array_dict = np.load(file_path)
            array_names = array_dict.files
            shapes = [array_dict[name].shape for name in array_names]
            choice = _choose_array_name(array_names, shapes, file_path)
            if choice is None:
                return
            new_array = array_dict[choice]

        elif ext in {'.h5', '.hdf5'}:
            # Check the available arrays and have the user choose.
            with h5py.File(file_path, "r") as f:
                array_names = [key for key in f.keys()]
                shapes = [f[name].shape for name in array_names]
                choice = _choose_array_name(array_names, shapes, file_path)
                if choice is None:
                    return
                new_array = f[choice][()]
                data_dict = {name: f[choice].attrs[name] for name in f[choice].attrs.keys()}

        if new_array.ndim == 2:
            new_array = new_array[..., np.newaxis]
            self.original_data[image_index] = new_array
        elif new_array.ndim == 3:
            self.original_data[image_index] = new_array
        elif new_array.ndim == 4:
            num_volumes_to_load = min(new_array.shape[-1], self.n_volumes)
            self._show_message(True, message='Loading first {} volumes in 4D array. Press Esc to exit'.format(
                num_volumes_to_load))
            for j in range(num_volumes_to_load):
                self.original_data[j] = new_array[..., j]
                self.data[j] = np.transpose(self.original_data[j], self.axes_perms[j])
            image_index = min(image_index, num_volumes_to_load - 1)
            new_array = new_array[..., image_index]
        else:
            raise ValueError("Loaded array must be 2D, 3D, or 4D")

        self.data_dicts[image_index] = data_dict
        self.axes_perms[image_index] = self._get_perm_from_slice_ind(self.axes_perms[image_index][-1])
        transposed = np.transpose(new_array, self.axes_perms[image_index])
        self.data[image_index] = transposed
        self.cur_slices[image_index] = transposed.shape[2] // 2
        self._draw_images()
        self._on_restore(image_index)

    def _make_option(self, label, position, y_offset, callback):
        # Build a matplotlib menu one option at a time.  This is to allow for cross platform display.
        bounds = self.fig.bbox.bounds
        x_frac = (position.x - bounds[0]) / (bounds[2] - bounds[0])
        y_frac = (position.y + y_offset - bounds[1]) / (bounds[3] - bounds[1])
        txt = self.fig.text(
            x_frac, y_frac,
            label,
            ha='left', va='bottom', fontsize=10, color='white',
            bbox=dict(facecolor='black', edgecolor='white')
        )
        self._menu_texts.append(txt)
        txt._viewer_callback = callback

    def _render_menu(self, event, image_index):
        # Get the menu options, set them up, and display.
        options = self._get_context_menu_options(image_index)

        y_offset = 0
        y_skip = Y_SKIP
        for option in options:
            self._make_option(option[0], event, y_offset, option[1])
            y_offset -= y_skip

        self.fig.canvas.draw_idle()

    def _remove_menu(self):
        if self._menu_texts:
            for txt in self._menu_texts:
                txt.remove()
            self._menu_texts.clear()
            self.fig.canvas.draw_idle()

    def show(self):
        """Display the viewer window and block execution until the window is closed."""
        fignum = self.fig.number
        plt.show()
        # Open and close the figure to make sure it closes properly.
        plt.figure(fignum)
        plt.close(fignum)


def _choose_array_name(array_names, shapes, file_path):
    # Display a dialog for the user to choose an array from a list - for use with _load_file
    if len(array_names) > 1:
        choices = array_names + ['Cancel']
        msg = ['Arrays in {}'.format(os.path.basename(file_path))]
        for name, shape in zip(array_names, shapes):
            msg += ['{}: shape={}'.format(name, shape)]
        choice = easygui.buttonbox(msg=multiline(*msg), title='Choose the array to display', choices=choices)
        if choice == 'Cancel':
            choice = None
    else:
        choice = array_names[0]
    return choice


def slice_viewer(*datasets, data_dicts=None, title='', vmin=None, vmax=None, slice_label=None,
                 slice_axis=None, cmap='gray', show_instructions=True):
    """
    Launch an interactive viewer for inspecting one or more 2D or 3D image arrays.

    This function provides a graphical interface for exploring one or more 3D volumes or 2D slices.
    Features include synchronized slice navigation, ROI statistics, axis transposition, file loading,
    dynamic intensity range adjustment, and interactive GUI tools for zooming and panning.

    Each image can have an associated data dict, typically obtained from :meth:`TomographyModel.recon`, which
    can be viewed as a text file within the viewer.

    Designed primarily for inspecting CT or other volumetric reconstructions in research workflows.

    Args:
        *datasets (ndarray or None): One or more 2D or 3D NumPy arrays to display.
            - 2D arrays are automatically promoted to 3D via a singleton axis.
            - `None` values are replaced with placeholder zero arrays.

        data_dicts (None or dict or list of None or dicts, optional): Dictionary of string entries to associated with the data (e.g., from :meth:`TomographyModel.get_recon_dict`)
        title (str, optional): Window title. Defaults to an empty string.
        vmin (float, optional): Minimum intensity value for display. Defaults to the global minimum across all datasets.
        vmax (float, optional): Maximum intensity value for display. Defaults to the global maximum across all datasets.
        slice_label (str or list of str, optional): Label(s) for the current slice. Defaults to "Slice".
        slice_axis (int or list of int, optional): Axis along which to slice (0, 1, or 2). Defaults to the last axis (2).
        cmap (str, optional): Colormap to use. Defaults to "gray".
        show_instructions (bool, optional): Whether to display usage instructions in the figure. Defaults to True.

    Notes:
        - This function blocks execution until the viewer window is closed.
        - Right-click an image to access a context menu with options such as axis transposition and file loading.
        - Right-click the intensity slider (if using TkAgg backend) to manually set display range bounds.
        - Press 'h' to show help overlay. Press 'Esc' to clear overlays or reset ROI selections.
    """
    viewer = SliceViewer(*datasets, data_dicts=data_dicts, title=title, vmin=vmin, vmax=vmax,
                         slice_label=slice_label,  slice_axis=slice_axis, cmap=cmap,
                         show_instructions=show_instructions)
