import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import gridspec
from matplotlib.widgets import RangeSlider, Slider, RadioButtons, CheckButtons

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
        self.slice_axes = []
        self.labels = []
        self.cur_slices = []
        self.axes = []
        self.images = []
        self.circles = []
        self.text_boxes = []
        self.axis_buttons = []

        self._normalize_inputs()
        self._prepare_data()
        self._build_figure()

    def _normalize_inputs(self):
        if isinstance(self.slice_axis, int) or self.slice_axis is None:
            self.slice_axes = [2 if self.slice_axis is None else self.slice_axis] * self.n_volumes
        else:
            self.slice_axes = list(self.slice_axis)

        if isinstance(self.slice_label, str) or self.slice_label is None:
            self.labels = [f"Slice {i+1}" if self.slice_label is None else self.slice_label
                           for i in range(self.n_volumes)]
        else:
            self.labels = list(self.slice_label)

    def _prepare_data(self):
        self.original_data = list(self.datasets)
        for i, data in enumerate(self.datasets):
            if data.ndim == 2:
                data = data[..., np.newaxis]
            elif data.ndim != 3:
                raise ValueError("Each input data must be a 2D or 3D array")
            data = np.moveaxis(data, self.slice_axes[i], 2)
            self.data.append(data)

        self.cur_slices = [d.shape[2] // 2 for d in self.data]
        all_data = np.concatenate([d.ravel() for d in self.data])
        if self.vmin is None:
            self.vmin = np.min(all_data)
        if self.vmax is None:
            self.vmax = np.max(all_data)
        if self.vmin == self.vmax:
            eps = 1e-6
            scale = np.clip(eps * np.abs(self.vmax), a_min=eps, a_max=None)
            self.vmin -= scale
            self.vmax += scale

    def _build_figure(self):
        figwidth = 6 * self.n_volumes
        self.fig = plt.figure(figsize=(figwidth, 8))
        self.fig.suptitle(self.title)
        self.gs = gridspec.GridSpec(nrows=5, ncols=self.n_volumes, height_ratios=[10, 1, 1, 1, 1])

        self._draw_images()
        self._add_slice_slider()
        self._add_intensity_slider()
        self._add_axis_controls()
        self._connect_events()

        self.fig.text(0.01, 0.85, 'Close plot\nto continue', rotation='vertical')

    def _draw_images(self):
        self.axes = []
        self.images = []
        self.circles = [None] * self.n_volumes
        self.text_boxes = [None] * self.n_volumes
        for i, d in enumerate(self.data):
            ax = self.fig.add_subplot(self.gs[0, i])
            img = ax.imshow(d[:, :, self.cur_slices[i]], cmap=self.cmap,
                            vmin=self.vmin, vmax=self.vmax)
            ax.set_title(f"{self.labels[i]} {self.cur_slices[i]}\nShape: {self.datasets[i].shape}")
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            self.fig.colorbar(img, cax=cax, orientation='vertical')
            self.axes.append(ax)
            self.images.append(img)

    def _add_slice_slider(self):
        ax = self.fig.add_subplot(self.gs[1, :])
        self.slice_slider = Slider(ax, label="Slice", valmin=0,
                                   valmax=max(d.shape[2] for d in self.data) - 1,
                                   valinit=self.cur_slices[0], valfmt='%0.0f')
        self.slice_slider.on_changed(self._update_slice)

    def _add_intensity_slider(self):
        ax = self.fig.add_subplot(self.gs[2, :])
        log_range = np.log10(self.vmax - self.vmin)
        digits = max(-int(np.round(log_range)) + 2, 0)
        valfmt = '%0.' + str(digits) + 'f'
        self.intensity_slider = RangeSlider(ax=ax, label="Intensity range",
                                            valmin=self.vmin, valmax=self.vmax,
                                            valinit=(self.vmin, self.vmax), valfmt=valfmt)
        self.intensity_slider.on_changed(self._update_intensity)

    def _add_axis_controls(self):
        self.axis_buttons = []
        all_same = all(ax == self.slice_axes[0] for ax in self.slice_axes)

        if self.n_volumes == 1:
            self._create_axis_buttons(False)
        else:
            self.cb_ax = self.fig.add_subplot(self.gs[4, 0])
            self.cb = CheckButtons(self.cb_ax, labels=["Decouple slice axes"], actives=[not all_same])
            self.cb.on_clicked(self._toggle_decouple)
            self._create_axis_buttons(not all_same)

    def _create_axis_buttons(self, decoupled):
        for b in self.axis_buttons:
            b.ax.remove()
        self.axis_buttons.clear()

        if not decoupled:
            ax = self.fig.add_subplot(self.gs[3, :])
            btns = RadioButtons(ax, labels=["0", "1", "2"])
            for lbl in btns.labels: lbl.set_fontsize(8)
            btns.set_active(self.slice_axes[0])
            btns.on_clicked(lambda label: [self._update_axis(i, int(label)) for i in range(self.n_volumes)])
            ax.set_title("Slice axis")
            self.axis_buttons.append(btns)
        else:
            for i in range(self.n_volumes):
                ax = self.fig.add_subplot(self.gs[3, i])
                btns = RadioButtons(ax, labels=["0", "1", "2"])
                for lbl in btns.labels: lbl.set_fontsize(8)
                btns.set_active(self.slice_axes[i])
                btns.on_clicked(lambda label, i=i: self._update_axis(i, int(label)))
                self.axis_buttons.append(btns)

    def _toggle_decouple(self, label):
        self._create_axis_buttons(self.cb.get_status()[0])
        self.fig.canvas.draw_idle()

    def _update_axis(self, i, new_axis):
        if new_axis != self.slice_axes[i]:
            orig = self.original_data[i]
            self.slice_axes[i] = new_axis
            new_data = np.moveaxis(orig, new_axis, 2)
            self.data[i] = new_data
            self.cur_slices[i] = new_data.shape[2] // 2
            self.images[i].set_data(new_data[:, :, self.cur_slices[i]])
            self.axes[i].set_title(f"{self.labels[i]} {self.cur_slices[i]}\nShape: {orig.shape}")
            self.fig.canvas.draw_idle()

    def _update_slice(self, val):
        new_slice = int(round(val))
        for i, d in enumerate(self.data):
            frac = new_slice / d.shape[2]
            idx = int(round(frac * d.shape[2]))
            idx = np.clip(idx, 0, d.shape[2] - 1)
            self.cur_slices[i] = idx
            self.images[i].set_data(d[:, :, idx])
            self.axes[i].set_title(f"{self.labels[i]} {idx}\nShape: {self.original_data[i].shape}")
        self._display_mean()
        self.fig.canvas.draw_idle()

    def _update_intensity(self, val):
        for img in self.images:
            img.set_clim(val[0], val[1])
        self.fig.canvas.draw_idle()

    def _get_mask(self, data, x, y, r):
        ny, nx = data.shape[:2]
        xv, yv = np.meshgrid(np.arange(nx), np.arange(ny))
        return (xv - x)**2 + (yv - y)**2 <= r**2

    def _display_mean(self):
        if all(c is None for c in self.circles):
            return
        for i, circle in enumerate(self.circles):
            if circle is None:
                continue
            x, y = circle.center
            r = circle.get_radius()
            mask = self._get_mask(self.data[i][:, :, self.cur_slices[i]], x, y, r)
            values = self.data[i][:, :, self.cur_slices[i]][mask]
            if values.size:
                mean, std = np.mean(values), np.std(values)
                if self.text_boxes[i]:
                    self.text_boxes[i].remove()
                self.text_boxes[i] = self.axes[i].text(0.05, 0.95,
                    f"Mean: {mean:.3g}\nStd Dev: {std:.3g}",
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

    def _connect_events(self):
        self._is_moving = False
        self._move_offset = None
        self.tooltips = [ax.annotate('Click and drag to move\nPress Esc to remove', xy=(0, 0), xytext=(10, 10),
                                     textcoords='offset points', bbox=dict(boxstyle='round', fc='w'),
                                     arrowprops=dict(arrowstyle='->'), visible=False)
                         for ax in self.axes]
        self._drag_start = None
        self._is_drawing = False  # Add a flag to track drawing state
        def on_press(event):
            if event.inaxes not in self.axes: return
            if hasattr(self.fig.canvas, 'toolbar') and getattr(self.fig.canvas.toolbar, 'mode', ''): return
            for i, circle in enumerate(self.circles):
                if circle is None: continue
                x, y = circle.center
                r = circle.get_radius()
                if event.xdata is not None and event.ydata is not None:
                    dx = event.xdata - x
                    dy = event.ydata - y
                    if dx**2 + dy**2 <= r**2:
                        self._is_moving = True
                        self._move_offset = (dx, dy)
                        return

            self._remove_graphics()
            self._is_drawing = True
            self._drag_start = (event.xdata, event.ydata)
            for i, ax in enumerate(self.axes):
                circ = plt.Circle((event.xdata, event.ydata), 0, color='red', lw=2, fill=False, alpha=1.0)
                self.circles[i] = circ
                ax.add_patch(circ)
            self.fig.canvas.draw_idle()

        def on_motion(event):
            if event.inaxes not in self.axes: return
            if hasattr(self.fig.canvas, 'toolbar') and getattr(self.fig.canvas.toolbar, 'mode', ''): return

            for i, circle in enumerate(self.circles):
                if circle is None: continue
                x, y = circle.center
                r = circle.get_radius()
                if event.xdata is not None and event.ydata is not None:
                    dx = event.xdata - x
                    dy = event.ydata - y
                    if dx**2 + dy**2 <= r**2:
                        self.tooltips[i].xy = (event.xdata, event.ydata)
                        self.tooltips[i].set_visible(True)
                    else:
                        self.tooltips[i].set_visible(False)

            if self._is_drawing and self._drag_start is not None:
                x0, y0 = self._drag_start
                r = np.sqrt((event.xdata - x0)**2 + (event.ydata - y0)**2)
                for i, ax in enumerate(self.axes):
                    if self.circles[i]:
                        self.circles[i].remove()
                    circ = plt.Circle((x0, y0), r, color='red', lw=2, fill=False, alpha=1.0)
                    self.circles[i] = circ
                    ax.add_patch(circ)
                self._display_mean()

            elif getattr(self, '_is_moving', False):
                dx, dy = self._move_offset
                new_center = (event.xdata - dx, event.ydata - dy)
                for j in range(self.n_volumes):
                    self.circles[j].center = new_center
                self._display_mean()

            self.fig.canvas.draw_idle()

        def on_release(event):
            if getattr(self, '_is_moving', False):
                self._is_moving = False
                self._move_offset = None
            self._is_drawing = False
            self._drag_start = None
            self._display_mean()

        def on_key(event):
            if event.key == 'escape':
                for i in range(self.n_volumes):
                    if self.text_boxes[i] is not None:
                        self.text_boxes[i].remove()
                self.text_boxes = [None] * self.n_volumes
                self._remove_graphics()
                self.circles = [None] * self.n_volumes
                self._display_mean()
                self.fig.canvas.draw_idle()

        self.fig.canvas.mpl_connect('button_press_event', on_press)
        self.fig.canvas.mpl_connect('motion_notify_event', on_motion)
        self.fig.canvas.mpl_connect('button_release_event', on_release)
        self.fig.canvas.mpl_connect('key_press_event', on_key)

    def show(self):
        plt.tight_layout()
        plt.show()


def slice_viewer(*datasets, **kwargs):
    viewer = SliceViewer(*datasets, **kwargs)
    viewer.show()
