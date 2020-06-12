#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""
import argparse
import warnings
from logging import debug

import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib.gridspec import GridSpec
import numpy as np
import numpy.ma as ma
import colorspacious
from colorspacious import cspace_convert

try:
    from PyQt5 import QtWidgets
except ImportError:
    from PyQt4 import QtGui as QtWidgets


def clamp_rgb(rgb):
    return np.clip(rgb, 0, 1)


def add_alpha(im):
    im = np.asanyarray(im)
    if im.shape[-1] != 3:
        raise ValueError(
            "Input image appears to have %d "
            "color channels, "
            "should have exactly 3" % im.shape[-1]
        )
    new_shape = im.shape[:-1] + (4,)
    new_im = ma.zeros(new_shape, dtype=im.dtype)
    new_im[..., :3] = im
    new_im[..., 3] = 1
    new_im[ma.any(ma.getmaskarray(im), axis=-1), :] = 0
    return new_im


def mask_image(im, limit=None):
    c = ~np.all(np.isfinite(im), axis=-1)
    im = im.copy()
    im[c, :] = 0
    c |= np.any(im < 0, axis=-1)
    if limit is not None:
        c |= np.any(im > limit, axis=-1)
    mim = ma.array(im)
    mim[c, :] = ma.masked
    return mim


class ColorPlot(object):
    def __init__(self, pixel_size=600):
        self.pixel_size = pixel_size

    def set_color(self, color):
        pass


def limit(a):
    if a == "C":
        return 115
    elif a == "Q":
        return 128
    elif a == "H":
        return 400
    elif a == "h":
        return 360
    else:
        return 100


class ColorPlotCircular(ColorPlot):
    def __init__(self, pixel_size=600, fixed="J", radial="C", angular="h"):
        super().__init__(pixel_size=pixel_size)
        self.fixed = fixed
        self.radial = radial
        self.angular = angular
        self.color = None
        self.space = fixed + radial + angular
        self.fixed_value = 0
        self.radial_value = 0
        self.angular_value = 0
        self.angular_max = limit(angular)
        self.radial_max = limit(radial)
        #        if angular == 'h':
        #            self.angular_max = 360
        #            self.radial_max = 100
        #        elif angular == 'H':
        #            self.angular_max = 400
        #            self.radial_max = 115
        self.srgba = None
        self.axes = None
        self.dot = None
        self.image = None

    def set_color(self, color):
        if color == self.color:
            return
        else:
            self.color = color
            # And do all the rest of the update too

        debug("Axes %s changing color to %s" % (self.space, color))
        self.fixed_value = getattr(color, self.fixed)
        self.radial_value = getattr(color, self.radial)
        self.angular_value = getattr(color, self.angular)
        debug("Fixed value is now %s=%s", (self.fixed, self.fixed_value))

        v = np.zeros((self.pixel_size, self.pixel_size + 1, 3))
        v[:, :, 0] = self.fixed_value
        v[:, :, 1] = np.linspace(-self.radial_max, self.radial_max, v.shape[0])[:, None]
        v[:, :, 2] = np.linspace(-self.radial_max, self.radial_max, v.shape[1])[None, :]
        angs = np.arctan2(v[:, :, 2], v[:, :, 1]) * self.angular_max / (2 * np.pi)
        angs %= self.angular_max
        rads = np.hypot(v[:, :, 2], v[:, :, 1])
        v[:, :, 1] = rads
        v[:, :, 2] = angs
        self.srgba = add_alpha(
            mask_image(cspace_convert(v, self.space, "sRGB1"), limit=1)
        )
        self.color_pos = (
            self.radial_value
            * np.cos(2 * np.pi * self.angular_value / self.angular_max),
            self.radial_value
            * np.sin(2 * np.pi * self.angular_value / self.angular_max),
        )
        if self.dot is not None:
            debug("new color position %s" % (self.color_pos,))
            self.dot.set_xdata(self.color_pos[0])
            self.dot.set_ydata(self.color_pos[1])
        if self.image is not None:
            self.image.set_data(self.srgba.transpose((1, 0, 2)))

    def construct_plot(self, axes):
        if self.axes is not None:
            warnings.warn("Overwriting existing axes")
        if self.srgba is None:
            raise ValueError("Initial color not yet available")
        self.axes = axes
        self.image = axes.imshow(
            self.srgba.transpose((1, 0, 2)),
            extent=(
                -self.radial_max,
                self.radial_max,
                -self.radial_max,
                self.radial_max,
            ),
            origin="lower",
        )
        (self.dot,) = axes.plot(
            self.color_pos[0], self.color_pos[1], "+", color="white"
        )
        axes.tick_params(
            axis="x", which="both", bottom=False, top=False, labelbottom=False
        )
        axes.tick_params(
            axis="y", which="both", left=False, right=False, labelleft=False
        )
        axes.set_xlabel("angle: %s" % self.angular)
        axes.set_ylabel("radius: %s" % self.radial)
        axes.set_aspect("equal")

    def click_color(self, xy):
        debug("converting color at %s" % (xy,))
        x, y = xy
        ang = np.arctan2(y, x) * self.angular_max / (2 * np.pi)
        ang %= self.angular_max
        rad = np.hypot(y, x)

        c = self.fixed_value, rad, ang
        debug("decodes to %s in space %s" % (c, self.space))
        r = cspace_convert(c, self.space, "CIECAM02")
        debug(r)
        return r


class ColorPlotRectangular(ColorPlot):
    def __init__(self, pixel_size=600, fixed="h", x="C", y="J", double=False):
        super().__init__(pixel_size=pixel_size)
        self.fixed = fixed
        self.x = x
        self.y = y
        self.color = None
        self.space = fixed + x + y
        self.fixed_value = 0
        self.x_value = 0
        self.y_value = 0
        self.x_max = limit(x)
        self.y_max = limit(y)
        self.srgba = None
        self.axes = None
        self.dot = None
        self.image = None
        self.double = double

    def set_color(self, color):
        if color == self.color:
            return
        else:
            self.color = color
            # And do all the rest of the update too

        debug("Axes %s changing color to %s" % (self.space, color))
        self.fixed_value = getattr(color, self.fixed)
        self.x_value = getattr(color, self.x)
        self.y_value = getattr(color, self.y)

        v = np.zeros((self.pixel_size, self.pixel_size + 1, 3))
        if self.double:
            h = v.shape[0] // 2
            v[:h, :, 0] = (self.fixed_value + limit(self.fixed) / 2) % limit(self.fixed)
            v[h:, :, 0] = self.fixed_value
            v[:h, :, 0] = (self.fixed_value + limit(self.fixed) / 2) % limit(self.fixed)
            v[:, :, 1] = np.abs(np.linspace(-self.x_max, self.x_max, v.shape[0]))[
                :, None
            ]
            v[:, :, 2] = np.linspace(0, self.y_max, v.shape[1])[None, :]
        else:
            v[:, :, 0] = self.fixed_value
            v[:, :, 1] = np.linspace(0, self.x_max, v.shape[0])[:, None]
            v[:, :, 2] = np.linspace(0, self.y_max, v.shape[1])[None, :]
        self.srgba = add_alpha(
            mask_image(cspace_convert(v, self.space, "sRGB1"), limit=1)
        )
        self.color_pos = self.x_value, self.y_value
        if self.dot is not None:
            debug("new color position %s" % (self.color_pos,))
            self.dot.set_xdata(self.color_pos[0])
            self.dot.set_ydata(self.color_pos[1])
        if self.image is not None:
            self.image.set_data(self.srgba.transpose((1, 0, 2)))

    def construct_plot(self, axes):
        if self.axes is not None:
            warnings.warn("Overwriting existing axes")
        if self.srgba is None:
            raise ValueError("Initial color not yet available")
        self.axes = axes
        if self.double:
            self.image = axes.imshow(
                self.srgba.transpose((1, 0, 2)),
                extent=(-self.x_max, self.x_max, 0, self.y_max),
                origin="lower",
            )
        else:
            self.image = axes.imshow(
                self.srgba.transpose((1, 0, 2)),
                extent=(0, self.x_max, 0, self.y_max),
                origin="lower",
            )
        (self.dot,) = axes.plot(
            self.color_pos[0], self.color_pos[1], "+", color="white"
        )
        axes.tick_params(
            axis="x", which="both", bottom=False, top=False, labelbottom=False
        )
        axes.tick_params(
            axis="y", which="both", left=False, right=False, labelleft=False
        )
        axes.set_xlabel(self.x)
        axes.set_ylabel(self.y)
        axes.set_aspect("auto")

    def click_color(self, xy):
        debug("converting color at %s" % (xy,))
        x, y = xy
        f = self.fixed_value
        if x < 0:
            f = (f + limit(self.fixed) / 2) % limit(self.fixed)
            x = -x
        c = f, x, y
        debug("decodes to %s in space %s" % (c, self.space))
        r = cspace_convert(c, self.space, "CIECAM02")
        debug(r)
        return r


class ColorTool(object):
    def __init__(self, color="red"):
        self.color = None
        self.intensity = "J"
        self.colorfulness = "C"
        self.hue = "H"
        self.space = self.intensity + self.colorfulness + self.hue
        self.plots = [
            [
                ColorPlotCircular(
                    fixed=self.intensity, radial=self.colorfulness, angular=self.hue
                ),
                ColorPlotCircular(
                    fixed=self.colorfulness, radial=self.intensity, angular=self.hue
                ),
            ],
            [
                ColorPlotRectangular(
                    fixed=self.hue, x=self.colorfulness, y=self.intensity, double=True
                ),
                ColorStripPlot(
                    fixed=self.hue,
                    fixed_2=self.colorfulness,
                    y=self.intensity,
                    double=True,
                ),
                ColorStripPlot(
                    fixed=self.hue,
                    fixed_2=self.intensity,
                    y=self.colorfulness,
                    double=True,
                ),
                ColorStripPlot(
                    fixed=self.hue, fixed_2=self.intensity, y="s", double=True
                ),
                ColorStripPlot(
                    fixed="h", fixed_2=self.colorfulness, y=self.intensity, double=True
                ),
                ColorStripPlot(
                    fixed="h", fixed_2=self.intensity, y=self.colorfulness, double=True
                ),
                ColorStripPlot(fixed="h", fixed_2=self.intensity, y="s", double=True),
            ],
        ]
        self.figure = None
        self.swatch_figure = plt.figure()
        self.set_color(color)
        self.figure = plt.figure()
        gs = GridSpec(3, 12, figure=self.figure)
        self.axes = [
            [self.figure.add_subplot(gs[0, :3]), self.figure.add_subplot(gs[0, 3:6])],
            [
                self.figure.add_subplot(gs[1:, :6]),
                self.figure.add_subplot(gs[:, 6]),
                self.figure.add_subplot(gs[:, 7]),
                self.figure.add_subplot(gs[:, 8]),
                self.figure.add_subplot(gs[:, 9]),
                self.figure.add_subplot(gs[:, 10]),
                self.figure.add_subplot(gs[:, 11]),
            ],
        ]
        self.figure.patch.set_facecolor("darkgray")
        for ar, pr in zip(self.axes, self.plots):
            for a, p in zip(ar, pr):
                a.patch.set_facecolor("gray")
                p.construct_plot(a)
        self.figure.canvas.mpl_connect("button_press_event", self.onclick)
        self.figure.suptitle(self.space)
        self.figure.canvas.set_window_title("CIECAM02 color tool")
        try:
            win = self.swatch_figure.canvas.manager.window
        except AttributeError:
            win = self.swatch_figure.canvas.window()
        toolbar = win.findChild(QtWidgets.QToolBar)
        toolbar.setVisible(False)
        win.statusBar().setVisible(False)

    def set_color(self, color):
        if isinstance(color, str):
            color = matplotlib.colors.to_rgb(color)
        if len(color) in (3, 4):
            if max(*color) > 1:
                raise ValueError("Colors should be in RGB [0,1]: %s" % color)
            debug("RGB color %s" % (color,))
            color = cspace_convert(color, "sRGB1", "CIECAM02")
            debug("Converted to %s" % (color,))
            debug(
                "And back via JCh to %s"
                % (cspace_convert((color.J, color.C, color.h), "JCh", "sRGB1"),)
            )
            debug(
                "And back via JCH to %s"
                % (cspace_convert((color.J, color.C, color.H), "JCH", "sRGB1"),)
            )
        if not isinstance(color, colorspacious.JChQMsH):
            raise ValueError("Don't know how to interpret color %s" % color)
        print(cspace_convert(color, "CIECAM02", "sRGB1"))
        for pr in self.plots:
            for p in pr:
                p.set_color(color)
        if self.figure is not None:
            self.figure.canvas.draw()
            self.figure.canvas.flush_events()
        if self.swatch_figure is not None:
            rgb = clamp_rgb(cspace_convert((color.J, color.C, color.H), "JCH", "sRGB1"))
            debug(repr(rgb))
            self.swatch_figure.set_facecolor(rgb)
            self.swatch_figure.canvas.set_window_title(
                "R:%.2f G:%.2f B:%.2f" % (rgb[0], rgb[1], rgb[2])
            )
            self.swatch_figure.canvas.draw()
            self.swatch_figure.canvas.flush_events()

    def onclick(self, event):
        debug("Click: %s" % event)
        c = None
        for pr in self.plots:
            for p in pr:
                if event.inaxes == p.axes:
                    c = p.click_color((event.xdata, event.ydata))
        if c is not None:
            debug("Changing color to %s" % (c,))
            self.set_color(c)


class ColorStripPlot(ColorPlot):
    def __init__(self, pixel_size=600, fixed="h", fixed_2="C", y="J", double=False):
        super().__init__(pixel_size=pixel_size)
        self.fixed = fixed
        self.fixed_2 = fixed_2
        self.y = y
        self.color = None
        self.space = fixed + fixed_2 + y
        self.fixed_value = 0
        self.fixed_2_value = 0
        self.y_value = 0
        self.y_max = limit(y)
        self.srgba = None
        self.axes = None
        self.line = None
        self.image = None
        self.double = double

    def set_color(self, color):
        if color == self.color:
            return
        else:
            self.color = color
            # And do all the rest of the update too

        debug("Axes %s changing color to %s" % (self.space, color))
        self.fixed_value = getattr(color, self.fixed)
        self.fixed_2_value = getattr(color, self.fixed_2)
        self.y_value = getattr(color, self.y)

        v = np.zeros((2, self.pixel_size, 3))
        if self.double:
            v[1, :, 0] = self.fixed_value
            v[0, :, 0] = (self.fixed_value + limit(self.fixed) / 2) % limit(self.fixed)
            v[:, :, 1] = self.fixed_2_value
            v[:, :, 2] = np.linspace(0, self.y_max, v.shape[1])[None, :]
        else:
            v[:, :, 0] = self.fixed_value
            v[:, :, 1] = self.fixed_2_value
            v[:, :, 2] = np.linspace(0, self.y_max, v.shape[1])[None, :]
        self.srgba = add_alpha(
            mask_image(cspace_convert(v, self.space, "sRGB1"), limit=1)
        )
        self.color_pos = self.y_value
        if self.line is not None:
            debug("new color position %s" % (self.color_pos,))
            self.line.set_ydata(self.color_pos)
        if self.image is not None:
            self.image.set_data(self.srgba.transpose((1, 0, 2)))

    def construct_plot(self, axes):
        if self.axes is not None:
            warnings.warn("Overwriting existing axes")
        if self.srgba is None:
            raise ValueError("Initial color not yet available")
        self.axes = axes
        self.image = axes.imshow(
            self.srgba.transpose((1, 0, 2)),
            extent=(0, 1, 0, self.y_max),
            origin="lower",
        )
        self.line = axes.axhline(self.color_pos, color="white")
        axes.tick_params(
            axis="x", which="both", bottom=False, top=False, labelbottom=False
        )
        axes.tick_params(
            axis="y", which="both", left=False, right=False, labelleft=False
        )
        axes.set_ylabel(self.y)
        axes.set_title("%s, %s" % (self.fixed, self.fixed_2))
        axes.set_aspect("auto")

    def click_color(self, xy):
        debug("converting color at %s" % (xy,))
        x, y = xy
        if self.double and x < 0.5:
            c = (
                (self.fixed_value + limit(self.fixed) / 2) % limit(self.fixed),
                self.fixed_2_value,
                y,
            )
        else:
            c = self.fixed_value, self.fixed_2_value, y
        debug("decodes to %s in space %s" % (c, self.space))
        r = cspace_convert(c, self.space, "CIECAM02")
        debug(r)
        return r


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Display a color adjustment tool.")
    parser.add_argument("--color", default="green", help="Initial color")
    args = parser.parse_args()

    C = ColorTool(args.color)

    plt.show()
