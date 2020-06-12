# CIECAM02 color tool

This is a toy for exploring the CIECAM02 perceptual color space. It's an early draft and is awkward to use and install. The key dependencies are `colorspacious`  for conversions, and `matplotlib`, `numpy`, and `pyqt5` for  UI and computational needs.

Its primary function is to display slices through the CIECAM02 colour space; if you click on a colour it will update to show slices passing through that colour. The coordinates along the axes are marked; CIECAM02 uses J C and H (or a slightly modified version called h).

You should be able to just run `python color_tool.py`. The other stuff is even more preliminary and/or experimental.
