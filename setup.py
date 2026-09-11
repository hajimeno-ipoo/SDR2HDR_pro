"""Build the macOS OpenGL-to-Metal display bridge with the platform SDK."""
import sys
from setuptools import Extension, setup

setup(ext_modules=[Extension(
    'sdr2hdr._compare_gl', ['src/sdr2hdr/compare_gl.m'],
    extra_compile_args=['-Wno-deprecated-declarations'],
    extra_link_args=['-framework', 'Cocoa', '-framework', 'QuartzCore', '-framework', 'OpenGL',
                     '-framework', 'CoreVideo', '-framework', 'Metal'],
)] if sys.platform == 'darwin' else [])
