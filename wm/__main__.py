# -*- coding: utf-8 -*-
"""python -m wm [run|checks|version]"""
import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
