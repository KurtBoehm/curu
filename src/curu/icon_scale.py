# This file is part of https://github.com/KurtBoehm/curu.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from argparse import ArgumentParser
from pathlib import Path

from pyvips import Image


def run():
    parser = ArgumentParser()
    parser.add_argument("src", type=Path)
    parser.add_argument("prefix", type=Path)
    args = parser.parse_args()
    src: Path = args.src
    prefix: Path = args.prefix

    src_img = Image.new_from_file(src)
    w = src_img.width
    for pix in [512, 384, 256, 128, 96, 64, 48]:
        dst = src_img.resize(pix / w)
        dst.write_to_file(prefix.parent / f"{prefix.name}-{pix}.png")
