"""Fetch pretrained weights into $PROJECT_DIR/weights (a few hundred MB, safe on the login node).

Compute nodes may not have internet access, so do this once beforehand. `yolo26n.pt` is only
used by Ultralytics' automatic-mixed-precision self-check at the start of training.
"""

import argparse

import common

DEFAULT = [f"yolo26{s}-cls.pt" for s in "nsmlx"] + ["yolo11s-cls.pt", "yolo26n.pt"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", default=DEFAULT)
    a = ap.parse_args()
    common.init_ultralytics()
    from ultralytics.utils.downloads import attempt_download_asset

    for n in a.names:
        target = common.weights_dir() / n
        if target.exists():
            print("have", target)
            continue
        got = attempt_download_asset(str(target))
        print("downloaded", got)


if __name__ == "__main__":
    main()
