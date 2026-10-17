#!/usr/bin/env python
"""Download the PointMaze diffusion-policy checkpoints from Hugging Face into ``checkpoints/`` (git-ignored).

    python scripts/download_checkpoints.py                      # epoch 1900 only (golden tests, run.py default)
    python scripts/download_checkpoints.py --all                # all five Fig. 4 checkpoints (~2.6 GB)
    python scripts/download_checkpoints.py --all --local-dir <dir>
                                                                # symlink <dir>/<name> (an existing copy) instead
    python scripts/download_checkpoints.py --dataset            # only the DP training dataset (303 MB), see below

The files are the ``pointmaze/`` folder of https://huggingface.co/RoboWonsuhk/JM2D. The five checkpoints are the
"5 random seeds" of Fig. 4: epochs 1250/1500/1750/2000 of one robomimic training run
(``configs/train_dp_maze2d_large.json``) and the epoch-1900 checkpoint of an earlier training (byte-identical to the
public RAIL "maze-large" model). Every file is checked against the md5 below.

``--dataset`` downloads the robomimic training dataset ``pointmaze/dataset/maze2d_large_dense_v1.hdf5`` (D4RL
``maze2d-large-dense-v1`` converted with ``robomimic/scripts/conversion/convert_d4rl.py``; D4RL data: CC BY 4.0) to
``datasets/d4rl/converted/`` (git-ignored), the path of ``configs/train_dp_maze2d_large.json``. Checkpoints are then
only downloaded if named or ``--all`` is given.
"""
import argparse
import hashlib
import os
import shutil
import sys
import tempfile

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CKPT_DIR = os.path.join(REPO_ROOT, "checkpoints")
DATASET_DIR = os.path.join(REPO_ROOT, "datasets", "d4rl", "converted")  # see configs/train_dp_maze2d_large.json
REPO_ID = "RoboWonsuhk/JM2D"
REPO_FOLDER = "pointmaze"

# name -> md5, size (bytes); stored in the Hugging Face repo as pointmaze/<name>
CHECKPOINTS = {
    "maze2d_large_ep1900.pth": dict(md5="a756af86fe2070c2e77358ca380ccc44", size=523634990),
    "maze2d_large_ep1250.pth": dict(md5="98614093985811a0dbcbb4a66b3ccfc4", size=523635054),
    "maze2d_large_ep1500.pth": dict(md5="000b0f287307bd98289cc9b28479566c", size=523635054),
    "maze2d_large_ep1750.pth": dict(md5="a7fd3fe32e1b35723dde1c566c61f1b9", size=523635054),
    "maze2d_large_ep2000.pth": dict(md5="7f76546cb300ec7577c475d1f3b40f97", size=523635054),
}
DEFAULT = ["maze2d_large_ep1900.pth"]
# stored in the Hugging Face repo as pointmaze/dataset/<name>
DATASET = ("maze2d_large_dense_v1.hdf5", dict(md5="aa4be41d7444c0bdc8fcf48707e11c84", size=302731504))


def md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def hf_download(repo_id, revision, filename, dst):
    """Download ``filename`` of the model repo to ``dst``. Returns False if this file cannot be downloaded; exits on
    errors that concern the whole repo (not accessible, unknown revision, offline)."""
    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import (EntryNotFoundError, HfHubHTTPError, LocalEntryNotFoundError,
                                           RepositoryNotFoundError, RevisionNotFoundError)
    except ImportError:
        sys.exit("huggingface_hub is not installed: pip install huggingface_hub")
    tmp = tempfile.mkdtemp(dir=os.path.dirname(dst), prefix=".hf_download_")
    try:
        path = hf_hub_download(repo_id, filename, revision=revision, local_dir=tmp)  # uses the HF token if logged in
        os.replace(path, dst)
        return True
    except RepositoryNotFoundError:
        sys.exit(f"Cannot access the Hugging Face repo {repo_id}: it does not exist, or it is private and you are not "
                 f"logged in. If you have access, log in with `hf auth login` (older huggingface_hub: "
                 f"`huggingface-cli login`) or set HF_TOKEN, then retry.")
    except RevisionNotFoundError:
        sys.exit(f"Revision {revision!r} not found in the Hugging Face repo {repo_id}.")
    except LocalEntryNotFoundError:
        sys.exit("Cannot reach huggingface.co (offline?); check the network connection and retry.")
    except EntryNotFoundError:
        print(f"{filename} not found in the Hugging Face repo {repo_id} (revision {revision})", file=sys.stderr)
    except HfHubHTTPError as e:
        print(f"{filename}: download failed: {e}", file=sys.stderr)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return False


def fetch(name, spec, local_dir=None, verify=True, repo_id=REPO_ID, revision="main", dst_dir=CKPT_DIR,
          repo_folder=REPO_FOLDER):
    dst = os.path.join(dst_dir, name)
    if os.path.exists(dst) and (not verify or md5(dst) == spec["md5"]):
        print(f"{name}: present")
        return True
    if os.path.lexists(dst):
        os.remove(dst)
    if local_dir is not None:
        src = os.path.join(local_dir, name)
        if not os.path.exists(src):
            print(f"{name}: {src} not found", file=sys.stderr)
            return False
        os.symlink(os.path.abspath(src), dst)
        print(f"{name}: linked -> {src}")
    else:
        print(f"{name}: downloading {repo_id}/{repo_folder}/{name} ({spec['size'] / 1e6:.0f} MB)", flush=True)
        if not hf_download(repo_id, revision, f"{repo_folder}/{name}", dst):
            return False
    if verify:
        got = md5(dst)
        if got != spec["md5"]:
            print(f"{name}: md5 mismatch ({got} != {spec['md5']})", file=sys.stderr)
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="*", help=f"checkpoint names (default {DEFAULT}); one of {list(CHECKPOINTS)}")
    parser.add_argument("--all", action="store_true", help="all five Fig. 4 checkpoints")
    parser.add_argument("--dataset", action="store_true",
                        help=f"download the training dataset {DATASET[0]} to datasets/d4rl/converted/ "
                             f"(no checkpoint unless named or --all)")
    parser.add_argument("--local-dir", default=None,
                        help="symlink <dir>/<name> from a directory that already holds the files instead of downloading")
    parser.add_argument("--no-verify", action="store_true", help="skip the md5 check")
    parser.add_argument("--repo-id", default=REPO_ID, help=f"Hugging Face model repo (default {REPO_ID})")
    parser.add_argument("--revision", default="main", help="branch, tag or commit of the repo (default main)")
    args = parser.parse_args()

    names = list(CHECKPOINTS) if args.all else (args.names or ([] if args.dataset else DEFAULT))
    unknown = [n for n in names if n not in CHECKPOINTS]
    if unknown:
        parser.error(f"unknown checkpoint(s) {unknown}")
    ok = []
    if args.dataset:
        os.makedirs(DATASET_DIR, exist_ok=True)
        ok.append(fetch(*DATASET, args.local_dir, verify=not args.no_verify, repo_id=args.repo_id,
                        revision=args.revision, dst_dir=DATASET_DIR, repo_folder=f"{REPO_FOLDER}/dataset"))
    if names:
        os.makedirs(CKPT_DIR, exist_ok=True)
    ok += [fetch(n, CHECKPOINTS[n], args.local_dir, verify=not args.no_verify, repo_id=args.repo_id,
                 revision=args.revision) for n in names]
    if not all(ok):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
