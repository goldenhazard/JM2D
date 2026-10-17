"""Download the five Diffuser planner checkpoints (seeds 0-4) used for Table 2 / Table 3 from Hugging Face, or (with
``--data``) the D3IL Avoiding demonstrations.

    python scripts/download_checkpoints.py [--dest checkpoints/avoiding-d3il/H8_K20_Dmodels.GaussianDiffusion]
    python scripts/download_checkpoints.py --data [--data-dest third_party/d3il/environments/dataset/data/avoiding/data]

The checkpoints are the ``d3il/avoiding-d3il/H8_K20_Dmodels.GaussianDiffusion/`` folder of
https://huggingface.co/RoboWonsuhk/JM2D, downloaded one file at a time (files already present are kept).
Each seed directory holds ``state_best.pt`` (~32 MB; keys step, model, ema) and the config pickles
``{dataset,model,diffusion,trainer}_config.pkl`` + ``losses.pkl`` (they reference ``diffuser.*`` module paths, which
this repository keeps). The md5 of every ``state_best.pt`` is checked against ``CHECKPOINTS`` in this script.
Alternatively pass any directory with the same layout to ``scripts/run.py --ckpt-dir``.

``--data`` downloads the 96 ``env_*.pkl`` files of ``d3il/data/avoiding/`` (2.6 MB; an unmodified copy of the D3IL
Avoiding demonstrations, ALRhub/d3il, MIT License) into the folder the dataset loader reads (the patched D3IL clone
must exist, README Installation). Every file is checked against ``DATA_MD5SUMS`` (= ``d3il/data/avoiding/MD5SUMS``).
The folder must hold nothing else: the loader reads every file in it.
"""
import argparse
import hashlib
import os
import shutil
import sys
import tempfile

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_DEST = os.path.join(REPO, "checkpoints", "avoiding-d3il", "H8_K20_Dmodels.GaussianDiffusion")
REPO_ID = "RoboWonsuhk/JM2D"
REPO_FOLDER = "d3il/avoiding-d3il/H8_K20_Dmodels.GaussianDiffusion"  # <seed>/<file> below it
D3IL_DIR = os.path.join(REPO, "third_party", "d3il")
DATA_DEST = os.path.join(D3IL_DIR, "environments", "dataset", "data", "avoiding", "data")
DATA_FOLDER = "d3il/data/avoiding"  # env_*.pkl below it

FILES = ["state_best.pt", "dataset_config.pkl", "model_config.pkl", "diffusion_config.pkl", "trainer_config.pkl",
         "losses.pkl"]
CHECKPOINTS = {  # seed -> md5(state_best.pt)
    "0": "8446a72feebe0e9b1839bc5cb48f448e",
    "1": "c09dd9a85fcf435201d8e61835d053f0",
    "2": "4c55388894fdf2ebeb22206678125ac3",
    "3": "7d219c1b2df29557297bde3ab561b784",
    "4": "7238d20b674872da4a8d88e8469a5a3b",
}


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(dest):
    ok = True
    for seed, digest in CHECKPOINTS.items():
        for name in FILES:
            path = os.path.join(dest, seed, name)
            if not os.path.exists(path):
                print(f"missing {path}")
                ok = False
        path = os.path.join(dest, seed, "state_best.pt")
        if os.path.exists(path) and md5(path) != digest:
            print(f"md5 mismatch {path}")
            ok = False
    return ok


def data_md5s():
    """{file name: md5} of the Avoiding demonstrations (``DATA_MD5SUMS``)."""
    return {name: digest for digest, name in (line.split() for line in DATA_MD5SUMS.strip().splitlines())}


def verify_data(dest):
    ok = True
    table = data_md5s()
    for name, digest in table.items():
        path = os.path.join(dest, name)
        if not os.path.exists(path):
            print(f"missing {path}")
            ok = False
        elif md5(path) != digest:
            print(f"md5 mismatch {path}")
            ok = False
    extra = sorted(set(os.listdir(dest)) - set(table)) if os.path.isdir(dest) else []
    if extra:
        print(f"unexpected files in {dest} (the dataset loader reads every file there): {extra}")
        ok = False
    return ok


def hf_download(repo_id, revision, filename, dst, tmp_dir=None):
    """Download ``filename`` of the model repo to ``dst`` (via a temporary folder in ``tmp_dir``, default the folder of
    ``dst``); exits with a message if that is not possible."""
    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import (EntryNotFoundError, HfHubHTTPError, LocalEntryNotFoundError,
                                           RepositoryNotFoundError, RevisionNotFoundError)
    except ImportError:
        sys.exit("huggingface_hub is not installed: pip install huggingface_hub")
    tmp = tempfile.mkdtemp(dir=tmp_dir or os.path.dirname(dst), prefix=".hf_download_")
    try:
        path = hf_hub_download(repo_id, filename, revision=revision, local_dir=tmp)  # uses the HF token if logged in
        os.replace(path, dst)
    except RepositoryNotFoundError:
        sys.exit(f"Cannot access the Hugging Face repo {repo_id}: it does not exist, or it is private and you are not "
                 f"logged in. If you have access, log in with `hf auth login` (older huggingface_hub: "
                 f"`huggingface-cli login`) or set HF_TOKEN, then retry.")
    except RevisionNotFoundError:
        sys.exit(f"Revision {revision!r} not found in the Hugging Face repo {repo_id}.")
    except LocalEntryNotFoundError:
        sys.exit("Cannot reach huggingface.co (offline?); check the network connection and retry.")
    except EntryNotFoundError:
        sys.exit(f"{filename} not found in the Hugging Face repo {repo_id} (revision {revision}).")
    except HfHubHTTPError as e:
        sys.exit(f"{filename}: download failed: {e}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", default=DEFAULT_DEST, help="directory of the <seed>/ folders (default "
                    "checkpoints/avoiding-d3il/H8_K20_Dmodels.GaussianDiffusion)")
    ap.add_argument("--verify-only", action="store_true", help="only check the files and md5s in --dest "
                    "(with --data: in --data-dest)")
    ap.add_argument("--data", action="store_true", help="download the D3IL Avoiding demonstrations (96 .pkl) instead "
                    "of the checkpoints")
    ap.add_argument("--data-dest", default=DATA_DEST, help="folder of the demonstrations (default "
                    "third_party/d3il/environments/dataset/data/avoiding/data)")
    ap.add_argument("--repo-id", default=REPO_ID, help=f"Hugging Face model repo (default {REPO_ID})")
    ap.add_argument("--revision", default="main", help="branch, tag or commit of the repo (default main)")
    args = ap.parse_args()
    if args.data:
        dest = os.path.abspath(args.data_dest)
        if not args.verify_only:
            if dest == DATA_DEST and not os.path.isdir(os.path.join(D3IL_DIR, "environments")):
                sys.exit("third_party/d3il not found: clone and patch D3IL first (README, Installation), or pass "
                         "--data-dest")
            os.makedirs(dest, exist_ok=True)
            table = data_md5s()
            todo = [name for name, digest in table.items()
                    if not (os.path.exists(os.path.join(dest, name)) and md5(os.path.join(dest, name)) == digest)]
            if todo:
                print(f"downloading {len(todo)}/{len(table)} files {args.repo_id}/{DATA_FOLDER}/env_*.pkl -> {dest}",
                      flush=True)
            for name in todo:
                dst = os.path.join(dest, name)
                if os.path.lexists(dst):
                    os.remove(dst)
                # temporary folder next to (not inside) dest: the dataset loader reads every file in dest
                hf_download(args.repo_id, args.revision, f"{DATA_FOLDER}/{name}", dst, tmp_dir=os.path.dirname(dest))
        ok = verify_data(dest)
        print("ok" if ok else "verification FAILED")
        sys.exit(0 if ok else 1)
    if not args.verify_only:
        for seed, digest in CHECKPOINTS.items():
            os.makedirs(os.path.join(args.dest, seed), exist_ok=True)
            for name in FILES:
                dst = os.path.join(args.dest, seed, name)
                if os.path.exists(dst) and (name != "state_best.pt" or md5(dst) == digest):
                    continue
                if os.path.lexists(dst):
                    os.remove(dst)
                print(f"downloading {args.repo_id}/{REPO_FOLDER}/{seed}/{name}", flush=True)
                hf_download(args.repo_id, args.revision, f"{REPO_FOLDER}/{seed}/{name}", dst)
    ok = verify(args.dest)
    print("ok" if ok else "verification FAILED")
    sys.exit(0 if ok else 1)


# md5sum format; = d3il/data/avoiding/MD5SUMS of the Hugging Face repo
DATA_MD5SUMS = """
52fa8d99965c25674179ee258750f378  env_000_00.pkl
403e29e11d8f9a48a71c85b822fdf52d  env_001_00.pkl
4c1b5738314b137f33c6d789e35e1bbd  env_002_00.pkl
4509ce519c46fcdd25c6bd552936179c  env_003_00.pkl
d00e3a5753755950cb6319613502f16f  env_004_00.pkl
00cf7f2e8e2d80e32f30f231111e0d24  env_005_00.pkl
f8359c5b411fa234510a8d1cc2bf634f  env_006_00.pkl
ba6280f63822d0ddf71fdea703eaef39  env_007_00.pkl
e3f85b151820166789c13777a8a235ef  env_009_00.pkl
8aa76034465a0a63017ee3876a0c1a8a  env_010_00.pkl
ada5ea7ef485847e235b4f6319e84a2c  env_011_00.pkl
40696276b460f8764acf82b6c305230f  env_012_00.pkl
5d9bd715b4ee891e0875f0ad08733b76  env_013_00.pkl
e9c31156c3b88f6bd3e64da52acceff1  env_014_00.pkl
cd3d7b3bf8fa05099959cefbeb07a1a4  env_015_00.pkl
8bfc1118b9fd645a6a2d0c41c3641a55  env_016_00.pkl
d383dab3cf3781481dfa58183e91b572  env_017_00.pkl
52329f618ce1e073fb5239f6a258b641  env_018_00.pkl
42055e671eeed9092eb8098881986faa  env_019_00.pkl
94162013566f7932eb7082ebe1bff5e5  env_020_00.pkl
518a54ce35aa8e083c4048f892a97e51  env_021_00.pkl
e60410645c82b35275d9637c8ea30334  env_022_00.pkl
3c9aa9a88b640a8958423b8495d73e4a  env_023_00.pkl
90d5a63d5819df8bd8708ffaec3d6410  env_024_00.pkl
69ae199b4aada9b733dcddca5818a779  env_025_00.pkl
f5e3185f5b7a49bd89d137d63f3d0907  env_026_00.pkl
5ec5e8267af6e5c2940727e06b81bff3  env_027_00.pkl
4d39c4ad4b3ce418008af473c291cc97  env_028_00.pkl
3c371e7ed0c0e87fb58494d5a0f56d65  env_029_00.pkl
c09b7febfa411d92c836918e07690da4  env_030_00.pkl
137f9778e3cf1d11fb4476ea41d6dcf2  env_031_00.pkl
34f56de72cbcf362d3d56659ff2451d9  env_032_00.pkl
f850f69eb5fcde3e12320960fab9f718  env_033_00.pkl
ffebf0a82c437ef77e8b7054cc4bdcfb  env_034_00.pkl
9acc18d4c20599ecf81a012ce71b0a47  env_035_00.pkl
c5133aac37fe0873622f9b54c494baf8  env_036_00.pkl
5f0fcd88d233d4553951a8278f653ec9  env_037_00.pkl
a319314870b9512b01b9d7fbee0ed2b0  env_038_00.pkl
c09054461c067899b23ce6933f25b653  env_039_00.pkl
401e61f425f804c1dbf3428fa7206b4c  env_040_00.pkl
3063b50b4ddf4b550e4a0aac8614e43f  env_041_00.pkl
798e1b0e1de1bc54cedbc33a2e4343a0  env_042_00.pkl
d4599c27c9f2a2fdecc09dfff88e11b8  env_043_00.pkl
4580f26f03aad1d0c98ace1a651cf031  env_044_00.pkl
4655388744c1155de9f523337923d534  env_045_00.pkl
c93f5734692bb8772f2e80ce99d6310b  env_046_00.pkl
75240bc2658b2a78ccc230921c7810cf  env_047_00.pkl
9d01164fac67beaf3597fc5a4d1665cc  env_048_00.pkl
3d818a97ce7059d53c2ae2e326be5c00  env_049_00.pkl
eb8e9a7a70cf8de1cdca84eb9e899539  env_050_00.pkl
8de2e4bf5637ba358ffbc531ffbc8958  env_051_00.pkl
924524fe379c2db329da755c8437def2  env_052_00.pkl
8590bf80c03b37cc5d696752d22119ef  env_053_00.pkl
d2d5b08f4545c2c8ca8807070fc8a9ef  env_054_00.pkl
913a36659e982943e2e7ce7d687b173c  env_055_00.pkl
f30d8b091bbe9e357cbb5f2b1438a43c  env_056_00.pkl
ad09a6d25a06516049aa6e61b3fb3997  env_057_00.pkl
f5182ef70f5de3edade2e36f0d67d35a  env_058_00.pkl
f5b59d12f5cf1884c3609aa2e5796ba8  env_059_00.pkl
2eb1e077ff0992ef45cd232bc271986e  env_060_00.pkl
abdb94ae9aec72108d4c10c87dea87fe  env_061_00.pkl
b3f529411d13bbef2f57bd2d18c2d156  env_062_00.pkl
9fbd177fa748ecfdf0a3a2e7462d063d  env_063_00.pkl
559b9e14fb06af516ff40d587093516b  env_064_00.pkl
d464cb22cbdff4454283e3621c8ccae0  env_065_00.pkl
e8cb7521a23711b5e7924070f47ad7c6  env_066_00.pkl
b2b7ce4e338a90620c5c9dc1a1414ca1  env_067_00.pkl
875fd57815bccdea0b9dde5933f1eb7d  env_068_00.pkl
97b4d535bef964f808a7b1bb0fe6ca04  env_069_00.pkl
9adbf1f36dbd570ac690b7ca525241ce  env_070_00.pkl
35e31a4eb12f2b683029d89c56d42c4e  env_071_00.pkl
e97028bfc7ac19fea064a866d7e0ac19  env_072_00.pkl
3cff26aec13ff3f96863680c36d74d49  env_073_00.pkl
3ad7b10c0c9c2677fe084c446d1b6438  env_074_00.pkl
26939de54c12788d915985e6c3320c66  env_075_00.pkl
27e583790effdc04527f0e55ff370d4d  env_076_00.pkl
4209672fe037217b0a81ea844606db89  env_077_00.pkl
19edab09b4ecec2f3c7e49b89b2744cb  env_078_00.pkl
a3f4638b03224d824206beaa6e7a9eb9  env_079_00.pkl
bb26c1b575eb0150b0cbfda7974e495a  env_080_00.pkl
8970b45d4ebc705a0b91d617d57dfb71  env_081_00.pkl
3c23dca60df2bf922def252ec0ee0fdf  env_082_00.pkl
786e343a04774a4b474eb244262e912b  env_083_00.pkl
c74cb116360c6307739e3e1cc8bafa76  env_084_00.pkl
5f7cd66d6fda9b876ba5b6e35ff51a32  env_085_00.pkl
e29d2f0cf6231ed880ef2016f6585fd0  env_086_00.pkl
6d3fe5077ce5ebff343575aa204fd38d  env_087_00.pkl
d3038a000598d86239538d6f0c4e9e41  env_088_00.pkl
d3f95243f0d9bbc63f98c7c346e598ed  env_089_00.pkl
239ac56f0dfa918e7d1a36cc8c0c571d  env_090_00.pkl
f65b7772dfbd8d13bbc07839130bdb6c  env_091_00.pkl
30ec8e4bd7865160debbad0d9c487791  env_092_00.pkl
a0dc185509030714ebb0f77e27494d6e  env_093_00.pkl
ef1380c0217e165ceeb51be4d5363ea2  env_094_00.pkl
6f9296e171e11f9723959dae2aec6481  env_095_00.pkl
fc56b4841c9efd837216d6adf6e2066e  env_096_00.pkl
"""

if __name__ == "__main__":
    main()
