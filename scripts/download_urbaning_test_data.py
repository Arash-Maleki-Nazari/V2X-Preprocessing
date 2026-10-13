"""Download one UrbanIng sequence using the official devkit."""
import argparse
from project_paths import dataset_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", "--destination", dest="root")
    parser.add_argument("--config")
    parser.add_argument("--sequence", help="Official sequence name (devkit default if omitted)")
    args = parser.parse_args(argv)
    destination = dataset_path("urbaning", args.root, args.config, must_exist=False)
    destination.mkdir(parents=True, exist_ok=True)
    from urbaning.data import download_one_sequence
    options = {"download_dir": str(destination)}
    if args.sequence is not None:
        options["sequence_name"] = args.sequence
    download_one_sequence(**options)


if __name__ == "__main__":
    main()
