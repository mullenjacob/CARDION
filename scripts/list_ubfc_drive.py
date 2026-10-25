"""List a user-supplied UBFC Drive folder without downloading its recordings.

UBFC was considered but was not used in the CARDION benchmark.
Obtain access and the appropriate folder URL from the dataset provider.
"""
import argparse
import json
import gdown


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder-url", required=True, help="Dataset folder URL supplied by its provider")
    parser.add_argument("--output", required=True, help="Output file for the folder index")
    args = parser.parse_args()
    files = gdown.download_folder(url=args.folder_url, skip_download=True, quiet=True, use_cookies=False)
    if files is None:
        raise SystemExit("The folder could not be listed; check its URL and access.")
    index = [{"id": f.id, "path": f.path.replace("\\", "/")} for f in files]
    with open(args.output, "w", encoding="utf-8") as stream:
        json.dump(index, stream, indent=1)
    print("Indexed files:", len(index))


if __name__ == "__main__":
    main()
