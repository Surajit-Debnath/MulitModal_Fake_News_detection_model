import argparse
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm


def download_image(row_id, image_url, images_dir, timeout=10):

    image_path = os.path.join(images_dir, row_id + ".jpg")

    if os.path.exists(image_path):
        return None

    try:
        with urllib.request.urlopen(image_url, timeout=timeout) as response:
            with open(image_path, "wb") as f:
                f.write(response.read())
        return None
    except Exception:
        return row_id


def download_images(tsv_path, images_dir="images", num_workers=16, timeout=10):

    df = pd.read_csv(tsv_path, sep="\t")
    df = df.fillna("")
    df = df[(df["hasImage"] == True) & (df["image_url"] != "")]

    os.makedirs(images_dir, exist_ok=True)

    failed_ids = []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(
                download_image, row["id"], row["image_url"], images_dir, timeout
            ): row["id"]
            for _, row in df.iterrows()
        }

        for future in tqdm(as_completed(futures), total=len(futures)):
            failed_id = future.result()
            if failed_id:
                failed_ids.append(failed_id)

    print(f"Done. {len(df) - len(failed_ids)} succeeded, {len(failed_ids)} failed.")

    return failed_ids


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Download Fakeddit images referenced by image_url in a tsv file"
    )
    parser.add_argument(
        "tsv_path",
        type=str,
        help="Path to a Fakeddit tsv file (e.g. multimodal_test_public.tsv)",
    )
    parser.add_argument("--images_dir", type=str, default="images")
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--timeout", type=int, default=10)

    args = parser.parse_args()

    failed = download_images(
        args.tsv_path, args.images_dir, args.num_workers, args.timeout
    )

    if failed:
        failed_path = "failed_downloads.csv"
        pd.DataFrame({"id": failed}).to_csv(failed_path, index=False)
        print(f"Saved {len(failed)} failed ids to {failed_path}")
