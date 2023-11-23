# coding=utf-8
import json
from pathlib import Path

import numpy as np


def main(args):
    if args.data_path is not None:
        all_filenames = list(Path(args.data_path).iterdir())

        # keep only the filenames that contain the word "transversal"
        all_filenames = [filename for filename in all_filenames if "transversal" in str(filename)]

        # val set should be p9 and p19
        val_set = [
            str(filename)
            for filename in all_filenames
            if "dd969854-ec56-4ccc-b7ac-ff4cd7735095" in str(filename)
            or "d089cbe0-48b3-4ae2-9475-53ca89ee90fe" in str(filename)
        ]
        # test set should be p8, p10, and p13
        test_set = [
            str(filename)
            for filename in all_filenames
            if "ec00945c-ad90-46b7-8c38-a69e9e801074" in str(filename)
            or "efa383b6-9446-438a-9901-1fe951653dbd" in str(filename)
            or "ee2efe48-1e9d-480e-9364-e53db01532d4" in str(filename)
        ]
        # train set should be the rest
        train_set = [
            str(filename)
            for filename in all_filenames
            if "dd969854-ec56-4ccc-b7ac-ff4cd7735095" not in str(filename)
            and "d089cbe0-48b3-4ae2-9475-53ca89ee90fe" not in str(filename)
            and "ec00945c-ad90-46b7-8c38-a69e9e801074" not in str(filename)
            and "efa383b6-9446-438a-9901-1fe951653dbd" not in str(filename)
            and "ee2efe48-1e9d-480e-9364-e53db01532d4" not in str(filename)
        ]
    else:
        all_filenames = list(Path(args.train_path).iterdir())
        all_filenames += list(Path(args.val_path).iterdir())
        all_filenames += list(Path(args.test_path).iterdir())

        # keep only the filenames that contain the word "transversal"
        all_filenames = [filename for filename in all_filenames if "transversal" in filename.name]

        # val set should be p9 and p19
        val_set = [str(filename) for filename in all_filenames if "p9" in filename.name or "p19" in filename.name]
        # test set should be p8, p10, and p13
        test_set = [
            str(filename)
            for filename in all_filenames
            if "p8" in filename.name or "p10" in filename.name or "p13" in filename.name
        ]
        # train set should be the rest
        train_set = [
            str(filename) for filename in all_filenames if filename not in val_set and filename not in test_set
        ]

        # create a directory to store the folds
        output_path = Path(args.output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        with open(output_path / f"train.json", "w") as f:
            json.dump(train_set, f)
        with open(output_path / f"val.json", "w") as f:
            json.dump(val_set, f)
        with open(output_path / f"test.json", "w") as f:
            json.dump(test_set, f)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=Path,
        default=None,
        help="Path to the directory containing the data. If provided, the train, val, and test paths are ignored.",
    )
    parser.add_argument("--train_path", type=Path, default="data/train", help="Path to the train directory.")
    parser.add_argument("--val_path", type=Path, default="data/val", help="Path to the val directory.")
    parser.add_argument("--test_path", type=Path, default="data/test", help="Path to the test directory.")
    parser.add_argument("--output_path", type=Path, default="data/folds", help="Path to the output directory.")
    args = parser.parse_args()
    main(args)
