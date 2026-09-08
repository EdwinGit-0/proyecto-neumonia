from pathlib import Path

import pandas as pd

from src.data.make_dataset import (
    collect_image_records,
    get_dataset_summary,
    save_eda_figures,
)


def _create_test_image(path: Path, width: int = 32, height: int = 32, color_mode: str = "RGB") -> None:
    from PIL import Image

    image = Image.new(color_mode, (width, height), color="white")
    image.save(path)


def test_collect_image_records_counts_images_by_split_and_class(tmp_path: Path) -> None:
    data_dir = tmp_path / "chest_xray"
    for split, label, count in [
        ("train", "NORMAL", 2),
        ("train", "PNEUMONIA", 1),
        ("val", "NORMAL", 1),
        ("test", "PNEUMONIA", 2),
    ]:
        folder = data_dir / split / label
        folder.mkdir(parents=True, exist_ok=True)
        for index in range(count):
            _create_test_image(folder / f"img_{index}.png")

    records = collect_image_records(data_dir)

    assert isinstance(records, pd.DataFrame)
    assert records["split"].nunique() == 3
    assert records["label"].nunique() == 2
    assert len(records) == 6
    assert records.groupby(["split", "label"]).size().to_dict() == {
        ("train", "NORMAL"): 2,
        ("train", "PNEUMONIA"): 1,
        ("val", "NORMAL"): 1,
        ("test", "PNEUMONIA"): 2,
    }


def test_get_dataset_summary_detects_corrupt_and_duplicate_images(tmp_path: Path) -> None:
    data_dir = tmp_path / "chest_xray"
    train_dir = data_dir / "train" / "NORMAL"
    train_dir.mkdir(parents=True, exist_ok=True)

    image_path = train_dir / "sample.png"
    duplicate_path = train_dir / "sample_copy.png"
    _create_test_image(image_path)
    image_path.read_bytes()
    duplicate_path.write_bytes(image_path.read_bytes())

    corrupt_path = train_dir / "broken.png"
    corrupt_path.write_bytes(b"not-a-valid-image")

    summary = get_dataset_summary(data_dir)

    assert summary["total_images"] == 3
    assert summary["counts_by_split_and_class"]["train"]["NORMAL"] == 3
    assert len(summary["duplicate_paths"]) == 1
    assert corrupt_path in summary["corrupt_images"]


def test_save_eda_figures_creates_outputs(tmp_path: Path) -> None:
    records = pd.DataFrame(
        [
            {"split": "train", "label": "NORMAL", "path": "/tmp/norm1.png", "extension": ".png", "file_size_bytes": 200, "width": 224, "height": 224, "mode": "RGB"},
            {"split": "train", "label": "PNEUMONIA", "path": "/tmp/pneu1.png", "extension": ".png", "file_size_bytes": 250, "width": 224, "height": 224, "mode": "RGB"},
            {"split": "val", "label": "NORMAL", "path": "/tmp/norm2.png", "extension": ".png", "file_size_bytes": 180, "width": 224, "height": 224, "mode": "RGB"},
        ]
    )

    output_dir = tmp_path / "figures"
    created = save_eda_figures(records, output_dir)

    assert set(created.keys()) == {"distribution", "sizes", "dimensions"}
    for path in created.values():
        assert path.exists()
        assert path.suffix == ".png"
