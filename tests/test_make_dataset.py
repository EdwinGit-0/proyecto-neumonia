from pathlib import Path

import pandas as pd

from src.data.make_dataset import (
    recopilar_registros_imagenes,
    construir_resumen_dataset,
    guardar_figuras_eda,
)


def _crear_imagen_prueba(path: Path, width: int = 32, height: int = 32, color_mode: str = "RGB") -> None:
    from PIL import Image

    image = Image.new(color_mode, (width, height), color="white")
    image.save(path)


def test_recopilar_registros_imagenes_cuenta_imagenes_por_conjunto_y_clase(tmp_path: Path) -> None:
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
            _crear_imagen_prueba(folder / f"img_{index}.png")

    records = recopilar_registros_imagenes(data_dir)

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


def test_obtener_resumen_dataset_detecta_imagenes_corruptas_y_duplicadas(tmp_path: Path) -> None:
    data_dir = tmp_path / "chest_xray"
    train_dir = data_dir / "train" / "NORMAL"
    train_dir.mkdir(parents=True, exist_ok=True)

    ruta_imagen = train_dir / "sample.png"
    duplicate_path = train_dir / "sample_copy.png"
    _crear_imagen_prueba(ruta_imagen)
    ruta_imagen.read_bytes()
    duplicate_path.write_bytes(ruta_imagen.read_bytes())

    corrupt_path = train_dir / "broken.png"
    corrupt_path.write_bytes(b"not-a-valid-image")

    summary = construir_resumen_dataset(data_dir)

    assert summary["total_images"] == 3
    assert summary["counts_by_split_and_class"]["train"]["NORMAL"] == 3
    assert len(summary["duplicate_paths"]) == 1
    assert corrupt_path in summary["corrupt_images"]


def test_guardar_figuras_eda_crea_archivos(tmp_path: Path) -> None:
    records = pd.DataFrame(
        [
            {"split": "train", "label": "NORMAL", "path": "/tmp/norm1.png", "extension": ".png", "file_size_bytes": 200, "width": 224, "height": 224, "mode": "RGB"},
            {"split": "train", "label": "PNEUMONIA", "path": "/tmp/pneu1.png", "extension": ".png", "file_size_bytes": 250, "width": 224, "height": 224, "mode": "RGB"},
            {"split": "val", "label": "NORMAL", "path": "/tmp/norm2.png", "extension": ".png", "file_size_bytes": 180, "width": 224, "height": 224, "mode": "RGB"},
        ]
    )

    directorio_salida = tmp_path / "figures"
    created = guardar_figuras_eda(records, directorio_salida)

    assert set(created.keys()) == {"distribution", "sizes", "dimensions"}
    for path in created.values():
        assert path.exists()
        assert path.suffix == ".png"
