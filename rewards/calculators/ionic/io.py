"""Input helpers shared by ionic-conductor command-line tools."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from pymatgen.core import Structure


STRUCTURE_SUFFIXES = {".cif", ".json", ".mcif", ".poscar", ".vasp"}
ASE_SUFFIXES = {".extxyz", ".xyz"}
SPECIAL_FILENAMES = {"POSCAR", "CONTCAR"}


def discover_structure_paths(inputs: Iterable[str | Path]) -> list[Path]:
    """Expand structure files and directories into deterministic unique paths."""

    discovered: list[Path] = []
    for raw_path in inputs:
        path = Path(raw_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Structure input does not exist: {path}")
        if path.is_dir():
            candidates = sorted(item for item in path.rglob("*") if item.is_file())
        else:
            candidates = [path]
        for candidate in candidates:
            if (
                candidate.suffix.lower() in STRUCTURE_SUFFIXES | ASE_SUFFIXES
                or candidate.name.upper() in SPECIAL_FILENAMES
            ):
                discovered.append(candidate.resolve())

    # Preserve the deterministic discovery order while removing duplicates.
    return list(dict.fromkeys(discovered))


def load_structure_file(path: str | Path) -> list[Structure]:
    """Load one or more structures from a supported crystal file."""

    source = Path(path)
    if source.suffix.lower() in ASE_SUFFIXES:
        try:
            import ase.io
            from pymatgen.io.ase import AseAtomsAdaptor
        except ImportError as exc:
            raise ImportError("ASE is required to read XYZ or extxyz files") from exc
        atoms = ase.io.read(source, index=":")
        return [AseAtomsAdaptor.get_structure(item) for item in atoms]
    return [Structure.from_file(source)]
