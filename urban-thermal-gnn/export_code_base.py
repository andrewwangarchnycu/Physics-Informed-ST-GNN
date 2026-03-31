import pathspec
from pathlib import Path
from typing import List


SEPARATOR: str = "-" * 92

# [REMOVED_ZH:7]「[REMOVED_ZH:6]」([REMOVED_ZH:2]and .gitignore [REMOVED_ZH:2])
CUSTOM_IGNORES: List[str] = [
    ".vscode/",
    "export_code_base.py",
    "*tmpclaude-*"
]


def get_ignore_spec(root_dir: Path) -> pathspec.PathSpec:
    """
    [REMOVED_ZH:2] .gitignore and[REMOVED_ZH:4]，[REMOVED_ZH:5] PathSpec [REMOVED_ZH:2]。
    """
    ignore_lines: List[str] = list(CUSTOM_IGNORES)
    gitignore_path = root_dir / ".gitignore"

    if gitignore_path.exists():
        with gitignore_path.open("r", encoding="utf-8") as f:
            ignore_lines.extend(f.readlines())

    return pathspec.PathSpec.from_lines("gitwildmatch", ignore_lines)


def generate_folder_tree(
    dir_path: Path, spec: pathspec.PathSpec, root_dir: Path, prefix: str = ""
) -> str:
    """
    [REMOVED_ZH:14]，[REMOVED_ZH:11]and[REMOVED_ZH:3]。
    """
    tree_str = ""

    # [REMOVED_ZH:13]
    if prefix == "":
        tree_str += f"{dir_path.name}/\n"

    items = list(dir_path.iterdir())
    valid_items = []

    for item in items:
        # [REMOVED_ZH:4] Git [REMOVED_ZH:5]
        if item.name == ".git":
            continue

        # [REMOVED_ZH:3] POSIX [REMOVED_ZH:6] pathspec [REMOVED_ZH:2]
        rel_path = item.relative_to(root_dir).as_posix()

        # [REMOVED_ZH:2]：[REMOVED_ZH:11]，[REMOVED_ZH:2] `__pycache__/` [REMOVED_ZH:17]
        if item.is_dir():
            rel_path += "/"

        # [REMOVED_ZH:7]，[REMOVED_ZH:7]
        if not spec.match_file(rel_path):
            valid_items.append(item)

    # [REMOVED_ZH:2]：[REMOVED_ZH:7]，[REMOVED_ZH:5]，[REMOVED_ZH:9]
    valid_items.sort(key=lambda x: (x.is_file(), x.name))

    for i, item in enumerate(valid_items):
        is_last = i == len(valid_items) - 1
        connector = "└─ " if is_last else "├─ "

        tree_str += f"{prefix}{connector}{item.name}{'/' if item.is_dir() else ''}\n"

        # [REMOVED_ZH:6]，[REMOVED_ZH:8]
        if item.is_dir():
            extension = "   " if is_last else "│  "
            tree_str += generate_folder_tree(item, spec, root_dir, prefix + extension)

    return tree_str


def find_python_files(root_dir: Path, spec: pathspec.PathSpec) -> List[Path]:
    """
    [REMOVED_ZH:11] .py [REMOVED_ZH:2]。
    """
    valid_files: List[Path] = []
    patterns: List[str] = ["*.py", "*.yaml"]

    files = []
    for pattern in patterns:
        files.extend(root_dir.rglob(pattern))

    for file_path in files:
        rel_path = file_path.relative_to(root_dir).as_posix()
        if ".git" not in file_path.parts and not spec.match_file(rel_path):
            valid_files.append(file_path)

    return sorted(valid_files)


def export_codebase(root_dir: Path, output_file: Path) -> None:
    """
    [REMOVED_ZH:6]and Python [REMOVED_ZH:12]。
    """
    # 1. [REMOVED_ZH:6]
    spec = get_ignore_spec(root_dir)

    # 2. [REMOVED_ZH:8]and[REMOVED_ZH:5]
    py_files = find_python_files(root_dir, spec)
    tree_output = generate_folder_tree(root_dir, spec, root_dir)

    with output_file.open("w", encoding="utf-8") as f:
        # [REMOVED_ZH:7]
        f.write("code folder structure...\n")
        f.write(SEPARATOR + "\n")
        f.write(tree_output)
        f.write("\n\n")

        # [REMOVED_ZH:2] Codebase [REMOVED_ZH:2]
        f.write("code base\n")
        for file_path in py_files:
            rel_path: Path = file_path.relative_to(root_dir)

            f.write(SEPARATOR + "\n")
            f.write(str(rel_path) + "\n")

            try:
                code: str = file_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                code = file_path.read_text(encoding="latin-1")

            f.write(code)
            f.write("\n")

    print(f"Exported folder tree and {len(py_files)} python files -> {output_file}")


def main() -> None:
    """
    Entry point.
    """
    root_dir: Path = Path(
        r"C:\Users\user\Desktop\UTIC GNN\Physics-Informed ST-GNN\urban-thermal-gnn"
    )  # current directory
    output_file: Path = Path("urban-thermal-gnn.txt")

    export_codebase(root_dir, output_file)


if __name__ == "__main__":
    main()
