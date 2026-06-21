# 🛠️ Curu: Various Programming Utilities

Curu (Quenya “skill”, Sindarin “skill, craft, magic”) is a lightweight toolbox of Python helpers that simplify everyday programming.
It currently provides these CLI tools:

| Command        | What it does                                                                                        |
| -------------- | --------------------------------------------------------------------------------------------------- |
| `check-git`    | Recursively finds Git repos under a folder, shows uncommitted changes and non-pushed commits.       |
| `parse-nested` | Pretty-prints and colorizes nested/templated C++-like expressions (supports clipboard input).       |
| `icon-scale`   | Generates multiple icon PNGs at standard sizes from a single source image using `libvips`/`pyvips`. |

## 📦 Installation

Curu is [available on PyPI](https://pypi.org/project/curu/) and can be installed with `pip`, `uv`, or `pipx`:

```sh
pip install curu
# or
uv tool install curu
# or
pipx install curu
```

## 🚀 Usage

### `check-git`

```sh
# Check a directory (recursively) for dirty Git repos and non-pushed commits
check-git path/to/folder

# Also update the default remote before checking
check-git --update-remote path/to/folder
```

Typical output:

```text
/path/to/repo is dirty:
untracked: ['new_file.py']
unstaged: ['modified_file.py']
staged: ['staged_change.py']

/path/to/other_repo has non-pushed commits:
f3c2e1a Add new feature
...
```

### `parse-nested`

```sh
# Format text passed as an argument
parse-nested "std::vector<std::pair<int, std::string>>"

# Format clipboard contents (no positional args → reads from clipboard)
parse-nested

# Limit maximum line width (default: 80)
parse-nested -w 100 "some very long templated type"

# Only print formatted output; do not overwrite clipboard
parse-nested -p

# Cut off nesting after a certain depth
parse-nested -d 3 "very deeply nested type..."
```

By default (no `-p` and no `-d`), when reading from the clipboard, it prints a colorized version to the terminal and copies a cleaned, uncolored version back to the clipboard.

### `icon-scale`

```sh
# Generate multiple PNGs at standard sizes from one source image
icon-scale path/to/icon.png path/to/output/icon

# This writes files like:
#   path/to/output/icon-512.png
#   path/to/output/icon-384.png
#   ...
#   path/to/output/icon-48.png
```

The second argument is treated as a prefix: `icon-scale src.png prefix`
creates `prefix-512.png`, `prefix-384.png`, etc.

## 📜 Licence

This library is licensed under the terms of the Mozilla Public Licence 2.0, provided in [`License`](https://github.com/KurtBoehm/curu/blob/main/License).
