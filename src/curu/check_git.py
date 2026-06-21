# This file is part of https://github.com/KurtBoehm/curu.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from argparse import ArgumentParser
from pathlib import Path

from colorama import Fore, Style


def handle_repo(folder: Path, update_remote: bool):
    from git import BadName, GitCommandError, InvalidGitRepositoryError, Repo

    try:
        repo = Repo(folder)
    except InvalidGitRepositoryError:
        for p in sorted(folder.iterdir()):
            if p.is_dir():
                handle_repo(p, update_remote=update_remote)
        return
    if repo.is_dirty(untracked_files=True):
        index = repo.index
        untracked = [p for p in repo.untracked_files if not (folder / p).is_dir()]
        unstaged = [f.a_path for f in index.diff(None)]
        try:
            staged = [f.a_path for f in index.diff("HEAD")]
        except BadName:
            return
        if len(untracked) + len(unstaged) + len(staged) > 0:
            print(f"{Style.BRIGHT}{Fore.RED}{folder} is dirty:{Style.RESET_ALL}")
            print(f"untracked: {untracked}")
            print(f"unstaged: {unstaged}")
            print(f"staged: {staged}")

    try:
        local_branch = repo.active_branch
    except TypeError:
        return
    if len(repo.remotes) == 0:
        print(f"{Fore.GREEN}{folder} has no remotes{Fore.RESET}")
        return
    if update_remote:
        remote = repo.remote()
        try:
            remote.update()
        except GitCommandError:
            print(
                f"{Fore.RED}The remote {remote.url} of {folder} "
                + f"cannot be updated{Fore.RESET}"
            )

    remote_branch = f"origin/{local_branch}"
    commits = list(repo.iter_commits(f"{remote_branch}..{local_branch}"))
    if len(commits) > 0:
        print(
            f"{Style.BRIGHT}{Fore.YELLOW}"
            + f"{folder} has non-pushed commits:"
            + f"{Fore.RESET}{Style.RESET_ALL}"
        )
        for commit in commits:
            print(f"{commit.hexsha} {commit.summary}")


def run() -> None:
    parser = ArgumentParser(
        description="Check whether there are uncommitted changes in a Git repository.",
    )
    parser.add_argument("--update-remote", "-u", action="store_true")
    parser.add_argument("folder", type=Path, nargs="+")
    args = parser.parse_args()
    update_remote: bool = args.update_remote
    folder: list[Path] = args.folder

    for p in folder:
        handle_repo(p, update_remote)
