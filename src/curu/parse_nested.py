# This file is part of https://github.com/KurtBoehm/curu.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import re
from argparse import ArgumentParser
from dataclasses import dataclass, field
from subprocess import PIPE
from subprocess import run as _run
from typing import Final, final

import pyperclip as pc
from colorama import Fore
from pydantic import BaseModel, TypeAdapter

INDENT = "  "
LINEDENT = "▏" + INDENT[1:]
RAINBOW = [Fore.RED, Fore.YELLOW, Fore.GREEN, Fore.CYAN, Fore.BLUE, Fore.MAGENTA]
LIGHT_RAINBOW = [
    Fore.LIGHTRED_EX,
    Fore.LIGHTYELLOW_EX,
    Fore.LIGHTGREEN_EX,
    Fore.LIGHTCYAN_EX,
    Fore.LIGHTBLUE_EX,
    Fore.LIGHTMAGENTA_EX,
]
SPECIALS = {"->"}
PREFIX_SUFFIX_PAIRS: set[tuple[str, str]] = {
    ("(", ")"),
    ("[", "]"),
    ("{", "}"),
    ("<", ">"),
    ("“", "”"),
    ("‘", "’"),
}
PREFIXES = {p for p, _ in PREFIX_SUFFIX_PAIRS}
SUFFIXES = {s for _, s in PREFIX_SUFFIX_PAIRS}
PREFIX_OF_SUFFIX = {
    s: {p for p, so in PREFIX_SUFFIX_PAIRS if so == s} for s in SUFFIXES
}
SUFFIX_OF_PREFIX = {p: s for p, s in PREFIX_SUFFIX_PAIRS}
INFIXES = {",", ";"}
STOP_CHAR = PREFIXES | SUFFIXES | INFIXES

NO_SPACE_BEFORE = PREFIXES | {"'"}


@final
class Style:
    RESET_ALL = "\033[0m"
    BOLD = "\033[1m"
    ITALIC = "\033[3m"


def indent_depth(line: str) -> int:
    depth = len(line) - len(line.lstrip())
    assert depth % len(INDENT) == 0, f"{depth} % len({line!r}) = {len(line)}"
    return depth // len(INDENT)


def rainbow_color(i: int) -> str:
    return RAINBOW[i % len(RAINBOW)]


def light_rainbow_color(i: int) -> str:
    return LIGHT_RAINBOW[i % len(LIGHT_RAINBOW)]


def merge_lines(txt: str, *, line_limit: int):
    @dataclass
    class Node:
        children: list["Node | str"] = field(default_factory=list)

        @staticmethod
        def _child_single(c: "Node | str") -> str:
            return c if isinstance(c, str) else c._single_string()

        def _single_string(self) -> str:
            last: Node | str = self.children[0]
            out = self._child_single(last)
            for child in self.children[1:]:
                prefix = (
                    " "
                    if isinstance(last, str)
                    and any(last.endswith(infix) for infix in INFIXES | SUFFIXES)
                    else ""
                )
                out += prefix + self._child_single(child).strip()
                last = child
            return out

        def string(self, *, line_limit: int) -> str:
            if len(short := self._single_string()) <= line_limit:
                return short

            i = 0
            children = self.children[:]
            while i < len(children):
                ci = children[i]
                if (
                    isinstance(ci, str)
                    and len(prefs := [p for p in PREFIXES if ci.endswith(p)]) > 0
                ):
                    (pre,) = prefs
                    suf = SUFFIX_OF_PREFIX[pre]
                    j = [
                        j
                        for j in range(i + 1, len(children))
                        if isinstance(c := children[j], str)
                        and c.strip().startswith(suf)
                    ][0]
                    assert j - i in (1, 2)
                    str_ada = TypeAdapter(str)
                    if j == i + 1:
                        joined = ci + str_ada.validate_python(children[j]).strip()
                    else:
                        cmid = children[i + 1]
                        assert isinstance(cmid, Node)
                        cj = str_ada.validate_python(children[j]).strip()
                        joined = ci + cmid._single_string().strip() + cj
                    if len(joined) <= line_limit:
                        children = children[:i] + [joined] + children[j + 1 :]
                        i += 1
                    else:
                        i = j
                    continue
                i += 1

            if len(children) != len(self.children):
                return Node(children=children).string(line_limit=line_limit)

            return "\n".join(
                c if isinstance(c, str) else c.string(line_limit=line_limit)
                for c in self.children
            )

    stack: list[Node] = []
    for line in txt.splitlines():
        if len(line.strip()) == 0:
            continue
        depth = indent_depth(line)
        if len(stack) > depth:
            # level exists: append
            stack = stack[: depth + 1]
            stack[-1].children.append(line)
        else:
            # level does not exist: create
            assert len(stack) == depth
            node = Node([line])
            if len(stack) > 0:
                stack[-1].children.append(node)
            stack.append(node)

    head = stack[0]
    return head.string(line_limit=line_limit)


def escape(txt: str):
    i = 0
    out = ""
    while i < len(txt):
        starts = [s for s in SPECIALS if txt[i:].startswith(s)]
        if len(starts) == 0:
            out += txt[i]
            i += 1
            continue
        [s] = starts
        out += f"\x1b{s}\x1b"
        i += len(s)
    return out


def demangle(txt: str):
    def impl(s: str) -> str:
        if not s.startswith("_Z"):
            return s
        try:
            ret = _run(["llvm-cxxfilt", s], check=True, stdout=PIPE, encoding="utf-8")
            return ret.stdout if ret.returncode == 0 else s
        except FileNotFoundError as err:
            print(err)
            return s

    return " ".join(impl(s) for s in txt.split())


def parse(txt: str, line_limit: int, depth_cutoff: int | None) -> tuple[str, str]:
    if len(txt) == 0:
        return txt, txt

    txt = demangle(txt)
    while "\n" in txt:
        txt = txt.replace("\n", " ")
    while "  " in txt:
        txt = txt.replace("  ", " ")
    esc_txt = escape(txt)

    indented = ""
    stack: list[str] = []
    remainder = esc_txt
    newline = False
    while len(remainder) > 0:
        idx = 0
        escaped = False
        while idx < len(remainder) and (escaped or remainder[idx] not in STOP_CHAR):
            if remainder[idx] == "\x1b":
                escaped = not escaped
            idx += 1
        if idx == len(remainder):
            r = remainder.strip()
            prefix = (
                " "
                if any(indented.endswith(suffix) for suffix in SUFFIXES)
                and not any(r.startswith(s) for s in NO_SPACE_BEFORE)
                else ""
            )
            indented += prefix + r
            break

        c = remainder[idx]
        prefix = f"\n{INDENT * len(stack)}" if newline else ""
        current = remainder[: idx + 1]
        remainder = remainder[idx + 1 :]

        if c in SUFFIXES:
            assert stack[-1] in PREFIX_OF_SUFFIX[c]
            stack.pop()
            indented += (
                f"{prefix}{current[:-1].rstrip()}\n{INDENT * len(stack)}{current[-1]}"
            )
            newline = False
            continue

        if c in PREFIXES:
            stack.append(c)

        indented += f"{prefix}{current.rstrip()}"
        remainder = remainder.lstrip()
        newline = True
    indented = "".join(c for c in indented if c != "\x1b")

    # Merge lines if short enough
    indented = merge_lines(indented, line_limit=line_limit)

    if depth_cutoff is not None:
        placeholder = INDENT * depth_cutoff + "…"
        reduced: list[str] = []
        for line in indented.splitlines():
            depth = indent_depth(line)
            if depth < depth_cutoff:
                reduced.append(line)
            elif depth == depth_cutoff:
                reduced.append(placeholder)
        prev = []
        while len(prev) != len(reduced):
            prev = reduced
            reduced = [
                line
                for i, line in enumerate(reduced)
                if i == 0 or reduced[i - 1] != placeholder or line != placeholder
            ]
        indented = "\n".join(reduced)
        indented = merge_lines(indented, line_limit=line_limit)

    # TODO This is a stop-gap solution!
    old_indented: str | None = None
    while indented != old_indented:
        old_indented = indented
        indented = indented.replace(":  ", ": ")
    indented = indented.replace("( )", "()")

    keywords = {
        "class",
        "const",
        "constexpr",
        "decltype",
        "false",
        "lambda",
        "requires",
        "static",
        "struct",
        "template",
        "true",
        "typename",
        "using",
    }
    builtins = {
        "auto",
        "bool",
        "char",
        "double",
        "float",
        "int",
        "long",
        "short",
        "unsigned",
        "void",
    }

    num_re = re.compile(r"(^|(?<=[^A-Za-z0-9]))(0x[0-9A-Fa-f]+|[0-9]+u?l?)")
    form_in = indented
    form_out = ""
    i = 0
    for m in num_re.finditer(form_in):
        i0, i1 = m.span()
        form_out += f"{form_in[i:i0]}{Fore.CYAN}{form_in[i0:i1]}{Style.RESET_ALL}"
        i = i1
    form_out += form_in[i:]

    name_re = re.compile(r"(^|(?<=[^A-Za-z0-9])|::)[A-Za-z_](?:[A-Za-z0-9_]| |::)+<?")
    form_in = form_out
    form_out = ""
    last_end = 0
    for m in name_re.finditer(form_in):
        start, end = m.start(), m.end()

        raw_name = form_in[start:end]
        template = raw_name.endswith("<")
        if template:
            last_color = Fore.BLUE
            suffix = "<"
            raw_name = raw_name[:-1]
        else:
            last_color = Fore.BLUE
            suffix = ""

        raw_words = raw_name.split(" ")
        words: list[str] = []
        for w in raw_words:
            if "::" not in w:
                color = (
                    f"{Fore.RED}{Style.BOLD}"
                    if w in keywords
                    else f"{Fore.BLUE}{Style.BOLD}"
                    if w in builtins
                    else Fore.GREEN
                )
                words.append(f"{color}{w}{Style.RESET_ALL}")
            else:
                parts = w.split("::")
                new_parts: list[str] = []
                for i, p in enumerate(parts):
                    color = (
                        f"{Fore.RED}{Style.BOLD}"
                        if p in keywords
                        else (
                            f"{Fore.BLUE}{Style.BOLD}"
                            if p in builtins
                            else last_color
                            if i + 1 == len(parts)
                            else Fore.MAGENTA
                        )
                    )
                    new_parts.append(f"{color}{p}{Style.RESET_ALL}")
                words.append("::".join(new_parts))
        name = " ".join(words)

        form_out += form_in[last_end:start] + name + suffix
        last_end = end
    formatted = form_out + form_in[last_end:]

    form_lines: list[str] = []
    for line in formatted.splitlines():
        depth = indent_depth(line)
        indent = "".join(
            f"{light_rainbow_color(j)}{LINEDENT}{Fore.RESET}" for j in range(depth)
        )

        # Remove multiple spaces
        prev_line: str | None = None
        line: str = line.lstrip()
        while line != prev_line:
            prev_line, line = line, line.replace("  ", " ")

        form_lines.append(indent + line)
    formatted = "\n".join(form_lines)

    return formatted, indented


class Args(BaseModel):
    txt: list[str]
    line_limit: int
    print_only: bool
    depth_cutoff: int | None


def run():
    parser = ArgumentParser()
    parser.add_argument("txt", nargs="*")
    parser.add_argument("-w", "--line-limit", type=int, default=80)
    parser.add_argument("-p", "--print-only", action="store_true")
    parser.add_argument("-d", "--depth-cutoff", type=int)
    args: Final[Args] = Args.model_validate(vars(parser.parse_args()))

    print("*" * args.line_limit)
    if len(args.txt) == 0:
        contents = pc.paste()
        formatted, unformatted = parse(contents, args.line_limit, args.depth_cutoff)
        print(formatted)

        if not args.print_only and args.depth_cutoff is None:
            pc.copy(unformatted)
    else:
        [s] = args.txt
        formatted, _ = parse(s, args.line_limit, args.depth_cutoff)
        print(formatted)
