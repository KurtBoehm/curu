#!/usr/bin/env python3
"""
Demangle Itanium C++ ABI names.

Usage:
    python demangle_cpp.py _ZN3foo3barEv ...
    cat symbols.txt | python demangle_cpp.py

API:
    demangle(name) -> demangled string, or ``name`` unchanged on failure.
    demangle_strict(name) -> demangled string, raises ``DemangleError`` on failure.

Output style follows GNU ``c++filt`` or LLVM ``cxxfilt`` reasonably closely.
"""

from __future__ import annotations

import argparse
import re
import shutil
import struct
import subprocess
import sys
from abc import ABC
from collections.abc import Iterable
from typing import Literal, NoReturn, final, override

__all__ = ["demangle", "demangle_strict", "DemangleError", "Demangler"]

Style = Literal["gcc", "llvm"]


class DemangleError(Exception):
    """Raised when a string cannot be parsed as an Itanium ABI mangled name."""


########################################################################################
# Rendering helpers
########################################################################################


def _decl(base: str, inner: str) -> str:
    """
    Attach declarator text ``inner`` to the type spelling ``base``.

    :param base: Base spelling.
    :param inner: Declarator suffix.
    :return: Combined spelling.
    """
    if not inner:
        return base
    if inner[0] in "*&":
        return f"{base}{inner}"
    return f"{base} {inner}"


def _pre(tok: str, inner: str) -> str:
    """
    Prepend a declarator token to an existing declarator.

    :param tok: Declarator token.
    :param inner: Existing declarator text.
    :return: Combined declarator.
    """
    if not inner:
        return tok
    if inner[0] in "*&(":
        return f"{tok}{inner}"
    return f"{tok} {inner}"


def _join(items: Iterable[Node], style: Style) -> str:
    """
    Join rendered items with commas.

    :param items: Items to join.
    :param style: Output style.
    :return: Joined text.
    """
    parts = [item.render(style=style) for item in items]
    parts = [part for part in parts if part]
    return ", ".join(parts)


def _targs(args: Iterable[Node], style: Style) -> str:
    """
    Render template arguments.

    GCC-style output avoids emitting ``>>`` by inserting a space.

    :param args: Template arguments.
    :param style: Output style.
    :return: Rendered template-argument list.
    """
    rendered = _join(args, style)
    if style == "gcc" and rendered.endswith(">"):
        return f"<{rendered} >"
    return f"<{rendered}>"


########################################################################################
# AST nodes: everything renders lazily so forward references work
########################################################################################


class Node(ABC):
    """Base class for AST nodes."""

    def render(self, inner: str = "", style: Style = "gcc") -> str:
        """
        Render this node.

        :param inner: Declarator suffix.
        :param style: Output style.
        :return: Rendered text.
        """
        raise NotImplementedError

    @override
    def __str__(self) -> str:
        return self.render()

    def base_name(self) -> str:
        """
        Return the node’s base name, ignoring declarators.

        :return: Base name.
        """
        return self.render()

    def as_pack(self) -> list[Node] | None:
        """
        Return pack contents if this node resolves to a pack.

        :return: Pack contents, or ``None``.
        """
        return None


@final
class Lit(Node):
    """Literal text node."""

    def __init__(self, text: str) -> None:
        self.text = text

    @override
    def render(self, inner: str = "", style: Style = "gcc") -> str:
        return _decl(self.text, inner)

    @override
    def base_name(self) -> str:
        return self.text


@final
class Qualified(Node):
    """Qualified name node."""

    def __init__(self, left: Node, right: Node) -> None:
        self.left = left
        self.right = right

    @override
    def render(self, inner: str = "", style: Style = "gcc") -> str:
        return _decl(
            f"{self.left.render(style=style)}::{self.right.render(style=style)}",
            inner,
        )

    @override
    def base_name(self) -> str:
        return self.right.base_name()


@final
class Template(Node):
    """Template-id node."""

    def __init__(self, name: Node, args: list[Node]) -> None:
        self.name = name
        self.args = args

    @override
    def render(self, inner: str = "", style: Style = "gcc") -> str:
        return _decl(
            f"{self.name.render(style=style)}{_targs(self.args, style)}", inner
        )

    @override
    def base_name(self) -> str:
        return self.name.base_name()


@final
class Ptr(Node):
    """Pointer node."""

    def __init__(self, t: Node) -> None:
        self.t = t

    @override
    def render(self, inner: str = "", style: Style = "gcc") -> str:
        return self.t.render(_pre("*", inner), style=style)


@final
class LRef(Node):
    """Lvalue reference node."""

    def __init__(self, t: Node) -> None:
        self.t = t

    @override
    def render(self, inner: str = "", style: Style = "gcc") -> str:
        return self.t.render(_pre("&", inner), style=style)


@final
class RRef(Node):
    """Rvalue reference node."""

    def __init__(self, t: Node) -> None:
        self.t = t

    @override
    def render(self, inner: str = "", style: Style = "gcc") -> str:
        return self.t.render(_pre("&&", inner), style=style)


@final
class Wrap(Node):
    """Prefix + type, for example ``_Complex double``."""

    def __init__(self, prefix: str, t: Node) -> None:
        self.prefix = prefix
        self.t = t

    @override
    def render(self, inner: str = "", style: Style = "gcc") -> str:
        return _decl(f"{self.prefix}{self.t.render(style=style)}", inner)


@final
class Suffixed(Node):
    """Type + trailing vendor attribute."""

    def __init__(self, t: Node, suffix: str) -> None:
        self.t = t
        self.suffix = suffix

    @override
    def render(self, inner: str = "", style: Style = "gcc") -> str:
        return _decl(f"{self.t.render(style=style)}{self.suffix}", inner)


@final
class Qual(Node):
    """Qualified type node."""

    def __init__(self, t: Node, quals: Iterable[str]) -> None:
        self.t = t
        self.quals = list(quals)

    @override
    def render(self, inner: str = "", style: Style = "gcc") -> str:
        quals = " ".join(self.quals)
        if isinstance(self.t, (Ptr, LRef, RRef, ArrayT, FuncT, MemberPtr)):
            return self.t.render(
                f"{quals}{(' ' + inner) if inner else ''}", style=style
            )
        return _decl(f"{self.t.render(style=style)} {quals}", inner)


@final
class ArrayT(Node):
    """Array type node."""

    def __init__(self, t: Node, dim: str) -> None:
        self.t = t
        self.dim = dim

    @override
    def render(self, inner: str = "", style: Style = "gcc") -> str:
        if inner and inner[0] in "*&":
            inner = f"({inner})"
        core = f"{inner} " if inner else ""
        return self.t.render(f"{core}[{self.dim}]", style=style)


@final
class VectorT(Node):
    """Vector type node."""

    def __init__(self, t: Node, dim: str) -> None:
        self.t = t
        self.dim = dim

    @override
    def render(self, inner: str = "", style: Style = "gcc") -> str:
        return _decl(f"{self.t.render(style=style)} vector[{self.dim}]", inner)


@final
class FuncT(Node):
    """Function type node."""

    def __init__(
        self,
        ret: Node | None,
        params: list[Node],
        cv: Iterable[str] = (),
        ref: str = "",
        extern_c: bool = False,
        noexcept: str = "",
    ) -> None:
        self.ret = ret
        self.params = params
        self.cv = list(cv)
        self.ref = ref
        self.extern_c = extern_c
        self.noexcept = noexcept

    @override
    def render(self, inner: str = "", style: Style = "gcc") -> str:
        if inner and inner[0] in "*&":
            inner = f"({inner})"
        core = f"{inner}({_join(self.params, style)})"
        for qualifier in self.cv:
            core += f" {qualifier}"
        if self.ref:
            core += f" {self.ref}"
        if self.noexcept:
            core += f" {self.noexcept}"
        return self.ret.render(core, style=style) if self.ret is not None else core


@final
class MemberPtr(Node):
    """Pointer-to-member type node."""

    def __init__(self, cls: Node, member: Node) -> None:
        self.cls = cls
        self.member = member

    @override
    def render(self, inner: str = "", style: Style = "gcc") -> str:
        declarator = f"{self.cls.render(style=style)}::*{inner}"
        if isinstance(self.member, (FuncT, ArrayT)):
            return self.member.render(f"({declarator})", style=style)
        return self.member.render(declarator, style=style)


@final
class ArgPack(Node):
    """Template argument pack node."""

    def __init__(self, items: list[Node]) -> None:
        self.items = items

    @override
    def render(self, inner: str = "", style: Style = "gcc") -> str:
        rendered = _join(self.items, style)
        return _decl(rendered, inner) if inner else rendered

    @override
    def as_pack(self) -> list[Node] | None:
        return self.items


@final
class PackExpansion(Node):
    """Pack expansion node."""

    def __init__(self, t: Node) -> None:
        self.t = t

    @override
    def render(self, inner: str = "", style: Style = "gcc") -> str:
        items = self.t.as_pack()
        if items is not None:
            return _join(items, style)
        return _decl(f"{self.t.render(style=style)}...", inner)


@final
class ParamRef(Node):
    """Template parameter reference."""

    def __init__(self, scope: list[Node] | None, index: int, spelling: str) -> None:
        self.scope = scope
        self.index = index
        self.spelling = spelling
        self._busy = False

    def resolve(self) -> Node | None:
        if self.scope is not None and 0 <= self.index < len(self.scope):
            return self.scope[self.index]
        return None

    @override
    def render(self, inner: str = "", style: Style = "gcc") -> str:
        target = self.resolve()
        if target is None or self._busy:
            return _decl(self.spelling, inner)
        self._busy = True
        try:
            return target.render(style=style)
        finally:
            self._busy = False

    @override
    def as_pack(self) -> list[Node] | None:
        target = self.resolve()
        if target is not None and not self._busy:
            return target.as_pack()
        return None


@final
class Lambda(Node):
    """Lambda closure type node."""

    def __init__(self, params: list[Node], num: int) -> None:
        self.params = params
        self.num = num

    @override
    def render(self, inner: str = "", style: Style = "gcc") -> str:
        return _decl(f"{{lambda({_join(self.params, style)})#{self.num}}}", inner)

    @override
    def base_name(self) -> str:
        return self.render()


@final
class UnnamedT(Node):
    """Unnamed type node."""

    def __init__(self, num: int) -> None:
        self.num = num

    @override
    def render(self, inner: str = "", style: Style = "gcc") -> str:
        return _decl(f"{{unnamed type#{self.num}}}", inner)


@final
class LocalName(Node):
    """Local-name node."""

    def __init__(self, encoding: Node, entity: Node) -> None:
        self.encoding = encoding
        self.entity = entity

    @override
    def render(self, inner: str = "", style: Style = "gcc") -> str:
        return _decl(
            f"{self.encoding.render(style=style)}::{self.entity.render(style=style)}",
            inner,
        )

    @override
    def base_name(self) -> str:
        return self.entity.base_name()


@final
class CtorDtor(Node):
    """Constructor or destructor node."""

    def __init__(self, prefix: Node | None, is_dtor: bool) -> None:
        self.prefix = prefix
        self.is_dtor = is_dtor

    @override
    def render(self, inner: str = "", style: Style = "gcc") -> str:
        base = self.prefix.base_name() if self.prefix is not None else ""
        return _decl(f"{'~' if self.is_dtor else ''}{base}", inner)


@final
class Placeholder(Node):
    """
    Stand-in for a substitution reference in placeholder mode.

    Renders as a short label (for example ``$0``); the actual spelling is deferred to a
    legend appended after the main demangled text. ``base_name`` and ``as_pack`` still
    delegate to the underlying node so constructors, destructors, and pack expansions
    remain correct even when their scope is a placeholder.
    """

    def __init__(self, label: str, target: Node) -> None:
        self.label = label
        self.target = target

    @override
    def render(self, inner: str = "", style: Style = "gcc") -> str:
        return _decl(self.label, inner)

    @override
    def base_name(self) -> str:
        return self.target.base_name()

    @override
    def as_pack(self) -> list[Node] | None:
        return self.target.as_pack()


@final
class FuncNode(Node):
    """Complete encoding: name, parameters, optional return type."""

    def __init__(
        self,
        name: Node,
        params: list[Node],
        ret: Node | None = None,
        cv: Iterable[str] = (),
        ref: str = "",
    ) -> None:
        self.name = name
        self.params = params
        self.ret = ret
        self.cv = list(cv)
        self.ref = ref

    @override
    def render(self, inner: str = "", style: Style = "gcc") -> str:
        rendered = f"{self.name.render(style=style)}({_join(self.params, style)})"
        for qualifier in self.cv:
            rendered += f" {qualifier}"
        if self.ref:
            rendered += f" {self.ref}"
        return (
            self.ret.render(rendered, style=style) if self.ret is not None else rendered
        )


########################################################################################
# Tables
########################################################################################

BUILTIN: dict[str, str] = {
    "v": "void",
    "w": "wchar_t",
    "b": "bool",
    "c": "char",
    "a": "signed char",
    "h": "unsigned char",
    "s": "short",
    "t": "unsigned short",
    "i": "int",
    "j": "unsigned int",
    "l": "long",
    "m": "unsigned long",
    "x": "long long",
    "y": "unsigned long long",
    "n": "__int128",
    "o": "unsigned __int128",
    "f": "float",
    "d": "double",
    "e": "long double",
    "g": "__float128",
    "z": "...",
}

BUILTIN_D: dict[str, str] = {
    "Dd": "decimal64",
    "De": "decimal128",
    "Df": "decimal32",
    "Dh": "half",
    "DF": "__float",
    "Di": "char32_t",
    "Ds": "char16_t",
    "Du": "char8_t",
    "Da": "auto",
    "Dc": "decltype(auto)",
    "Dn": "decltype(nullptr)",
}

STD_SUBS: dict[str, str] = {
    "t": "std",
    "a": "std::allocator",
    "b": "std::basic_string",
    "s": "std::basic_string<char, std::char_traits<char>, std::allocator<char> >",
    "i": "std::basic_istream<char, std::char_traits<char> >",
    "o": "std::basic_ostream<char, std::char_traits<char> >",
    "d": "std::basic_iostream<char, std::char_traits<char> >",
}

OPERATORS: dict[str, str] = {
    "aa": "&&",
    "ad": "&",
    "an": "&",
    "aN": "&=",
    "aS": "=",
    "aw": " co_await",
    "az": " alignof",
    "cl": "()",
    "cm": ",",
    "co": "~",
    "dV": "/=",
    "da": " delete[]",
    "de": "*",
    "dl": " delete",
    "dv": "/",
    "eO": "^=",
    "eo": "^",
    "eq": "==",
    "ge": ">=",
    "gt": ">",
    "ix": "[]",
    "lS": "<<=",
    "le": "<=",
    "ls": "<<",
    "lt": "<",
    "mI": "-=",
    "mL": "*=",
    "mi": "-",
    "ml": "*",
    "mm": "--",
    "na": " new[]",
    "ne": "!=",
    "ng": "-",
    "nt": "!",
    "nw": " new",
    "oR": "|=",
    "oo": "||",
    "or": "|",
    "pL": "+=",
    "pl": "+",
    "pm": "->*",
    "pp": "++",
    "ps": "+",
    "pt": "->",
    "qu": "?",
    "rM": "%=",
    "rS": ">>=",
    "rm": "%",
    "rs": ">>",
    "ss": "<=>",
    "dt": ".",
    "ds": ".*",
    "quz": "?",
}

UNARY: dict[str, str] = {
    "ps": "+",
    "ng": "-",
    "nt": "!",
    "co": "~",
    "de": "*",
    "ad": "&",
}

BINARY: dict[str, str] = {
    "aa": "&&",
    "an": "&",
    "aN": "&=",
    "aS": "=",
    "cm": ",",
    "dV": "/=",
    "dv": "/",
    "eO": "^=",
    "eo": "^",
    "eq": "==",
    "ge": ">=",
    "gt": ">",
    "lS": "<<=",
    "le": "<=",
    "ls": "<<",
    "lt": "<",
    "mI": "-=",
    "mL": "*=",
    "mi": "-",
    "ml": "*",
    "ne": "!=",
    "oR": "|=",
    "oo": "||",
    "or": "|",
    "pL": "+=",
    "pl": "+",
    "pm": "->*",
    "rM": "%=",
    "rS": ">>=",
    "rm": "%",
    "rs": ">>",
    "ss": "<=>",
    "ix": "[]",
}

INT_SUFFIX: dict[str, str] = {
    "int": "",
    "unsigned int": "u",
    "long": "l",
    "unsigned long": "ul",
    "long long": "ll",
    "unsigned long long": "ull",
}

SPECIAL: dict[str, str] = {
    "TV": "vtable for ",
    "TT": "VTT for ",
    "TI": "typeinfo for ",
    "TS": "typeinfo name for ",
    "TH": "thread-local initialization routine for ",
    "TW": "thread-local wrapper routine for ",
    "GV": "guard variable for ",
    "GR": "reference temporary for ",
    "TA": "template parameter object for ",
}


########################################################################################
# The parser
########################################################################################


class Demangler:
    """Parse an Itanium ABI mangled name into a human-readable spelling."""

    def __init__(
        self,
        text: str,
        style: Style = "gcc",
        placeholders: bool = False,
    ) -> None:
        self.s: str = text
        self.i: int = 0
        self.style: Style = style
        self.subs: list[Node] = []
        self.tparams: list[list[Node]] = []
        self.func_cv: list[str] = []
        self.func_ref: str = ""
        self.had_template_args: bool = False
        self.is_ctor_dtor: bool = False
        self.depth: int = 0
        self.use_placeholders: bool = placeholders
        self.placeholder_labels: dict[int, str] = {}
        self.placeholder_order: list[tuple[str, Node]] = []

    def _render(self, node: Node) -> str:
        """
        Render a node in the current style.

        :param node: Node.
        :return: Rendered text.
        """
        return node.render(style=self.style)

    # ---- character level -------------------------------------------------

    def peek(self, offset: int = 0) -> str:
        """
        Peek at the current input character.

        :param offset: Character offset.
        :return: The character, or ``""`` at end of input.
        """
        index = self.i + offset
        return self.s[index] if index < len(self.s) else ""

    def advance(self, count: int = 1) -> None:
        """
        Advance the input cursor.

        :param count: Number of characters to skip.
        """
        self.i += count

    def at_end(self) -> bool:
        """
        Return whether the parser is at the end of input.

        :return: ``True`` when the input is exhausted.
        """
        return self.i >= len(self.s)

    def consume_if(self, token: str) -> bool:
        """
        Consume ``token`` if it appears at the current position.

        :param token: Token to match.
        :return: ``True`` when the token was consumed.
        """
        if self.s.startswith(token, self.i):
            self.i += len(token)
            return True
        return False

    def expect(self, token: str) -> None:
        """
        Consume ``token`` or raise ``DemangleError``.

        :param token: Token to match.
        :raises DemangleError: If the token is absent.
        """
        if not self.consume_if(token):
            self.fail(f"expected {token!r}")

    def fail(self, message: str) -> NoReturn:
        """
        Raise a parse failure with context.

        :param message: Error message.
        :raises DemangleError: Always.
        """
        raise DemangleError(
            f"{message} at offset {self.i} (near {self.s[self.i : self.i + 24]!r})"
        )

    def add_sub(self, node: Node) -> Node:
        """
        Add a substitution-table entry.

        :param node: Node to add.
        :return: The same node.
        """
        self.subs.append(node)
        return node

    # ---- primitives ------------------------------------------------------

    def parse_digits(self) -> str:
        """
        Parse a run of decimal digits.

        :return: The digit string, possibly empty.
        """
        start = self.i
        while self.peek().isdigit():
            self.advance()
        return self.s[start : self.i]

    def parse_number(self) -> str:
        """
        Parse a signed decimal number.

        :return: Parsed number as text.
        """
        neg = self.consume_if("n")
        digits = self.parse_digits()
        if not digits:
            self.fail("expected number")
        return f"{'-' if neg else ''}{digits}"

    def parse_seq_id(self) -> int | None:
        """
        Parse a base-36 substitution identifier.

        :return: Identifier value, or ``None`` if no digits were consumed.
        """
        value = 0
        length = 0
        while True:
            c = self.peek()
            if c.isdigit():
                digit = ord(c) - 48
            elif "A" <= c <= "Z":
                digit = ord(c) - 55
            else:
                break
            value = value * 36 + digit
            length += 1
            self.advance()
        return value if length else None

    def parse_source_name(self) -> str:
        """
        Parse a length-prefixed source name.

        :return: Parsed source name.
        """
        digits = self.parse_digits()
        if not digits:
            self.fail("expected source-name length")
        length = int(digits)
        if self.i + length > len(self.s):
            self.fail("source-name runs past end of string")
        name = self.s[self.i : self.i + length]
        self.advance(length)
        return name

    def parse_cv_qualifiers(self) -> list[str]:
        """
        Parse CVR qualifiers.

        :return: Qualifiers in presentation order.
        """
        qualifiers: list[str] = []
        if self.consume_if("r"):
            qualifiers.append("restrict")
        if self.consume_if("V"):
            qualifiers.append("volatile")
        if self.consume_if("K"):
            qualifiers.append("const")
        qualifiers.reverse()
        return qualifiers

    def parse_ref_qualifier(self) -> str:
        """
        Parse a reference qualifier.

        :return: ``"&"``, ``"&&"``, or ``""``.
        """
        if self.consume_if("R"):
            return "&"
        if self.consume_if("O"):
            return "&&"
        return ""

    def parse_discriminator(self) -> None:
        """
        Parse an ABI discriminator if present.
        """
        if self.peek() != "_":
            return
        save = self.i
        self.advance()
        if self.consume_if("_"):
            if self.parse_digits() and self.consume_if("_"):
                return
            self.i = save
            return
        if self.parse_digits():
            return
        self.i = save

    # ---- names -----------------------------------------------------------

    def parse_name(self, tag: bool = False) -> Node:
        """
        Parse a mangled name.

        :param tag: Whether the result should extend the template scope.
        :return: Parsed name node.
        """
        c = self.peek()
        if c == "N":
            return self.parse_nested_name(tag)
        if c == "Z":
            return self.parse_local_name(tag)
        if c == "S" and self.peek(1) != "t":
            node = self.parse_substitution()
            if self.peek() == "I":
                node = self.add_sub(Template(node, self.parse_template_args(tag)))
                self.had_template_args = True
            else:
                self.had_template_args = False
            return node

        std = self.consume_if("St")
        self.is_ctor_dtor = False
        node = self.parse_unqualified_name(None)
        if std:
            node = Qualified(Lit("std"), node)
        had_args = False
        if self.peek() == "I":
            self.add_sub(node)
            node = Template(node, self.parse_template_args(tag))
            had_args = True
        self.had_template_args = had_args
        return node

    def parse_nested_name(self, tag: bool = False) -> Node:
        """
        Parse a nested name.

        :param tag: Whether the result should extend the template scope.
        :return: Parsed nested name.
        """
        self.expect("N")
        cv = self.parse_cv_qualifiers()
        ref = self.parse_ref_qualifier()
        node: Node | None = None
        registered: Node | None = None
        std_only = False
        had_args = False
        self.is_ctor_dtor = False

        while True:
            if self.at_end():
                self.fail("unterminated nested-name")
            if self.consume_if("E"):
                break
            if node is not None and node is not registered and not std_only:
                self.add_sub(node)
                registered = node
            std_only = False

            c = self.peek()
            if c == "I":
                assert node is not None
                node = Template(node, self.parse_template_args(tag))
                had_args = True
                continue
            if c == "M":
                self.advance()
                continue
            if c == "S" and self.peek(1) == "t":
                self.advance(2)
                comp: Node = Lit("std")
                std_only = node is None
            elif c == "S":
                comp = self.parse_substitution()
            elif c == "T":
                comp = self.add_sub(self.parse_template_param())
            elif c == "D" and self.peek(1) in "tT":
                self.advance(2)
                comp = self.add_sub(
                    Lit(f"decltype({self._render(self.parse_expression())})")
                )
            elif c == "u":
                self.advance()
                name = self.parse_source_name()
                if self.peek() == "I":
                    name += _targs(self.parse_template_args(), self.style)
                comp = Lit(name)
            else:
                comp = self.parse_unqualified_name(node)
                had_args = False

            node = comp if node is None else Qualified(node, comp)

        if node is None:
            self.fail("empty nested-name")
        self.func_cv = cv
        self.func_ref = ref
        self.had_template_args = had_args
        return node

    def parse_local_name(self, tag: bool = False) -> Node:
        """
        Parse a local name.

        :param tag: Whether the result should extend the template scope.
        :return: Parsed local-name node.
        """
        self.expect("Z")
        saved_params = list(self.tparams)
        encoding = self.parse_encoding()
        self.expect("E")
        if self.consume_if("s"):
            entity = Lit("string literal")
            self.parse_discriminator()
            had_args = False
        elif self.consume_if("d"):
            if self.peek().isdigit() or self.peek() == "n":
                self.parse_number()
            self.expect("_")
            entity = self.parse_name(tag)
            had_args = self.had_template_args
        else:
            entity = self.parse_name(tag)
            had_args = self.had_template_args
            self.parse_discriminator()
        self.tparams = saved_params
        self.had_template_args = had_args
        return LocalName(encoding, entity)

    def parse_unqualified_name(self, prefix: Node | None) -> Node:
        """
        Parse an unqualified name.

        :param prefix: Constructor/destructor prefix.
        :return: Parsed name node.
        """
        c = self.peek()
        if c.isdigit():
            node = Lit(self.parse_source_name())
        elif c == "L":
            self.advance()
            node = Lit(self.parse_source_name())
            self.parse_discriminator()
        elif c == "U":
            node = self.parse_unnamed_type_name()
        elif c == "D" and self.peek(1) == "C":
            self.advance(2)
            names: list[str] = []
            while not self.consume_if("E"):
                names.append(self.parse_source_name())
            node = Lit(f"[{', '.join(names)}]")
        elif (c == "C" and self.peek(1) in "12345I") or (
            c == "D" and self.peek(1) in "0125"
        ):
            node = self.parse_ctor_dtor(prefix)
        else:
            node = self.parse_operator_name()
        while self.peek() == "B":
            self.advance()
            node = Lit(f"{self._render(node)}[abi:{self.parse_source_name()}]")
        return node

    def parse_ctor_dtor(self, prefix: Node | None) -> Node:
        """
        Parse a constructor or destructor name.

        :param prefix: Prefix node.
        :return: Constructor or destructor node.
        """
        c = self.peek()
        self.advance()
        if c == "C" and self.consume_if("I"):
            self.advance()
            self.parse_type()
        else:
            self.advance()
        self.is_ctor_dtor = True
        return CtorDtor(prefix, c == "D")

    def parse_operator_name(self) -> Node:
        """
        Parse an operator name.

        :return: Parsed operator node.
        """
        two = self.s[self.i : self.i + 2]
        if two == "cv":
            self.advance(2)
            saved = list(self.tparams)
            t = self.parse_type()
            self.tparams = saved
            self.is_ctor_dtor = True
            return Lit(f"operator {self._render(t)}")
        if two == "li":
            self.advance(2)
            return Lit(f'operator"" {self.parse_source_name()}')
        if two[:1] == "v" and two[1:2].isdigit():
            self.advance(2)
            return Lit(f"operator {self.parse_source_name()}")
        if two in OPERATORS:
            self.advance(2)
            return Lit(f"operator{OPERATORS[two]}")
        self.fail("unknown operator or unqualified-name")

    def parse_unnamed_type_name(self) -> Node:
        """
        Parse an unnamed type name.

        :return: Parsed unnamed type node.
        """
        self.expect("U")
        if self.consume_if("t"):
            digits = self.parse_digits()
            self.expect("_")
            return UnnamedT(int(digits) + 1 if digits else 1)
        if self.consume_if("l"):
            saved: list[list[Node]] | None = None
            if self.peek() == "T" and self.peek(1) in "ytnpk":
                saved = list(self.tparams)
                self.tparams.append([])
                while self.peek() == "T" and self.peek(1) in "ytnpk":
                    self.parse_template_param_decl()
            params: list[Node] = []
            while not self.consume_if("E"):
                if self.at_end():
                    self.fail("unterminated lambda signature")
                params.append(self.parse_type())
            digits = self.parse_digits()
            self.expect("_")
            if saved is not None:
                self.tparams = saved
            if len(params) == 1 and self._render(params[0]) == "void":
                params = []
            return Lambda(params, int(digits) + 1 if digits else 1)
        self.fail("unknown unnamed-type-name")

    def parse_template_param_decl(self) -> None:
        """
        Parse a template parameter declaration.
        """
        self.expect("T")
        kind = self.peek()
        self.advance()
        if kind == "n":
            self.parse_type()
        elif kind == "t":
            while not self.consume_if("E"):
                self.parse_template_param_decl()
        elif kind == "k":
            self.parse_name()
        if self.tparams:
            self.tparams[-1].append(Lit("auto"))

    # ---- substitutions & template parameters ------------------------------

    def parse_substitution(self) -> Node:
        """
        Parse a substitution reference.

        :return: Parsed substitution node.
        """
        self.expect("S")
        c = self.peek()
        if c == "_":
            self.advance()
            idx = 0
        elif c in STD_SUBS:
            self.advance()
            return Lit(STD_SUBS[c])
        else:
            val = self.parse_seq_id()
            if val is None:
                self.fail("bad substitution")
            self.expect("_")
            idx = val + 1
        if idx >= len(self.subs):
            self.fail(f"substitution S{idx} out of range (have {len(self.subs)})")
        node = self.subs[idx]
        if self.use_placeholders:
            return self.placeholder_for(idx, node)
        return node

    def placeholder_for(self, idx: int, node: Node) -> Node:
        """
        Return (creating if needed) the placeholder standing in for substitution ``idx``.

        :param idx: Index into the substitution table.
        :param node: The node that substitution ``idx`` resolves to.
        :return: A :class:`Placeholder` node labelling ``node``.
        """
        label = self.placeholder_labels.get(idx)
        if label is None:
            label = f"${len(self.placeholder_order)}"
            self.placeholder_labels[idx] = label
            self.placeholder_order.append((label, node))
        return Placeholder(label, node)

    def parse_template_param(self) -> Node:
        """
        Parse a template parameter reference.

        :return: Parsed template parameter reference.
        """
        self.expect("T")
        start = self.i - 1
        if self.consume_if("L"):
            self.parse_number()
            self.expect("_")
            digits = self.parse_digits()
            self.expect("_")
            idx = int(digits) + 1 if digits else 0
        elif self.consume_if("_"):
            idx = 0
        else:
            digits = self.parse_digits()
            self.expect("_")
            idx = int(digits) + 1
        scope = self.tparams[-1] if self.tparams else None
        return ParamRef(scope, idx, self.s[start : self.i])

    def parse_template_args(self, tag: bool = False) -> list[Node]:
        """
        Parse template arguments.

        :param tag: Whether to push the result as a template scope.
        :return: Template arguments.
        """
        self.expect("I")
        args: list[Node] = []
        if tag:
            self.tparams.append(args)
        while not self.consume_if("E"):
            if self.at_end():
                self.fail("unterminated template-args")
            args.append(self.parse_template_arg())
        return args

    def parse_template_arg(self) -> Node:
        """
        Parse a single template argument.

        :return: Parsed argument.
        """
        c = self.peek()
        if c == "X":
            self.advance()
            e = self.parse_expression()
            self.expect("E")
            return e
        if c == "J":
            self.advance()
            items: list[Node] = []
            while not self.consume_if("E"):
                if self.at_end():
                    self.fail("unterminated argument pack")
                items.append(self.parse_template_arg())
            return ArgPack(items)
        if c == "L":
            return self.parse_expr_primary()
        return self.parse_type()

    # ---- types -----------------------------------------------------------

    def parse_type(self) -> Node:
        """
        Parse a type.

        :return: Parsed type node.
        """
        self.depth += 1
        if self.depth > 400:
            self.fail("type nesting too deep")
        try:
            return self._parse_type()
        finally:
            self.depth -= 1

    def _parse_type(self) -> Node:
        c = self.peek()
        if c == "":
            self.fail("expected type")

        if c in "rVK":
            quals = self.parse_cv_qualifiers()
            t = self.parse_type()
            if isinstance(t, FuncT):
                t.cv = quals + t.cv
                return t
            return self.add_sub(Qual(t, quals))
        if c == "P":
            self.advance()
            return self.add_sub(Ptr(self.parse_type()))
        if c == "R":
            self.advance()
            return self.add_sub(LRef(self.parse_type()))
        if c == "O":
            self.advance()
            return self.add_sub(RRef(self.parse_type()))
        if c == "C":
            self.advance()
            return self.add_sub(Wrap("_Complex ", self.parse_type()))
        if c == "G":
            self.advance()
            return self.add_sub(Wrap("_Imaginary ", self.parse_type()))
        if c == "M":
            self.advance()
            cls = self.parse_type()
            return self.add_sub(MemberPtr(cls, self.parse_type()))
        if c == "A":
            return self.add_sub(self.parse_array_type())
        if c == "F":
            return self.add_sub(self.parse_function_type())
        if c == "U" and self.peek(1) not in "tl":
            self.advance()
            name = self.parse_source_name()
            if self.peek() == "I":
                name += _targs(self.parse_template_args(), self.style)
            return self.add_sub(Suffixed(self.parse_type(), f" {name}"))
        if c == "T" and self.peek(1) in "sue":
            kw = {"s": "struct ", "u": "union ", "e": "enum "}[self.peek(1)]
            self.advance(2)
            return self.add_sub(Wrap(kw, self.parse_name()))
        if c == "T":
            node = self.add_sub(self.parse_template_param())
            if self.peek() == "I":
                node = self.add_sub(Template(node, self.parse_template_args()))
            return node
        if c == "S" and self.peek(1) != "t":
            node = self.parse_substitution()
            if self.peek() == "I":
                node = self.add_sub(Template(node, self.parse_template_args()))
            return node
        if c == "D":
            node = self.parse_d_type()
            if node is not None:
                return node
        if c == "u":
            self.advance()
            name = self.parse_source_name()
            if self.peek() == "I":
                name += _targs(self.parse_template_args(), self.style)
            return self.add_sub(Lit(name))
        if c in BUILTIN:
            self.advance()
            return Lit(BUILTIN[c])
        return self.add_sub(self.parse_name())

    def parse_d_type(self) -> Node | None:
        """
        Parse a D-prefixed type.

        :return: Parsed node, or ``None`` if the prefix is not a D-type.
        """
        two = self.s[self.i : self.i + 2]
        if two == "Dp":
            self.advance(2)
            return self.add_sub(PackExpansion(self.parse_type()))
        if two in ("Dt", "DT"):
            self.advance(2)
            e = self.parse_expression()
            self.expect("E")
            return self.add_sub(Lit(f"decltype({e})"))
        if two == "Dv":
            self.advance(2)
            if self.peek() == "_":
                self.advance()
                dim = str(self.parse_expression())
                self.expect("_")
            else:
                dim = self.parse_number()
                self.expect("_")
            return self.add_sub(VectorT(self.parse_type(), dim))
        if two in ("DB", "DU"):
            signed = two == "DB"
            self.advance(2)
            if self.peek().isdigit():
                width = self.parse_digits()
            else:
                width = str(self.parse_expression())
            self.expect("_")
            base = f"_BitInt({width})"
            return Lit(base if signed else f"unsigned {base}")
        if two == "DF":
            self.advance(2)
            width = self.parse_digits()
            sat = self.consume_if("x")
            self.expect("_")
            return Lit(f"__float{width}{'_sat' if sat else ''}")
        if two in BUILTIN_D:
            self.advance(2)
            return Lit(BUILTIN_D[two])
        return None

    def parse_array_type(self) -> Node:
        """
        Parse an array type.

        :return: Parsed array type node.
        """
        self.expect("A")
        if self.consume_if("_"):
            dim = ""
        elif self.peek().isdigit():
            dim = self.parse_digits()
            self.expect("_")
        else:
            dim = str(self.parse_expression())
            self.expect("_")
        return ArrayT(self.parse_type(), dim)

    def parse_function_type(self) -> FuncT:
        """
        Parse a function type.

        :return: Parsed function type node.
        """
        cv: list[str] = []
        noexcept = ""
        if self.consume_if("Do"):
            noexcept = "noexcept"
        elif self.consume_if("DO"):
            e = self.parse_expression()
            self.expect("E")
            noexcept = f"noexcept({e})"
        elif self.consume_if("Dw"):
            types: list[Node] = []
            while not self.consume_if("E"):
                types.append(self.parse_type())
            noexcept = f"throw({_join(types, self.style)})"
        self.expect("F")
        extern_c = self.consume_if("Y")
        ret = self.parse_type()
        params: list[Node] = []
        ref = ""
        while True:
            if self.at_end():
                self.fail("unterminated function type")
            if self.consume_if("E"):
                break
            if self.peek() in "RO" and self.peek(1) == "E":
                ref = self.parse_ref_qualifier()
                continue
            params.append(self.parse_type())
        if len(params) == 1 and self._render(params[0]) == "void":
            params = []
        return FuncT(ret, params, cv, ref, extern_c, noexcept)

    # ---- expressions -----------------------------------------------------

    def parse_expression(self) -> Node:
        """
        Parse an expression.

        :return: Parsed expression node.
        """
        self.depth += 1
        if self.depth > 400:
            self.fail("expression nesting too deep")
        try:
            return self._parse_expression()
        finally:
            self.depth -= 1

    def _parse_expression(self) -> Node:
        c = self.peek()
        if c == "L":
            return self.parse_expr_primary()
        if c == "T":
            return self.parse_template_param()
        if c == "f":
            return self.parse_function_param()

        two = self.s[self.i : self.i + 2]

        if two == "gs":
            self.advance(2)
            return Lit(f"::{self._render(self.parse_expression())}")
        if two == "sp":
            self.advance(2)
            return PackExpansion(self.parse_expression())
        if two == "sZ":
            self.advance(2)
            if self.peek() == "T":
                arg = self.parse_template_param()
            else:
                arg = self.parse_function_param()
            return Lit(f"sizeof...({arg})")
        if two == "sP":
            self.advance(2)
            items: list[Node] = []
            while not self.consume_if("E"):
                items.append(self.parse_template_arg())
            return Lit(f"sizeof...({_join(items, self.style)})")
        if two == "sz":
            self.advance(2)
            return Lit(f"sizeof ({self.parse_expression()})")
        if two == "st":
            self.advance(2)
            return Lit(f"sizeof ({self.parse_type()})")
        if two == "az":
            self.advance(2)
            return Lit(f"alignof ({self.parse_expression()})")
        if two == "at":
            self.advance(2)
            return Lit(f"alignof ({self.parse_type()})")
        if two == "tl":
            self.advance(2)
            t = self.parse_type()
            items = []
            while not self.consume_if("E"):
                items.append(self.parse_expression())
            return Lit(f"{t}{{{_join(items, self.style)}}}")
        if two == "il":
            self.advance(2)
            items = []
            while not self.consume_if("E"):
                items.append(self.parse_expression())
            return Lit(f"{{{_join(items, self.style)}}}")
        if two in ("cl", "cp"):
            self.advance(2)
            parts: list[Node] = []
            while not self.consume_if("E"):
                parts.append(self.parse_expression())
            if not parts:
                self.fail("empty call expression")
            return Lit(f"{parts[0]}({_join(parts[1:], self.style)})")
        if two in ("dt", "pt"):
            self.advance(2)
            obj = self.parse_expression()
            member = self.parse_unqualified_name(None)
            if self.peek() == "I":
                member = Template(member, self.parse_template_args())
            return Lit(f"{obj}{'.' if two == 'dt' else '->'}{member}")
        if two in ("ds", "pm"):
            self.advance(2)
            a = self.parse_expression()
            b = self.parse_expression()
            return Lit(f"{a}{'.*' if two == 'ds' else '->*'}{b}")
        if two == "sr":
            self.advance(2)
            t = self.parse_type()
            name = self.parse_unqualified_name(None)
            if self.peek() == "I":
                name = Template(name, self.parse_template_args())
            return Lit(f"{t}::{name}")
        if two == "on":
            self.advance(2)
            return self.parse_operator_name()
        if two == "cv":
            self.advance(2)
            t = self.parse_type()
            if self.consume_if("_"):
                items = []
                while not self.consume_if("E"):
                    items.append(self.parse_expression())
                return Lit(f"{t}({_join(items, self.style)})")
            return Lit(f"({t})({self.parse_expression()})")
        if two in ("sc", "dc", "cc", "rc"):
            kind = {
                "sc": "static_cast",
                "dc": "dynamic_cast",
                "cc": "const_cast",
                "rc": "reinterpret_cast",
            }[two]
            self.advance(2)
            t = self.parse_type()
            return Lit(f"{kind}<{t}>({self.parse_expression()})")
        if two == "ti":
            self.advance(2)
            return Lit(f"typeid ({self.parse_type()})")
        if two == "te":
            self.advance(2)
            return Lit(f"typeid ({self.parse_expression()})")
        if two == "nx":
            self.advance(2)
            return Lit(f"noexcept ({self.parse_expression()})")
        if two == "tw":
            self.advance(2)
            return Lit(f"throw {self.parse_expression()}")
        if two == "tr":
            self.advance(2)
            return Lit("throw")
        if two == "qu":
            self.advance(2)
            a = self.parse_expression()
            b = self.parse_expression()
            c2 = self.parse_expression()
            return Lit(f"({a}) ? ({b}) : ({c2})")
        if two in ("nw", "na"):
            self.advance(2)
            placement: list[Node] = []
            while not self.consume_if("_"):
                placement.append(self.parse_expression())
            t = self.parse_type()
            init = ""
            if self.consume_if("pi"):
                items = []
                while not self.consume_if("E"):
                    items.append(self.parse_expression())
                init = f"({_join(items, self.style)})"
            else:
                self.consume_if("E")
            kw = "new" if two == "nw" else "new[]"
            place = f"({_join(placement, self.style)}) " if placement else " "
            return Lit(f"{kw}{place}{t}{init}")
        if two in ("dl", "da"):
            self.advance(2)
            kw = "delete" if two == "dl" else "delete[]"
            return Lit(f"{kw} {self.parse_expression()}")
        if two in ("pp", "mm") and self.peek(2) == "_":
            self.advance(3)
            op = "++" if two == "pp" else "--"
            return Lit(f"{op}{self.parse_expression()}")
        if two in ("pp", "mm"):
            self.advance(2)
            op = "++" if two == "pp" else "--"
            return Lit(f"{self.parse_expression()}{op}")
        if two in UNARY:
            self.advance(2)
            return Lit(f"{UNARY[two]}({self.parse_expression()})")
        if two in BINARY:
            self.advance(2)
            a = self.parse_expression()
            b = self.parse_expression()
            if two == "ix":
                return Lit(f"{a}[{b}]")
            return Lit(f"({a}) {BINARY[two]} ({b})")
        if c in "NZSUDu123456789" or c.isdigit():
            return self.parse_type()
        self.fail("unsupported expression")

    def parse_function_param(self) -> Node:
        """
        Parse a function parameter reference.

        :return: Parsed function-parameter node.
        """
        self.expect("f")
        if self.consume_if("p"):
            if self.consume_if("T"):
                return Lit("this")
            self.parse_cv_qualifiers()
            digits = self.parse_digits()
            self.expect("_")
            idx = int(digits) + 1 if digits else 0
        elif self.consume_if("L"):
            self.parse_digits()
            self.expect("p")
            self.parse_cv_qualifiers()
            digits = self.parse_digits()
            self.expect("_")
            idx = int(digits) + 1 if digits else 0
        else:
            self.fail("bad function-param")
        return Lit(f"{{parm#{idx + 1}}}")

    def parse_expr_primary(self) -> Node:
        """
        Parse a primary expression.

        :return: Parsed primary-expression node.
        """
        self.expect("L")
        if self.peek() == "_" and self.peek(1) == "Z":
            self.advance(2)
            saved = list(self.tparams)
            node = self.parse_encoding()
            self.tparams = saved
            self.expect("E")
            return node

        t = self.parse_type()
        name = self._render(t)
        if name == "bool":
            value = self.parse_number()
            self.expect("E")
            return Lit("true" if value not in ("0", "-0") else "false")
        if name == "decltype(nullptr)":
            self.parse_digits()
            self.expect("E")
            return Lit("nullptr")

        start = self.i
        while not self.at_end() and self.peek() != "E":
            self.advance()
        raw = self.s[start : self.i]
        self.expect("E")
        return Lit(self.format_literal(name, raw))

    @staticmethod
    def format_literal(type_name: str, raw: str) -> str:
        """
        Format a literal value.

        :param type_name: Literal type name.
        :param raw: Raw literal spelling.
        :return: Formatted literal.
        """
        if type_name in ("float", "double", "long double") and re.fullmatch(
            r"[0-9a-f]+",
            raw,
        ):
            try:
                data = bytes.fromhex(raw if len(raw) % 2 == 0 else f"0{raw}")
                if type_name == "float" and len(data) == 4:
                    return f"{repr(struct.unpack('>f', data)[0])}f"
                if type_name == "double" and len(data) == 8:
                    return repr(struct.unpack(">d", data)[0])
            except (ValueError, struct.error):
                pass
            return f"({type_name})[0x{raw}]"
        value = f"-{raw[1:]}" if raw.startswith("n") else raw
        if type_name in INT_SUFFIX:
            return f"{value}{INT_SUFFIX[type_name]}"
        return f"({type_name}){value}"

    # ---- encodings -------------------------------------------------------

    def parse_encoding(self) -> Node:
        """
        Parse an encoding.

        :return: Parsed encoding node.
        """
        c = self.peek()
        two = self.s[self.i : self.i + 2]
        if two in SPECIAL:
            self.advance(2)
            if two in ("TV", "TT", "TI", "TS"):
                return Lit(f"{SPECIAL[two]}{self.parse_type()}")
            return Lit(f"{SPECIAL[two]}{self.parse_encoding()}")
        if c == "T" and self.peek(1) in "hvc":
            kind = self.peek(1)
            self.advance(2)
            self.parse_call_offset()
            if kind == "c":
                self.parse_call_offset()
            return Lit(f"virtual thunk to {self.parse_encoding()}")

        saved_depth = len(self.tparams)
        name = self.parse_name(tag=True)
        cv, ref = self.func_cv, self.func_ref
        self.func_cv, self.func_ref = [], ""
        had_args, is_cd = self.had_template_args, self.is_ctor_dtor

        if self.at_end() or self.peek() in "E.":
            del self.tparams[saved_depth:]
            return name

        ret = self.parse_type() if (had_args and not is_cd) else None
        params: list[Node] = []
        while not self.at_end() and self.peek() not in "E.":
            params.append(self.parse_type())
        if len(params) == 1 and self._render(params[0]) == "void":
            params = []
        node = FuncNode(name, params, ret, cv, ref)
        del self.tparams[saved_depth:]
        return node

    def parse_call_offset(self) -> None:
        """
        Parse a thunk call offset.
        """
        if self.consume_if("h"):
            self.parse_number()
            self.expect("_")
        elif self.consume_if("v"):
            self.parse_number()
            self.expect("_")
            self.parse_number()
            self.expect("_")

    # ---- entry point -----------------------------------------------------

    def run(self) -> str:
        """
        Demangle the full input string.

        :return: Demangled text.
        """
        self.expect("_Z")
        node = self.parse_encoding()
        if not self.at_end():
            self.fail("trailing characters")
        text = self._render(node)
        if self.placeholder_order:
            legend = ", ".join(
                f"{label} = {self._render(target)}"
                for label, target in self.placeholder_order
            )
            text += f" [{legend}]"
        return text


########################################################################################
# Public API.
########################################################################################

_SUFFIX_RE = re.compile(r"^(_+Z[^.$]*)([.$].*)?$")
_SYMBOL_RE = re.compile(r"_+Z[A-Za-z0-9_$.]+")


def demangle_strict(
    name: str,
    style: Style = "gcc",
    placeholders: bool = False,
) -> str:
    """
    Demangle ``name`` or raise ``DemangleError``.

    :param name: Mangled symbol.
    :param style: Output style, either ``"gcc"`` or ``"llvm"``.
    :param placeholders: When ``True``, substitution references (the components an
        Itanium mangled name backreferences via ``S_``, ``S0_``, ...) are rendered as
        short placeholders (``$0``, ``$1``, ...) instead of being inserted at every use
        site, with each placeholder's spelling given in a legend appended after the main
        text as ``[$0 = ..., $1 = ...]``. This keeps output compact when the same
        (potentially long) type or name is referenced repeatedly.
    :return: Demangled name.
    :raises ValueError: If ``style`` is not recognized.
    :raises DemangleError: If the input is not a valid mangled name.
    """
    m = _SUFFIX_RE.match(name.strip())
    if not m:
        raise DemangleError(f"not an Itanium mangled name: {name!r}")
    core, suffix = m.group(1), m.group(2) or ""
    core = core[core.index("_Z") :]
    result = Demangler(core, style=style, placeholders=placeholders).run()
    if suffix:
        result += f" [clone {suffix}]"
    return result


def demangle(name: str, style: Style = "gcc", placeholders: bool = False) -> str:
    """
    Demangle ``name`` or return it unchanged on failure.

    :param name: Mangled symbol.
    :param style: Output style, either ``"gcc"`` or ``"llvm"``.
    :param placeholders: See :func:`demangle_strict`.
    :return: Demangled name or original input.
    """
    try:
        return demangle_strict(name, style, placeholders)
    except (DemangleError, RecursionError):
        return name


def demangle_text(text: str, style: Style = "gcc", placeholders: bool = False) -> str:
    """
    Replace every mangled name in ``text`` with its demangled form.

    :param text: Input text.
    :param style: Output style, either ``"gcc"`` or ``"llvm"``.
    :param placeholders: See :func:`demangle_strict`.
    :return: Text with demangled symbols.
    """
    return _SYMBOL_RE.sub(
        lambda match: demangle(match.group(0), style, placeholders), text
    )


########################################################################################
# CLI helpers
########################################################################################


def _read_clipboard() -> str | None:
    """
    Best-effort clipboard read.

    :return: Clipboard text, or ``None`` if nothing worked.
    """
    try:
        import pyperclip
    except ImportError:
        pyperclip = None

    if pyperclip is not None:
        try:
            text = pyperclip.paste()
        except pyperclip.PyperclipException:
            text = ""
        if text:
            return text

    if sys.platform == "darwin":
        candidates: list[list[str]] = [["pbpaste"]]
    elif sys.platform.startswith("win"):
        candidates = [["powershell", "-NoProfile", "-Command", "Get-Clipboard"]]
    else:
        candidates = [
            ["xclip", "-selection", "clipboard", "-o"],
            ["xsel", "--clipboard", "--output"],
            ["wl-paste"],
        ]

    for command in candidates:
        if shutil.which(command[0]) is None:
            continue
        try:
            completed_process = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=2,
                check=True,
            )
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            continue
        if completed_process.stdout:
            return completed_process.stdout
    return None


def build_arg_parser() -> argparse.ArgumentParser:
    """
    Build the CLI argument parser.

    :return: Configured parser.
    """
    parser = argparse.ArgumentParser(
        prog="demangle_cpp",
        description="Demangle Itanium C++ ABI mangled names.",
    )
    parser.add_argument(
        "names",
        nargs="*",
        help=(
            "Mangled name(s) to demangle. If omitted, reads from stdin when piped, "
            "otherwise falls back to the system clipboard."
        ),
    )
    parser.add_argument(
        "--style",
        choices=["gcc", "llvm"],
        default="llvm",
        help=(
            "Output style to emulate. 'gcc' matches GNU c++filt/libiberty conventions, "
            "and 'llvm' matches llvm-cxxfilt conventions."
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Treat non-mangled or unparsable input as an error instead of passing it "
            "through unchanged."
        ),
    )
    parser.add_argument(
        "--placeholders",
        action="store_true",
        help=(
            "Render repeated substitution references as short placeholders "
            "($0, $1, ...) instead of inserting them at every use site, with each "
            "placeholder's spelling given in a legend appended after the main text."
        ),
    )
    return parser


def run(argv: Iterable[str] | None = None) -> int:
    """
    Run the command-line interface.

    :param argv: Optional argument vector.
    :return: Process exit code.
    """
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    names = args.names
    if not names:
        if not sys.stdin.isatty():
            names = [line.rstrip("\n") for line in sys.stdin]
        else:
            clipboard_text = _read_clipboard()
            if clipboard_text is None:
                parser.error(
                    "No names given, stdin is a tty, and the clipboard could not be read"
                )
            names = [line for line in clipboard_text.splitlines() if line.strip()]
            if not names:
                names = [clipboard_text]

    status = 0
    for name in names:
        if not name.strip():
            print()
            continue
        try:
            if "_Z" in name:
                print(demangle_strict(name, args.style, args.placeholders))
            elif args.strict:
                raise DemangleError("no '_Z' prefix found")
            else:
                print(name)
        except DemangleError as exc:
            if args.strict:
                print(f"Error: {exc}", file=sys.stderr)
                status = 1
            else:
                print(demangle_text(name, args.style, args.placeholders))
    return status


if __name__ == "__main__":
    raise SystemExit(run())
