"""
Compare ``demangle_strict()`` output against verified system demanglers.

The suite compares GCC-style output against GNU binutils ``c++filt`` and LLVM-style
output against LLVM’s ``cxxfilt``. Candidate binaries are verified by checking
``--version`` output before they are trusted. If no verified oracle is available for a
style, the matching test is skipped.
"""

import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

from curu.demangle_cpp import DemangleError, demangle_strict

SYMBOLS: tuple[str, ...] = (
    "_ZN3foo3barEv",
    "_ZSt4cout",
    "_ZNSt6vectorIiSaIiEE9push_backERKi",
    "_ZN3fooC1Ev",
    "_ZN3fooD2Ev",
    "_ZN1N1fIiEEvT_",
    "_ZZ4mainEN3fooEv",
    "_ZN3foo3barIiEEvT_",
    "_ZNK3foocvbEv",
    "_ZN3Foo3barEPKc",
    "_ZN3FooIiE3barIdEEvT_S1_",
    "_ZNSt3__14coutE",
    "_ZN3foo1fEPiS0_",
    "_ZN1N1CIiEC1Ev",
    "_ZNK1N1CIiE3getEv",
    "_ZN3fooILi1EEEvv",
    "_ZplRK1AS1_",
    "_ZL3fooi",
    "_ZN3std3fooEv",
    "_ZN3fooC2Ev",
    "_ZTV3foo",
    "_ZTI3foo",
    "_Z1fIiEvT_S0_",
)


# --------------------------------------------------------------------------------------
# Binary discovery
# --------------------------------------------------------------------------------------


def _version_text(binary_path: Path) -> str:
    """
    Return lowercased ``--version`` output for ``binary_path``.

    :param binary_path: Path to the candidate executable.
    :return: Lowercased standard output and standard error combined.
    """
    try:
        completed_process = subprocess.run(
            [binary_path, "--version"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""

    return f"{completed_process.stdout}{completed_process.stderr}".lower()


def _is_gnu(binary_path: Path) -> bool:
    """
    Return whether ``binary_path`` appears to be GNU-based.

    :param binary_path: Path to the candidate executable.
    :return: ``True`` when the version text indicates GNU binutils.
    """
    version_text = _version_text(binary_path)
    return "gnu" in version_text and "llvm" not in version_text


def _is_llvm(binary_path: Path) -> bool:
    """
    Return whether ``binary_path`` appears to be LLVM-based.

    :param binary_path: Path to the candidate executable.
    :return: ``True`` when the version text indicates LLVM or Clang.
    """
    version_text = _version_text(binary_path)
    return "llvm" in version_text or "clang" in version_text


def _homebrew_prefixes() -> list[Path]:
    """
    Return likely Homebrew install prefixes.

    Both Apple Silicon and Intel roots are included, plus ``brew --prefix`` if ``brew``
    is available on ``PATH``.

    :return: Candidate Homebrew prefixes in preference order.
    """
    prefixes = [Path("/opt/homebrew"), Path("/usr/local")]

    brew_binary = shutil.which("brew")
    if brew_binary:
        try:
            completed_process = subprocess.run(
                [brew_binary, "--prefix"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return prefixes

        brew_prefix = Path(completed_process.stdout.strip())
        if str(brew_prefix) and brew_prefix not in prefixes:
            prefixes.insert(0, brew_prefix)

    return prefixes


def _is_executable_file(path: Path) -> bool:
    """
    Return whether ``path`` is an executable regular file.

    :param path: Path to inspect.
    :return: ``True`` when ``path`` exists and is executable.
    """
    return path.is_file() and os.access(path, os.X_OK)


def _find_gnu_cxxfilt() -> Path | None:
    """
    Locate a verified GNU ``c++filt``.

    :return: The verified binary path, or ``None`` if no GNU oracle is found.
    """
    candidate_paths: list[Path] = []

    on_path = shutil.which("c++filt")
    if on_path:
        candidate_paths.append(Path(on_path))

    if sys.platform == "darwin":
        for prefix in _homebrew_prefixes():
            candidate_paths.append(prefix / "opt" / "binutils" / "bin" / "c++filt")

        alternate_name = shutil.which("g++filt")
        if alternate_name:
            candidate_paths.append(Path(alternate_name))

    for candidate_path in candidate_paths:
        if _is_executable_file(candidate_path) and _is_gnu(candidate_path):
            return candidate_path

    return None


def _find_llvm_cxxfilt() -> Path | None:
    """
    Locate a verified LLVM ``cxxfilt``.

    :return: The verified binary path, or ``None`` if no LLVM oracle is found.
    """
    candidate_paths: list[Path] = []

    if sys.platform == "darwin":
        on_path = shutil.which("c++filt")
        if on_path:
            candidate_paths.append(Path(on_path))

    dedicated_binary = shutil.which("llvm-cxxfilt")
    if dedicated_binary:
        candidate_paths.append(Path(dedicated_binary))

    if sys.platform != "darwin":
        on_path = shutil.which("c++filt")
        if on_path:
            candidate_paths.append(Path(on_path))

    for candidate_path in candidate_paths:
        if _is_executable_file(candidate_path) and _is_llvm(candidate_path):
            return candidate_path

    return None


GNU_CXXFILT = _find_gnu_cxxfilt()
LLVM_CXXFILT = _find_llvm_cxxfilt()


def _tool_demangle(binary_path: Path, symbol: str) -> str:
    """
    Demangle ``symbol`` with the external tool at ``binary_path``.

    :param binary_path: Path to the demangler executable.
    :param symbol: Mangled symbol to demangle.
    :return: The tool’s demangled output, stripped of surrounding whitespace.
    """
    completed_process = subprocess.run(
        [str(binary_path)],
        input=f"{symbol}\n",
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    return completed_process.stdout.strip()


# --------------------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------------------


class TestAgainstSystemDemanglers(unittest.TestCase):
    """
    Compare ``demangle_strict()`` against verified external demanglers.
    """

    def test_demangle_strict_valid_symbols_are_not_identity(self) -> None:
        """
        Ensure our demangler produces non-identical output for valid symbols.
        """
        for symbol in SYMBOLS:
            with self.subTest(symbol=symbol):
                result = demangle_strict(symbol)
                self.assertTrue(result)
                self.assertNotEqual(result, symbol)

    @unittest.skipUnless(GNU_CXXFILT is not None, "no verified GNU c++filt found")
    def test_matches_gnu_cxxfilt(self) -> None:
        """
        Ensure ``style=\"gcc\"`` matches GNU ``c++filt``.
        """
        assert GNU_CXXFILT is not None

        for symbol in SYMBOLS:
            with self.subTest(symbol=symbol, binary=str(GNU_CXXFILT)):
                expected = _tool_demangle(GNU_CXXFILT, symbol)
                try:
                    actual = demangle_strict(symbol, style="gcc")
                except DemangleError as exc:
                    self.fail(f"demangler raised on {symbol!r}: {exc}")
                self.assertEqual(actual, expected)

    @unittest.skipUnless(LLVM_CXXFILT is not None, "no verified LLVM cxxfilt found")
    def test_matches_llvm_cxxfilt(self) -> None:
        """
        Ensure ``style=\"llvm\"`` matches LLVM ``cxxfilt``.
        """
        assert LLVM_CXXFILT is not None

        for symbol in SYMBOLS:
            with self.subTest(symbol=symbol, binary=str(LLVM_CXXFILT)):
                expected = _tool_demangle(LLVM_CXXFILT, symbol)
                try:
                    actual = demangle_strict(symbol, style="llvm")
                except DemangleError as exc:
                    self.fail(f"demangler raised on {symbol!r}: {exc}")
                self.assertEqual(actual, expected)


class TestPlaceholderMode(unittest.TestCase):
    """
    Exercise ``placeholders=True``, which renders substitution references as short
    labels (``$0``, ``$1``, ...) with their spellings collected in a trailing legend
    instead of being inlined at every use site.
    """

    def test_no_repeated_substitution_has_no_legend(self) -> None:
        """
        A symbol with no reused substitution should render identically either way.
        """
        symbol = "_ZN3foo3barEv"
        self.assertEqual(
            demangle_strict(symbol, placeholders=True),
            demangle_strict(symbol),
        )

    def test_repeated_type_becomes_placeholder(self) -> None:
        """
        A parameter that back-references an earlier type is replaced by a placeholder.
        """
        symbol = "_ZN3foo1fEPiS0_"
        self.assertEqual(demangle_strict(symbol), "foo::f(int*, int*)")
        self.assertEqual(
            demangle_strict(symbol, placeholders=True),
            "foo::f(int*, $0) [$0 = int*]",
        )

    def test_repeated_template_argument_becomes_placeholder(self) -> None:
        """
        A repeated template parameter substitution is also placeholderized.
        """
        symbol = "_Z1fIiEvT_S0_"
        self.assertEqual(demangle_strict(symbol), "void f<int>(int, int)")
        self.assertEqual(
            demangle_strict(symbol, placeholders=True),
            "void f<int>(int, $0) [$0 = int]",
        )

    def test_only_the_backreferenced_occurrence_is_placeholderized(self) -> None:
        """
        The first (defining) occurrence of a type renders in full; only later
        occurrences reached via a substitution back-reference become placeholders.
        """
        symbol = "_ZplRK1AS1_"
        self.assertEqual(
            demangle_strict(symbol), "operator+(A const&, A const&)"
        )
        self.assertEqual(
            demangle_strict(symbol, placeholders=True),
            "operator+(A const&, $0) [$0 = A const&]",
        )

    def test_constructor_name_unaffected_by_placeholder_scope(self) -> None:
        """
        ``base_name`` on a placeholder still resolves to the real underlying name, so
        constructor/destructor spelling is unaffected by placeholder rendering.
        """
        symbol = "_ZN1N1CIiEC1Ev"
        self.assertEqual(
            demangle_strict(symbol, placeholders=True),
            demangle_strict(symbol),
        )


if __name__ == "__main__":
    if GNU_CXXFILT is None:
        print(
            "Note: no verified GNU c++filt found. On macOS, install binutils with "
            + "`brew install binutils` and use `<brew --prefix>/opt/binutils/bin/"
            + "c++filt`.",
            file=sys.stderr,
        )

    if LLVM_CXXFILT is None:
        print(
            "Note: no verified LLVM cxxfilt found. On Linux, install LLVM’s tools "
            + "(for example `apt install llvm`) to get `llvm-cxxfilt`.",
            file=sys.stderr,
        )

    unittest.main()
