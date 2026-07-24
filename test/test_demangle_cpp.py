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
    "_ZNH1S3fooERS_i",
    "_ZNH1S10byconstrefERKS_",
    "_ZNH1S4tmplIRS_EEvOT_",
    "_ZNH1S5byvalES_",
    "_ZNH1S6byrrefEOS_",
    "_Z1fIiEvPT_",
    "_ZN2ns4makeIdJidEEEDaDpOT0_",
    "_ZN2ns4makeIdiEENS_4ExprIT_vJN4thes5AllocIS2_EEEEEv",
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


class TestExplicitObjectParameter(unittest.TestCase):
    """
    Exercise the Itanium ABI extension for explicit object ('deducing this')
    parameters: an ``H`` right after the nested-name's opening ``N`` marks the first
    parameter as the object parameter, which is rendered with a leading ``this``.
    """

    def test_lvalue_reference_object_parameter(self) -> None:
        self.assertEqual(
            demangle_strict("_ZNH1S3fooERS_i"), "S::foo(this S&, int)"
        )

    def test_const_reference_object_parameter(self) -> None:
        self.assertEqual(
            demangle_strict("_ZNH1S10byconstrefERKS_"),
            "S::byconstref(this S const&)",
        )

    def test_by_value_object_parameter(self) -> None:
        self.assertEqual(demangle_strict("_ZNH1S5byvalES_"), "S::byval(this S)")

    def test_rvalue_reference_object_parameter(self) -> None:
        self.assertEqual(
            demangle_strict("_ZNH1S6byrrefEOS_"), "S::byrref(this S&&)"
        )

    def test_templated_object_parameter(self) -> None:
        self.assertEqual(
            demangle_strict("_ZNH1S4tmplIRS_EEvOT_"), "void S::tmpl<S&>(this S&)"
        )


class TestTemplateParamDeclaratorPropagation(unittest.TestCase):
    """
    Regression tests for a bug where a declarator wrapped around a template-parameter
    reference (``T_``, ``T0_``, ...) was silently dropped: ``ParamRef.render`` ignored
    its ``inner`` argument instead of forwarding it to the resolved target, so e.g.
    ``PT_`` (pointer to ``T_``) rendered as the bare parameter type with the ``*``
    missing entirely.
    """

    def test_pointer_to_template_param(self) -> None:
        """``PT_`` must keep its ``*``, not drop it like the resolved type's own."""
        self.assertEqual(demangle_strict("_Z1fIiEvPT_"), "void f<int>(int*)")

    def test_pack_expansion_of_rvalue_reference_to_pack(self) -> None:
        """
        ``Dp OT0_`` (pack expansion of ``Args&&...``) must apply the ``&&`` to each
        pack element individually, not to the comma-joined text as a whole.
        """
        self.assertEqual(
            demangle_strict("_ZN2ns4makeIdJidEEEDaDpOT0_"),
            "auto ns::make<double, int, double>(int&&, double&&)",
        )

    def test_reference_collapsing_on_forwarding_reference(self) -> None:
        """
        ``OT_`` where ``T_`` has been deduced to an lvalue reference (``S&``) must
        collapse to ``S&``, not naively concatenate into ``S&&&``.
        """
        self.assertEqual(
            demangle_strict("_ZNH1S4tmplIRS_EEvOT_"), "void S::tmpl<S&>(this S&)"
        )


class TestSubstitutionTableNumbering(unittest.TestCase):
    """
    Regression test for a bug where a bare substitution reference (``S_``, ``S0_``,
    ...) used as the *first* component of a nested-name was incorrectly re-added to
    the substitution table as if it were newly parsed content. Per the Itanium ABI, a
    production that resolves via the ``<substitution>`` alternative never itself gets
    a new table entry -- only a *combination* built on top of it does. The erroneous
    re-add shifted every later ``Sn_`` index by one, corrupting unrelated back
    references deep in the symbol (observed as a nonsensical type, e.g. a class
    template's own name, appearing where an unrelated template argument belonged).
    """

    def test_bare_substitution_reused_as_nested_name_prefix(self) -> None:
        symbol = "_ZN2ns4makeIdiEENS_4ExprIT_vJN4thes5AllocIS2_EEEEEv"
        self.assertEqual(
            demangle_strict(symbol, style="llvm"),
            "ns::Expr<double, void, thes::Alloc<double>> ns::make<double, int>()",
        )

    def test_generic_lambda_closure_type_referenced_via_local_name(self) -> None:
        """
        A class template argument that names a generic lambda's closure type via
        ``Z<encoding>E<unnamed-type-name>`` local-name syntax, where the lambda's own
        (generic, deduced) parameter type is itself expressed as a substitution
        back-reference into the *outer* scope (``Ul S..._ E_``). This is the exact
        construct from https://github.com/KurtBoehm/lineal's
        ``lineal::scale``/``OpExpr`` machinery (component-wise/vector-expression.hpp)
        that originally produced garbled substitution output. Verified against this
        machine's GCC 16 (``g++-16``), whose mangling of the identical construct is
        independently confirmed correct by both a fresh LLVM ``llvm-cxxfilt`` and a
        fresh GNU ``c++filt``.
        """
        symbol = (
            "_ZN6HolderIN6lineal6detail6OpExprIfZNS0_5scaleIfRNS0_11Dense"
            "VectorEEEDaOT0_T_EUlS8_E_JS5_EEEiE14post_executionIRiiEEvS8_S6_"
        )
        self.assertEqual(
            demangle_strict(symbol, style="llvm"),
            "void Holder<lineal::detail::OpExpr<float, auto lineal::scale<float, "
            "lineal::DenseVector&>(lineal::DenseVector&, float)::{lambda(float)#1}, "
            "lineal::DenseVector&>, int>::post_execution<int&, int>"
            "(float, lineal::DenseVector&)",
        )

    def test_recursive_opexpr_argument_via_explicit_object_member_function(self) -> None:
        """
        A member function taking an explicit object parameter (``this auto&&``, the
        ``H`` marker), on a class whose own template argument is itself another
        instantiation of the *same* class template (``OpExpr<Real, Op, Vecs...>``
        nested inside ``OpExpr<Real, Op, Vecs...>``), reached through a pack
        containing both a plain substitution back-reference and the nested
        instantiation. Verified byte-for-byte against a fresh LLVM ``llvm-cxxfilt``
        and fresh GNU ``c++filt`` on this machine (mangled by Apple Clang 21).
        """
        symbol = (
            "_ZNH6lineal10AssignCwOpILb0ERNS_11DenseVectorENS_6OpExprIdiJS2_"
            "NS3_IfiJS2_EEEEEEE14post_executionIRS6_NS_11OwnIndexTagEEEvOT_T0_"
        )
        self.assertEqual(
            demangle_strict(symbol, style="llvm"),
            "void lineal::AssignCwOp<false, lineal::DenseVector&, "
            "lineal::OpExpr<double, int, lineal::DenseVector&, "
            "lineal::OpExpr<float, int, lineal::DenseVector&>>>::post_execution<"
            "lineal::AssignCwOp<false, lineal::DenseVector&, "
            "lineal::OpExpr<double, int, lineal::DenseVector&, "
            "lineal::OpExpr<float, int, lineal::DenseVector&>>>&, "
            "lineal::OwnIndexTag>(this lineal::AssignCwOp<false, lineal::DenseVector&, "
            "lineal::OpExpr<double, int, lineal::DenseVector&, "
            "lineal::OpExpr<float, int, lineal::DenseVector&>>>&, lineal::OwnIndexTag)",
        )

    def test_constrained_template_argument_renders_as_the_underlying_argument(self) -> None:
        """
        ``Tk<concept-name><template-arg>`` (a "constrained template argument", the
        newer Itanium ABI extension https://github.com/itanium-cxx-abi/cxx-abi/issues/24
        for concept-constrained template/``auto`` parameters, e.g. ``NonGlobalIndexTag
        auto tag``) must render as just the underlying argument, discarding the
        concept name -- matching LLVM's own demangler, whose ``TemplateParamQualifiedArg``
        node is documented "don't print Param [the concept], ... print Arg". Verified
        against a fresh ``llvm-cxxfilt`` (this construct needs upstream/Homebrew LLVM
        clang 22 to even emit ``Tk`` -- Apple Clang 21 does not).
        """
        symbol = (
            "_ZNH6lineal10AssignCwOpILb0ERNS_11DenseVectorENS_6OpExprIdiJS2_"
            "NS3_IfiJS2_EEEEEEE14post_executionIRS6_TkNS_17NonGlobalIndexTagE"
            "NS_11OwnIndexTagEEEvOT_T0_"
        )
        result = demangle_strict(symbol, style="llvm")
        self.assertNotIn("NonGlobalIndexTag auto", result)
        self.assertIn(
            "post_execution<lineal::AssignCwOp<false, lineal::DenseVector&, "
            "lineal::OpExpr<double, int, lineal::DenseVector&, "
            "lineal::OpExpr<float, int, lineal::DenseVector&>>>&, "
            "lineal::OwnIndexTag>",
            result,
        )


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

    def test_min_placeholder_length_inlines_short_substitutions(self) -> None:
        """
        A substitution shorter than ``min_placeholder_length`` stays inlined even
        though it is reused, since a placeholder would not save any space.
        """
        symbol = "_ZN3foo1fEPiS0_"
        rendered_length = len("int*")
        self.assertEqual(
            demangle_strict(
                symbol, placeholders=True, min_placeholder_length=rendered_length
            ),
            "foo::f(int*, $0) [$0 = int*]",
        )
        self.assertEqual(
            demangle_strict(
                symbol, placeholders=True, min_placeholder_length=rendered_length + 1
            ),
            "foo::f(int*, int*)",
        )

    def test_min_placeholder_length_still_placeholderizes_long_substitutions(
        self,
    ) -> None:
        """
        A substitution at or above the threshold is placeholderized as usual, even
        with a threshold that would inline a shorter one.
        """
        symbol = "_ZNSt6vectorIiSaIiEE9push_backES1_"
        threshold = len("int*") + 1
        self.assertEqual(
            demangle_strict(symbol, placeholders=True, min_placeholder_length=threshold),
            "std::vector<int, std::allocator<int> >::push_back($0) "
            "[$0 = std::vector<int, std::allocator<int> >]",
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
