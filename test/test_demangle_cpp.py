"""
Compare ``demangle_strict()`` output against verified system demanglers.

The suite compares GCC-style output against GNU binutils ``c++filt`` and LLVM-style
output against LLVM's ``cxxfilt``. Candidate binaries are verified by checking
``--version`` output before they are trusted. If no verified oracle is available for a
style, the matching tests are skipped.
"""

from __future__ import annotations

import io
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from curu.demangle_cpp import (
    DemangleError,
    build_arg_parser,
    demangle,
    demangle_strict,
    demangle_text,
    run,
)

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
    # Free functions with references, function pointers, arrays, and overloads.
    "_Z10ptr_to_ptrPPi",
    "_Z10ref_paramsRiOiRKi",
    "_Z9bool_tmplILb1EEvv",
    "_Z8overloadd",
    "_Z8overloadf",
    "_ZN5outer5inner12func_ptr_varE",
    "_ZN5outer5inner12global_arrayE",
    # Nested namespaces, class templates, non-type template arguments, ctors/dtors.
    "_ZN5outer5inner3BoxIiLi4EE4fillEi",
    "_ZN5outer5inner4BaseC2Ev",
    "_ZN5outer5inner4BaseD2Ev",
    "_ZN5outer5inner6Widget5resetEv",
    "_ZN5outer5inner6WidgetC1Ev",
    "_ZN5outer5inner6WidgetD1Ev",
    "_ZN5outer5inner7Derived2goEv",
    "_ZN5outer5inner7pack_fnIJidPKcEEEvDpT_",
    "_ZN9anon_test10use_hiddenEv",
    "_ZN9anon_test12_GLOBAL__N_16Hidden3runEv",
    "_ZNK5outer5inner3BoxIiLi4EE7convertIdEET_v",
    "_ZNK5outer5inner6Widget5valueEi",
    "_ZNK5outer5inner6WidgetcvbEv",
    "_ZNK5outer5inner6WidgeteqERKS1_",
    "_ZNK5outer5inner6WidgetplERKS1_",
    # libc++'s inline namespace and an ABI tag.
    "_ZNKSt3__16vectorIiNS_9allocatorIiEEE4dataB9nqe210106Ev",
    # Vtables, operator delete, standard library free function.
    "_ZTVN5outer5inner4BaseE",
    "_ZTVN5outer5inner7DerivedE",
    "_ZdlPv",
    "_ZdlPvSt11align_val_t",
    "_ZSt9terminatev",
    # Guard variables and thread-local wrappers/initializers.
    "_ZGVZ7guardedvE1g",
    "_ZTW7tls_var",
    "_ZTH7tls_var",
    # Non-virtual, virtual, and covariant-return thunks.
    "_ZThn8_N1CD1Ev",
    "_ZTv0_n24_N1CD1Ev",
    "_ZTcv0_n24_v0_n32_N1CD1Ev",
    # A lambda closure named via Apple's legacy "$_N" scheme (containing a literal
    # '$', which must not be mistaken for the start of a clone suffix).
    "_ZZ11lambda_uservENK3$_0clEi",
    # std::complex, pointer-to-data-member, and char8_t/char16_t/char32_t.
    "_Z10complex_fnNSt3__17complexIdEE",
    "_Z16takes_member_ptrM1Si",
    "_Z7char_fnDuDsDi",
)

# Symbols using Clang's _BitInt(N) extension. GNU binutils' c++filt (as of 2.46) does
# not support these and returns them unchanged, so they are compared against LLVM's
# cxxfilt only rather than folded into SYMBOLS.
LLVM_ONLY_SYMBOLS: tuple[str, ...] = (
    "_Z9bitint_fnDB17_",
    "_Z18unsigned_bitint_fnDU9_",
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

    if sys.platform == "darwin":
        for prefix in _homebrew_prefixes():
            candidate_paths.append(prefix / "opt" / "llvm" / "bin" / "llvm-cxxfilt")

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
    :return: The tool's demangled output, stripped of surrounding whitespace.
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
# Tests against real demanglers
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("symbol", SYMBOLS)
def test_demangle_strict_valid_symbols_are_not_identity(symbol: str) -> None:
    """Ensure our demangler produces non-identical output for valid symbols."""
    result = demangle_strict(symbol)
    assert result
    assert result != symbol


@pytest.mark.skipif(GNU_CXXFILT is None, reason="no verified GNU c++filt found")
@pytest.mark.parametrize("symbol", SYMBOLS)
def test_matches_gnu_cxxfilt(symbol: str) -> None:
    """Ensure ``style="gcc"`` matches GNU ``c++filt``."""
    assert GNU_CXXFILT is not None

    expected = _tool_demangle(GNU_CXXFILT, symbol)
    try:
        actual = demangle_strict(symbol, style="gcc")
    except DemangleError as exc:
        pytest.fail(f"demangler raised on {symbol!r}: {exc}")
    assert actual == expected


@pytest.mark.skipif(LLVM_CXXFILT is None, reason="no verified LLVM cxxfilt found")
@pytest.mark.parametrize("symbol", SYMBOLS)
def test_matches_llvm_cxxfilt(symbol: str) -> None:
    """Ensure ``style="llvm"`` matches LLVM ``cxxfilt``."""
    assert LLVM_CXXFILT is not None

    expected = _tool_demangle(LLVM_CXXFILT, symbol)
    try:
        actual = demangle_strict(symbol, style="llvm")
    except DemangleError as exc:
        pytest.fail(f"demangler raised on {symbol!r}: {exc}")
    assert actual == expected


@pytest.mark.skipif(LLVM_CXXFILT is None, reason="no verified LLVM cxxfilt found")
@pytest.mark.parametrize("symbol", LLVM_ONLY_SYMBOLS)
def test_matches_llvm_cxxfilt_bitint(symbol: str) -> None:
    """
    ``_BitInt(N)`` symbols, verified against LLVM's ``cxxfilt`` only.

    GNU binutils' ``c++filt`` (2.46) does not support this Clang extension and returns
    such symbols unchanged, so there is no GNU oracle to compare against here.
    """
    assert LLVM_CXXFILT is not None

    expected = _tool_demangle(LLVM_CXXFILT, symbol)
    try:
        actual = demangle_strict(symbol, style="llvm")
    except DemangleError as exc:
        pytest.fail(f"demangler raised on {symbol!r}: {exc}")
    assert actual == expected


# --------------------------------------------------------------------------------------
# Explicit object ('deducing this') parameters
# --------------------------------------------------------------------------------------


class TestExplicitObjectParameter:
    """
    Exercise the Itanium ABI extension for explicit object ('deducing this')
    parameters: an ``H`` right after the nested-name's opening ``N`` marks the first
    parameter as the object parameter, which is rendered with a leading ``this``.
    """

    def test_lvalue_reference_object_parameter(self) -> None:
        assert demangle_strict("_ZNH1S3fooERS_i") == "S::foo(this S&, int)"

    def test_const_reference_object_parameter(self) -> None:
        assert (
            demangle_strict("_ZNH1S10byconstrefERKS_")
            == "S::byconstref(this S const&)"
        )

    def test_by_value_object_parameter(self) -> None:
        assert demangle_strict("_ZNH1S5byvalES_") == "S::byval(this S)"

    def test_rvalue_reference_object_parameter(self) -> None:
        assert demangle_strict("_ZNH1S6byrrefEOS_") == "S::byrref(this S&&)"

    def test_templated_object_parameter(self) -> None:
        assert (
            demangle_strict("_ZNH1S4tmplIRS_EEvOT_") == "void S::tmpl<S&>(this S&)"
        )

    def test_h_marker_no_longer_rejects_deeply_nested_real_world_symbol(self) -> None:
        """
        Regression test: this symbol used to raise ``DemangleError`` (``unknown
        operator or unqualified-name``) at offset 3 because the parser did not
        recognize the explicit-object-parameter ``H`` marker at all, and rejected the
        whole name outright before getting anywhere near its substitution table.

        The ``H`` marker is fixed and this no longer happens. The symbol still fails
        further in -- via a distinct, unresolved substitution-numbering discrepancy
        around a lambda closure type mangled through local-name (``Z...E``) syntax --
        but that is a separate, narrower bug tracked apart from the ``H`` marker fix,
        not the "unknown operator" failure this test guards against. Assert the
        specific error changed rather than that demangling now fully succeeds.
        """
        symbol = (
            "_ZNH6linalg10AssignCwOpILb0ERNS_11DenseVectorIdjRKNS_25Contiguous"
            "DistributedInfoINS_4test15DefaultDistDefsILm0EE12BaseDistDefsERKNS_"
            "3mpi17IntraCommunicatorEEELm2EN4thes18HugePagesAllocatorIdEEEENS_"
            "6OpExprIdNSt3__15minusIvEEJSI_NSJ_IdZNS_5scaleIdRNS1_IfjSD_Lm4ENSF_"
            "IfEEEEEEDaOT0_T_EUlSU_E_JSQ_EEEEEEE14post_executionIRSY_NS_11OwnIndex"
            "TagEEEvOSU_SS_"
        )
        with pytest.raises(DemangleError) as exc_info:
            demangle_strict(symbol)
        message = str(exc_info.value)
        assert "unknown operator or unqualified-name" not in message
        assert "substitution" in message


# --------------------------------------------------------------------------------------
# Template-parameter declarator propagation
# --------------------------------------------------------------------------------------


class TestTemplateParamDeclaratorPropagation:
    """
    Regression tests for a bug where a declarator wrapped around a template-parameter
    reference (``T_``, ``T0_``, ...) was silently dropped: ``ParamRef.render`` ignored
    its ``inner`` argument instead of forwarding it to the resolved target, so e.g.
    ``PT_`` (pointer to ``T_``) rendered as the bare parameter type with the ``*``
    missing entirely.
    """

    def test_pointer_to_template_param(self) -> None:
        """``PT_`` must keep its ``*``, not drop it like the resolved type's own."""
        assert demangle_strict("_Z1fIiEvPT_") == "void f<int>(int*)"

    def test_pack_expansion_of_rvalue_reference_to_pack(self) -> None:
        """
        ``Dp OT0_`` (pack expansion of ``Args&&...``) must apply the ``&&`` to each
        pack element individually, not to the comma-joined text as a whole.
        """
        assert demangle_strict("_ZN2ns4makeIdJidEEEDaDpOT0_") == (
            "auto ns::make<double, int, double>(int&&, double&&)"
        )

    def test_reference_collapsing_on_forwarding_reference(self) -> None:
        """
        ``OT_`` where ``T_`` has been deduced to an lvalue reference (``S&``) must
        collapse to ``S&``, not naively concatenate into ``S&&&``.
        """
        assert (
            demangle_strict("_ZNH1S4tmplIRS_EEvOT_") == "void S::tmpl<S&>(this S&)"
        )


# --------------------------------------------------------------------------------------
# Substitution table numbering
# --------------------------------------------------------------------------------------


class TestSubstitutionTableNumbering:
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
        assert demangle_strict(symbol, style="llvm") == (
            "ns::Expr<double, void, thes::Alloc<double>> ns::make<double, int>()"
        )

    def test_generic_lambda_closure_type_referenced_via_local_name(self) -> None:
        """
        A class template argument that names a generic lambda's closure type via
        ``Z<encoding>E<unnamed-type-name>`` local-name syntax, where the lambda's own
        (generic, deduced) parameter type is itself expressed as a substitution
        back-reference into the *outer* scope (``Ul S..._ E_``). This is a
        component-wise vector-expression construct from a linear-algebra library that
        originally produced garbled substitution output. Verified against this
        machine's Homebrew GCC 16 (``g++-16``), whose mangling of the identical
        construct is independently confirmed correct by both a fresh LLVM
        ``llvm-cxxfilt`` and a fresh GNU ``c++filt``.
        """
        symbol = (
            "_ZN6HolderIN6linalg6detail6OpExprIfZNS0_5scaleIfRNS0_11Dense"
            "VectorEEEDaOT0_T_EUlS8_E_JS5_EEEiE14post_executionIRiiEEvS8_S6_"
        )
        assert demangle_strict(symbol, style="llvm") == (
            "void Holder<linalg::detail::OpExpr<float, auto linalg::scale<float, "
            "linalg::DenseVector&>(linalg::DenseVector&, float)::{lambda(float)#1}, "
            "linalg::DenseVector&>, int>::post_execution<int&, int>"
            "(float, linalg::DenseVector&)"
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
            "_ZNH6linalg10AssignCwOpILb0ERNS_11DenseVectorENS_6OpExprIdiJS2_"
            "NS3_IfiJS2_EEEEEEE14post_executionIRS6_NS_11OwnIndexTagEEEvOT_T0_"
        )
        assert demangle_strict(symbol, style="llvm") == (
            "void linalg::AssignCwOp<false, linalg::DenseVector&, "
            "linalg::OpExpr<double, int, linalg::DenseVector&, "
            "linalg::OpExpr<float, int, linalg::DenseVector&>>>::post_execution<"
            "linalg::AssignCwOp<false, linalg::DenseVector&, "
            "linalg::OpExpr<double, int, linalg::DenseVector&, "
            "linalg::OpExpr<float, int, linalg::DenseVector&>>>&, "
            "linalg::OwnIndexTag>(this linalg::AssignCwOp<false, linalg::DenseVector&, "
            "linalg::OpExpr<double, int, linalg::DenseVector&, "
            "linalg::OpExpr<float, int, linalg::DenseVector&>>>&, linalg::OwnIndexTag)"
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
            "_ZNH6linalg10AssignCwOpILb0ERNS_11DenseVectorENS_6OpExprIdiJS2_"
            "NS3_IfiJS2_EEEEEEE14post_executionIRS6_TkNS_17NonGlobalIndexTagE"
            "NS_11OwnIndexTagEEEvOT_T0_"
        )
        result = demangle_strict(symbol, style="llvm")
        assert "NonGlobalIndexTag auto" not in result
        assert (
            "post_execution<linalg::AssignCwOp<false, linalg::DenseVector&, "
            "linalg::OpExpr<double, int, linalg::DenseVector&, "
            "linalg::OpExpr<float, int, linalg::DenseVector&>>>&, "
            "linalg::OwnIndexTag>"
        ) in result


# --------------------------------------------------------------------------------------
# Apple Clang 21 out-of-range substitution workaround
# --------------------------------------------------------------------------------------


class TestAppleClang21Workarounds:
    """
    Exercise ``apple_clang_21_workarounds=True``, which works around a real mangling
    bug in Apple Clang 21: certain lambda closures nested inside recursive
    class-template arguments are mangled with a substitution index one past the end of
    a spec-compliant substitution table (confirmed against both a fresh LLVM
    ``llvm-cxxfilt`` and a fresh GNU ``c++filt`` -- neither can demangle these symbols
    either, and upstream/mainline LLVM Clang 22 does not reproduce the bug for the
    equivalent construct). Rather than raise ``DemangleError`` on the out-of-range
    reference, the workaround falls back to the most recently registered substitution.
    This cannot be verified byte-for-byte against an oracle (none can parse the input),
    so these tests only assert the crash is gone and the output is plausible.
    """

    def test_disabled_by_default(self) -> None:
        """The out-of-range error still raises unless explicitly opted in."""
        symbol = (
            "_ZN6HolderIRN6linalg11DenseVectorENS0_6detail6OpExprIfZNS0_"
            "5scaleIfS2_EEDaOT0_T_EUlS9_E_JS2_EEEE14post_executionIiiEEvS9_S7_"
        )
        with pytest.raises(DemangleError) as exc_info:
            demangle_strict(symbol)
        assert "out of range" in str(exc_info.value)

    def test_turns_out_of_range_crash_into_best_effort_result(self) -> None:
        symbol = (
            "_ZN6HolderIRN6linalg11DenseVectorENS0_6detail6OpExprIfZNS0_"
            "5scaleIfS2_EEDaOT0_T_EUlS9_E_JS2_EEEE14post_executionIiiEEvS9_S7_"
        )
        result = demangle_strict(symbol, style="llvm", apple_clang_21_workarounds=True)
        assert "Holder<linalg::DenseVector&" in result
        assert "linalg::detail::OpExpr<float," in result
        assert "{lambda(float)#1}" in result

    def test_does_not_change_output_for_correctly_mangled_names(self) -> None:
        """The workaround only ever activates on an out-of-range reference."""
        for symbol in (
            "_ZN3foo3barEv",
            "_ZNSt6vectorIiSaIiEE9push_backERKi",
            "_ZN3foo1fEPiS0_",
        ):
            assert demangle_strict(symbol) == demangle_strict(
                symbol, apple_clang_21_workarounds=True
            )

    def test_real_world_symbol_no_longer_crashes(self) -> None:
        """
        A symbol representative of the construct that motivated this workaround.
        Demangling still cannot be verified against an oracle (both fresh LLVM and
        GNU demanglers fail on it too), so this only checks the workaround turns the
        crash into a complete, plausible result.
        """
        symbol = (
            "_ZNH6linalg10AssignCwOpILb0ERNS_11DenseVectorIdjRKNS_25Contiguous"
            "DistributedInfoINS_4test15DefaultDistDefsILm0EE12BaseDistDefsERKNS_"
            "3mpi17IntraCommunicatorEEELm2EN4thes18HugePagesAllocatorIdEEEENS_"
            "6OpExprIdNSt3__15minusIvEEJSI_NSJ_IdZNS_5scaleIdRNS1_IfjSD_Lm4ENSF_"
            "IfEEEEEEDaOT0_T_EUlSU_E_JSQ_EEEEEEE14post_executionIRSY_NS_11OwnIndex"
            "TagEEEvOSU_SS_"
        )
        with pytest.raises(DemangleError):
            demangle_strict(symbol)

        result = demangle_strict(symbol, apple_clang_21_workarounds=True)
        assert "linalg::AssignCwOp<false," in result
        assert "post_execution<" in result
        assert "linalg::OwnIndexTag>" in result


# --------------------------------------------------------------------------------------
# Placeholder mode
# --------------------------------------------------------------------------------------


class TestPlaceholderMode:
    """
    Exercise ``placeholders=True``, which renders substitution references as short
    labels (``$0``, ``$1``, ...) with their spellings collected in a trailing legend
    instead of being inlined at every use site.
    """

    def test_no_repeated_substitution_has_no_legend(self) -> None:
        """A symbol with no reused substitution should render identically either way."""
        symbol = "_ZN3foo3barEv"
        assert demangle_strict(symbol, placeholders=True) == demangle_strict(symbol)

    def test_repeated_type_becomes_placeholder(self) -> None:
        """A parameter that back-references an earlier type is replaced by a placeholder."""
        symbol = "_ZN3foo1fEPiS0_"
        assert demangle_strict(symbol) == "foo::f(int*, int*)"
        assert demangle_strict(symbol, placeholders=True) == (
            "foo::f(int*, $0) [$0 = int*]"
        )

    def test_repeated_template_argument_becomes_placeholder(self) -> None:
        """A repeated template parameter substitution is also placeholderized."""
        symbol = "_Z1fIiEvT_S0_"
        assert demangle_strict(symbol) == "void f<int>(int, int)"
        assert demangle_strict(symbol, placeholders=True) == (
            "void f<int>(int, $0) [$0 = int]"
        )

    def test_only_the_backreferenced_occurrence_is_placeholderized(self) -> None:
        """
        The first (defining) occurrence of a type renders in full; only later
        occurrences reached via a substitution back-reference become placeholders.
        """
        symbol = "_ZplRK1AS1_"
        assert demangle_strict(symbol) == "operator+(A const&, A const&)"
        assert demangle_strict(symbol, placeholders=True) == (
            "operator+(A const&, $0) [$0 = A const&]"
        )

    def test_constructor_name_unaffected_by_placeholder_scope(self) -> None:
        """
        ``base_name`` on a placeholder still resolves to the real underlying name, so
        constructor/destructor spelling is unaffected by placeholder rendering.
        """
        symbol = "_ZN1N1CIiEC1Ev"
        assert demangle_strict(symbol, placeholders=True) == demangle_strict(symbol)

    def test_multiple_distinct_placeholders_get_separate_labels(self) -> None:
        """Two different reused substitutions get two independently numbered labels."""
        symbol = "_ZN3foo1fEPiS0_RK1AS2_"
        assert demangle_strict(symbol) == "foo::f(int*, int*, A const&, A const)"
        result = demangle_strict(symbol, placeholders=True)
        assert result == (
            "foo::f(int*, $0, A const&, $1) [$0 = int*, $1 = A const]"
        )

    def test_min_placeholder_length_inlines_short_substitutions(self) -> None:
        """
        A substitution shorter than ``min_placeholder_length`` stays inlined even
        though it is reused, since a placeholder would not save any space.
        """
        symbol = "_ZN3foo1fEPiS0_"
        rendered_length = len("int*")
        assert demangle_strict(
            symbol, placeholders=True, min_placeholder_length=rendered_length
        ) == "foo::f(int*, $0) [$0 = int*]"
        assert demangle_strict(
            symbol, placeholders=True, min_placeholder_length=rendered_length + 1
        ) == "foo::f(int*, int*)"

    def test_min_placeholder_length_still_placeholderizes_long_substitutions(
        self,
    ) -> None:
        """
        A substitution at or above the threshold is placeholderized as usual, even
        with a threshold that would inline a shorter one.
        """
        symbol = "_ZNSt6vectorIiSaIiEE9push_backES1_"
        threshold = len("int*") + 1
        assert demangle_strict(
            symbol, placeholders=True, min_placeholder_length=threshold
        ) == (
            "std::vector<int, std::allocator<int> >::push_back($0) "
            "[$0 = std::vector<int, std::allocator<int> >]"
        )


# --------------------------------------------------------------------------------------
# Bug-fix regressions: thunks, anonymous namespaces, TLS labels, clone suffixes
# --------------------------------------------------------------------------------------


class TestThunks:
    """
    Regression tests for a bug where non-virtual, virtual, and covariant-return
    thunks were all mis-parsed: the code treated the call-offset's own leading ``h``/
    ``v`` marker as a separate "thunk kind" byte to skip over, so ``parse_call_offset``
    was then invoked one character too late, found neither ``h`` nor ``v`` at its
    position, silently consumed nothing, and left the actual offset digits to be
    mis-parsed as the target encoding -- always failing outright for non-virtual
    thunks, and always mislabeled as "virtual thunk to ..." for every kind including
    the non-virtual and covariant-return cases.
    """

    def test_non_virtual_thunk(self) -> None:
        assert demangle_strict("_ZThn8_N1CD1Ev") == "non-virtual thunk to C::~C()"

    def test_virtual_thunk(self) -> None:
        assert demangle_strict("_ZTv0_n24_N1CD1Ev") == "virtual thunk to C::~C()"

    def test_covariant_return_thunk(self) -> None:
        assert (
            demangle_strict("_ZTcv0_n24_v0_n32_N1CD1Ev")
            == "covariant return thunk to C::~C()"
        )


class TestAnonymousNamespace:
    """
    Regression test for a bug where GCC's ``_GLOBAL__N_<n>`` internal spelling for an
    anonymous namespace was rendered verbatim instead of as ``(anonymous namespace)``,
    which both GNU ``c++filt`` and LLVM ``cxxfilt`` use regardless of style.
    """

    def test_renders_as_anonymous_namespace(self) -> None:
        symbol = "_ZN9anon_test12_GLOBAL__N_16Hidden3runEv"
        assert demangle_strict(symbol) == (
            "anon_test::(anonymous namespace)::Hidden::run()"
        )

    def test_renders_regardless_of_discriminator_suffix(self) -> None:
        """The trailing digits after ``_GLOBAL__N_`` vary per translation unit."""
        symbol = "_ZN13_GLOBAL__N_113fooEv"
        assert demangle_strict(symbol) == "(anonymous namespace)::foo()"


class TestThreadLocalLabels:
    """
    Regression test for a bug where the ``TH``/``TW`` special names (thread-local
    initialization/wrapper routines) always used LLVM's wording even in ``"gcc"``
    style; GNU ``c++filt`` phrases these differently ("TLS init/wrapper function for"
    rather than "thread-local initialization/wrapper routine for").
    """

    def test_gcc_style_wording(self) -> None:
        assert (
            demangle_strict("_ZTH7tls_var", style="gcc")
            == "TLS init function for tls_var"
        )
        assert (
            demangle_strict("_ZTW7tls_var", style="gcc")
            == "TLS wrapper function for tls_var"
        )

    def test_llvm_style_wording(self) -> None:
        assert (
            demangle_strict("_ZTH7tls_var", style="llvm")
            == "thread-local initialization routine for tls_var"
        )
        assert (
            demangle_strict("_ZTW7tls_var", style="llvm")
            == "thread-local wrapper routine for tls_var"
        )


class TestCloneSuffixes:
    """
    Regression tests for the clone-suffix ("``.constprop.0``"-style) handling. This
    used to be detected with a pre-parse regex that treated the first ``.`` or ``$``
    anywhere in the string as the start of a suffix, which incorrectly truncated
    legitimately mangled names containing a literal ``$`` (Apple's legacy lambda
    naming scheme mangles closures as e.g. ``$_0``). The fix instead lets the parser
    consume a full ``<encoding>`` and only treats whatever is left over as a suffix.
    """

    def test_gcc_clone_suffix(self) -> None:
        assert demangle_strict("_ZN3foo3barEv.constprop.0") == (
            "foo::bar() [clone .constprop.0]"
        )

    def test_short_clone_suffix(self) -> None:
        assert demangle_strict("_Z3fooi.cold") == "foo(int) [clone .cold]"

    def test_dollar_sign_inside_a_valid_mangled_name_is_not_a_suffix(self) -> None:
        """
        Apple's legacy ``$_N`` lambda-closure naming embeds a literal ``$`` well
        before the end of the mangled name; this must demangle fully rather than
        being truncated at the ``$``.
        """
        symbol = "_ZZ11lambda_uservENK3$_0clEi"
        assert demangle_strict(symbol) == "lambda_user()::$_0::operator()(int) const"

    def test_garbage_trailing_content_still_raises(self) -> None:
        """Trailing content that isn't a plausible clone suffix is still an error."""
        with pytest.raises(DemangleError):
            demangle_strict("_Z1fi_garbage!!!")


# --------------------------------------------------------------------------------------
# Error handling
# --------------------------------------------------------------------------------------


class TestErrorHandling:
    def test_not_a_mangled_name(self) -> None:
        with pytest.raises(DemangleError, match="not an Itanium mangled name"):
            demangle_strict("hello world")

    def test_empty_string(self) -> None:
        with pytest.raises(DemangleError):
            demangle_strict("")

    def test_unterminated_nested_name(self) -> None:
        with pytest.raises(DemangleError):
            demangle_strict("_ZN3foo")

    def test_unknown_operator(self) -> None:
        with pytest.raises(DemangleError, match="unknown operator"):
            demangle_strict("_Zzzfoo")

    def test_substitution_out_of_range(self) -> None:
        with pytest.raises(DemangleError, match="out of range"):
            demangle_strict("_Z1fS_")

    def test_source_name_length_past_end_of_string(self) -> None:
        with pytest.raises(DemangleError, match="runs past end of string"):
            demangle_strict("_Z99tooshort")

    def test_error_message_includes_offset_and_context(self) -> None:
        try:
            demangle_strict("_Zzzfoo")
        except DemangleError as exc:
            message = str(exc)
            assert "offset" in message
            assert "near" in message
        else:
            pytest.fail("expected DemangleError")


# --------------------------------------------------------------------------------------
# Non-strict demangle() and demangle_text()
# --------------------------------------------------------------------------------------


class TestDemangleNonStrict:
    def test_succeeds_like_demangle_strict(self) -> None:
        assert demangle("_ZN3foo3barEv") == demangle_strict("_ZN3foo3barEv")

    def test_returns_input_unchanged_on_failure(self) -> None:
        assert demangle("not a mangled name") == "not a mangled name"

    def test_returns_input_unchanged_on_partial_garbage(self) -> None:
        assert demangle("_Z1fi_garbage!!!") == "_Z1fi_garbage!!!"

    def test_passes_through_style_and_placeholders(self) -> None:
        symbol = "_ZN3foo1fEPiS0_"
        assert demangle(symbol, placeholders=True) == demangle_strict(
            symbol, placeholders=True
        )


class TestDemangleText:
    def test_replaces_embedded_symbol(self) -> None:
        text = "call to _ZN3foo3barEv failed"
        assert demangle_text(text) == "call to foo::bar() failed"

    def test_replaces_multiple_embedded_symbols(self) -> None:
        text = "_ZN3foo3barEv and _ZN3foo1fEPiS0_ both referenced"
        result = demangle_text(text)
        assert "foo::bar()" in result
        assert "foo::f(int*, int*)" in result

    def test_leaves_non_symbol_text_untouched(self) -> None:
        text = "no mangled names here, just plain prose."
        assert demangle_text(text) == text

    def test_leaves_unparsable_look_alike_untouched(self) -> None:
        """A ``_Z``-prefixed token that fails to parse is left as-is, not blanked out."""
        text = "see _Zzzfoo for details"
        assert demangle_text(text) == text


# --------------------------------------------------------------------------------------
# Command-line interface
# --------------------------------------------------------------------------------------


class TestCLI:
    def test_single_argument(self, capsys: pytest.CaptureFixture[str]) -> None:
        status = run(["_ZN3foo3barEv"])
        assert status == 0
        assert capsys.readouterr().out == "foo::bar()\n"

    def test_gcc_style_flag(self, capsys: pytest.CaptureFixture[str]) -> None:
        status = run(["--style", "gcc", "_ZNSt6vectorIiSaIiEE9push_backERKi"])
        assert status == 0
        out = capsys.readouterr().out
        assert "std::vector<int, std::allocator<int> >" in out

    def test_llvm_style_is_default(self, capsys: pytest.CaptureFixture[str]) -> None:
        status = run(["_ZNSt6vectorIiSaIiEE9push_backERKi"])
        assert status == 0
        out = capsys.readouterr().out
        assert "std::vector<int, std::allocator<int>>" in out

    def test_multiple_arguments(self, capsys: pytest.CaptureFixture[str]) -> None:
        status = run(["_ZN3foo3barEv", "_ZN3foo1fEPiS0_"])
        assert status == 0
        lines = capsys.readouterr().out.splitlines()
        assert lines == ["foo::bar()", "foo::f(int*, int*)"]

    def test_non_strict_passes_through_unparsable_input(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        status = run(["not_a_mangled_name"])
        assert status == 0
        assert capsys.readouterr().out == "not_a_mangled_name\n"

    def test_strict_reports_error_and_nonzero_status(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        status = run(["--strict", "_Zzzfoo"])
        assert status == 1
        captured = capsys.readouterr()
        assert captured.out == ""
        assert "Error" in captured.err

    def test_placeholders_flag(self, capsys: pytest.CaptureFixture[str]) -> None:
        status = run(["--placeholders", "_ZN3foo1fEPiS0_"])
        assert status == 0
        assert capsys.readouterr().out == "foo::f(int*, $0) [$0 = int*]\n"

    def test_reads_from_stdin_when_no_arguments_given(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(sys, "stdin", io.StringIO("_ZN3foo3barEv\n"))
        status = run([])
        assert status == 0
        assert capsys.readouterr().out == "foo::bar()\n"

    def test_blank_stdin_line_prints_blank_line(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(sys, "stdin", io.StringIO("_ZN3foo3barEv\n\n"))
        status = run([])
        assert status == 0
        assert capsys.readouterr().out == "foo::bar()\n\n"

    def test_falls_back_to_clipboard_when_stdin_is_a_tty(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        tty_stdin = io.StringIO("")
        monkeypatch.setattr(tty_stdin, "isatty", lambda: True)
        monkeypatch.setattr(sys, "stdin", tty_stdin)
        monkeypatch.setattr(
            "curu.demangle_cpp._read_clipboard", lambda: "_ZN3foo3barEv"
        )
        status = run([])
        assert status == 0
        assert capsys.readouterr().out == "foo::bar()\n"

    def test_errors_when_clipboard_unavailable_and_stdin_is_a_tty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        tty_stdin = io.StringIO("")
        monkeypatch.setattr(tty_stdin, "isatty", lambda: True)
        monkeypatch.setattr(sys, "stdin", tty_stdin)
        monkeypatch.setattr("curu.demangle_cpp._read_clipboard", lambda: None)
        with pytest.raises(SystemExit):
            run([])

    def test_build_arg_parser_defaults(self) -> None:
        parser = build_arg_parser()
        args = parser.parse_args(["_ZN3foo3barEv"])
        assert args.names == ["_ZN3foo3barEv"]
        assert args.style == "llvm"
        assert args.strict is False
        assert args.placeholders is False
        assert args.min_placeholder_length == 0
        assert args.apple_clang_21_workarounds is False
