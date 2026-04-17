"""
Generating lines of code.
"""

import re
import sys
from collections.abc import Collection, Iterator
from dataclasses import replace
from enum import Enum, auto
from functools import partial, wraps
from typing import Union, cast

from black.brackets import (
    COMMA_PRIORITY,
    DOT_PRIORITY,
    STRING_PRIORITY,
    get_leaves_inside_matching_brackets,
    max_delimiter_priority_in_atom,
)
from black.comments import (
    FMT_OFF,
    FMT_ON,
    contains_fmt_directive,
    generate_comments,
    list_comments,
)
from black.lines import (
    Line,
    RHSResult,
    append_leaves,
    can_be_split,
    can_omit_invisible_parens,
    is_line_short_enough,
    line_to_string,
)
from black.mode import Feature, Mode, Preview
from black.nodes import (
    ASSIGNMENTS,
    BRACKETS,
    CLOSING_BRACKETS,
    OPENING_BRACKETS,
    STANDALONE_COMMENT,
    STATEMENT,
    WHITESPACE,
    Visitor,
    ensure_visible,
    fstring_tstring_to_string,
    get_annotation_type,
    has_sibling_with_type,
    is_arith_like,
    is_async_stmt_or_funcdef,
    is_atom_with_invisible_parens,
    is_docstring,
    is_empty_tuple,
    is_generator,
    is_lpar_token,
    is_multiline_string,
    is_name_token,
    is_one_sequence_between,
    is_one_tuple,
    is_parent_function_or_class,
    is_part_of_annotation,
    is_rpar_token,
    is_stub_body,
    is_stub_suite,
    is_tuple,
    is_tuple_containing_star,
    is_tuple_containing_walrus,
    is_type_ignore_comment_string,
    is_vararg,
    is_walrus_assignment,
    is_yield,
    syms,
    wrap_in_parentheses,
)
from black.numerics import normalize_numeric_literal
from black.strings import (
    fix_multiline_docstring,
    get_string_prefix,
    normalize_string_prefix,
    normalize_string_quotes,
    normalize_unicode_escape_sequences,
)
from black.trans import (
    CannotTransform,
    StringMerger,
    StringParenStripper,
    StringParenWrapper,
    StringSplitter,
    Transformer,
    hug_power_op,
)
from blib2to3.pgen2 import token
from blib2to3.pytree import Leaf, Node

# types
LeafID = int
LN = Union[Leaf, Node]


class CannotSplit(CannotTransform):
    """A readable split that fits the allotted line length is impossible."""


# This isn't a dataclass because @dataclass + Generic breaks mypyc.
# See also https://github.com/mypyc/mypyc/issues/827.
class LineGenerator(Visitor[Line]):
    """Generates reformatted Line objects.  Empty lines are not emitted.

    Note: destroys the tree it's visiting by mutating prefixes of its leaves
    in ways that will no longer stringify to valid Python code on the tree.
    """

    def __init__(self, mode: Mode, features: Collection[Feature]) -> None:
        self.mode = mode
        self.features = features
        self.current_line: Line
        self.__post_init__()

    def line(self, indent: int = 0) -> Iterator[Line]:
        """Generate a line.

        If the line is empty, only emit if it makes sense.
        If the line is too long, split it first and then generate.

        If any lines were generated, set up a new current_line.
        """
        if not self.current_line:
            self.current_line.depth += indent
            return  # Line is empty, don't emit. Creating a new one unnecessary.

        if len(self.current_line.leaves) == 1 and is_async_stmt_or_funcdef(
            self.current_line.leaves[0]
        ):
            # Special case for async def/for/with statements. `visit_async_stmt`
            # adds an `ASYNC` leaf then visits the child def/for/with statement
            # nodes. Line yields from those nodes shouldn't treat the former
            # `ASYNC` leaf as a complete line.
            return

        complete_line = self.current_line
        self.current_line = Line(mode=self.mode, depth=complete_line.depth + indent)
        yield complete_line

    def visit_default(self, node: LN) -> Iterator[Line]:
        """Default `visit_*()` implementation. Recurses to children of `node`."""
        if isinstance(node, Leaf):
            any_open_brackets = self.current_line.bracket_tracker.any_open_brackets()
            for comment in generate_comments(node, mode=self.mode):
                if any_open_brackets:
                    # any comment within brackets is subject to splitting
                    self.current_line.append(comment)
                elif comment.type == token.COMMENT:
                    # regular trailing comment
                    self.current_line.append(comment)
                    yield from self.line()

                else:
                    # regular standalone comment
                    yield from self.line()

                    self.current_line.append(comment)
                    yield from self.line()

            if any_open_brackets:
                node.prefix = ""
            if node.type not in WHITESPACE:
                self.current_line.append(node)
        yield from super().visit_default(node)

    def visit_test(self, node: Node) -> Iterator[Line]:
        """Visit an `x if y else z` test"""
        pass

    def visit_INDENT(self, node: Leaf) -> Iterator[Line]:
        """Increase indentation level, maybe yield a line."""
        pass

    def visit_DEDENT(self, node: Leaf) -> Iterator[Line]:
        """Decrease indentation level, maybe yield a line."""
        pass

    def visit_stmt(
        self, node: Node, keywords: set[str], parens: set[str]
    ) -> Iterator[Line]:
        """Visit a statement.

        This implementation is shared for `if`, `while`, `for`, `try`, `except`,
        `def`, `with`, `class`, `assert`, and assignments.

        The relevant Python language `keywords` for a given statement will be
        NAME leaves within it. This methods puts those on a separate line.

        `parens` holds a set of string leaf values immediately after which
        invisible parens should be put.
        """
        pass

    def visit_typeparams(self, node: Node) -> Iterator[Line]:
        pass

    def visit_typevartuple(self, node: Node) -> Iterator[Line]:
        pass

    def visit_paramspec(self, node: Node) -> Iterator[Line]:
        pass

    def visit_dictsetmaker(self, node: Node) -> Iterator[Line]:
        pass

    def visit_funcdef(self, node: Node) -> Iterator[Line]:
        """Visit function definition."""
        pass

    def visit_match_case(self, node: Node) -> Iterator[Line]:
        """Visit either a match or case statement."""
        pass

    def visit_suite(self, node: Node) -> Iterator[Line]:
        """Visit a suite."""
        pass

    def visit_simple_stmt(self, node: Node) -> Iterator[Line]:
        """Visit a statement without nested statements."""
        pass

    def visit_async_stmt(self, node: Node) -> Iterator[Line]:
        """Visit `async def`, `async for`, `async with`."""
        pass

    def visit_decorators(self, node: Node) -> Iterator[Line]:
        """Visit decorators."""
        pass

    def visit_power(self, node: Node) -> Iterator[Line]:
        pass

    def visit_SEMI(self, leaf: Leaf) -> Iterator[Line]:
        """Remove a semicolon and put the other statement on a separate line."""
        pass

    def visit_ENDMARKER(self, leaf: Leaf) -> Iterator[Line]:
        """End of file. Process outstanding comments and end with a newline."""
        pass

    def visit_STANDALONE_COMMENT(self, leaf: Leaf) -> Iterator[Line]:
        pass

    def visit_factor(self, node: Node) -> Iterator[Line]:
        """Force parentheses between a unary op and a binary power:

        -2 ** 8 -> -(2 ** 8)
        """
        pass

    def visit_tname(self, node: Node) -> Iterator[Line]:
        """
        Add potential parentheses around types in function parameter lists to be made
        into real parentheses in case the type hint is too long to fit on a line
        Examples:
        def foo(a: int, b: float = 7): ...

        ->

        def foo(a: (int), b: (float) = 7): ...
        """
        pass

    def visit_STRING(self, leaf: Leaf) -> Iterator[Line]:
        pass

    def visit_NUMBER(self, leaf: Leaf) -> Iterator[Line]:
        pass

    def visit_atom(self, node: Node) -> Iterator[Line]:
        """Visit any atom"""
        pass

    def visit_fstring(self, node: Node) -> Iterator[Line]:
        # If the fstring was converted to a STANDALONE_COMMENT by
        # normalize_fmt_off (e.g. it was inside a # fmt: off block),
        # skip the fstring-to-string conversion and just visit normally.
        pass

    def visit_tstring(self, node: Node) -> Iterator[Line]:
        # If the tstring was converted to a STANDALONE_COMMENT by
        # normalize_fmt_off, skip the conversion and just visit normally.
        pass

        # TODO: Uncomment Implementation to format f-string children
        # fstring_start = node.children[0]
        # fstring_end = node.children[-1]
        # assert isinstance(fstring_start, Leaf)
        # assert isinstance(fstring_end, Leaf)

        # quote_char = fstring_end.value[0]
        # quote_idx = fstring_start.value.index(quote_char)
        # prefix, quote = (
        #     fstring_start.value[:quote_idx],
        #     fstring_start.value[quote_idx:]
        # )

        # if not is_docstring(node, self.mode):
        #     prefix = normalize_string_prefix(prefix)

        # assert quote == fstring_end.value

        # is_raw_fstring = "r" in prefix or "R" in prefix
        # middles = [
        #     leaf
        #     for leaf in node.leaves()
        #     if leaf.type == token.FSTRING_MIDDLE
        # ]

        # if self.mode.string_normalization:
        #     middles, quote = normalize_fstring_quotes(quote, middles, is_raw_fstring)

        # fstring_start.value = prefix + quote
        # fstring_end.value = quote

        # yield from self.visit_default(node)

    def visit_comp_for(self, node: Node) -> Iterator[Line]:
        pass

    def visit_old_comp_for(self, node: Node) -> Iterator[Line]:
        pass

    def __post_init__(self) -> None:
        """You are in a twisty little maze of passages."""
        self.current_line = Line(mode=self.mode)

        v = self.visit_stmt
        Ø: set[str] = set()
        self.visit_assert_stmt = partial(v, keywords={"assert"}, parens={"assert", ","})
        self.visit_if_stmt = partial(
            v, keywords={"if", "else", "elif"}, parens={"if", "elif"}
        )
        self.visit_while_stmt = partial(v, keywords={"while", "else"}, parens={"while"})
        self.visit_for_stmt = partial(v, keywords={"for", "else"}, parens={"for", "in"})
        self.visit_try_stmt = partial(
            v, keywords={"try", "except", "else", "finally"}, parens=Ø
        )
        self.visit_except_clause = partial(v, keywords={"except"}, parens={"except"})
        self.visit_with_stmt = partial(v, keywords={"with"}, parens={"with"})
        self.visit_classdef = partial(v, keywords={"class"}, parens=Ø)

        self.visit_expr_stmt = partial(v, keywords=Ø, parens=ASSIGNMENTS)
        self.visit_return_stmt = partial(v, keywords={"return"}, parens={"return"})
        self.visit_import_from = partial(v, keywords=Ø, parens={"import"})
        self.visit_del_stmt = partial(v, keywords=Ø, parens={"del"})
        self.visit_async_funcdef = self.visit_async_stmt
        self.visit_decorated = self.visit_decorators

        # PEP 634
        self.visit_match_stmt = self.visit_match_case
        self.visit_case_block = self.visit_match_case
        self.visit_guard = partial(v, keywords=Ø, parens={"if"})


# Remove when `simplify_power_operator_hugging` becomes stable.
def _hugging_power_ops_line_to_string(
    line: Line,
    features: Collection[Feature],
    mode: Mode,
) -> str | None:
    try:
        return line_to_string(next(hug_power_op(line, features, mode)))
    except CannotTransform:
        return None


def transform_line(
    line: Line, mode: Mode, features: Collection[Feature] = ()
) -> Iterator[Line]:
    """Transform a `line`, potentially splitting it into many lines.

    They should fit in the allotted `line_length` but might not be able to.

    `features` are syntactical features that may be used in the output.
    """
    if line.is_comment:
        yield line
        return

    line_str = line_to_string(line)

    if Preview.simplify_power_operator_hugging in mode:
        line_str_hugging_power_ops = line_str
    else:
        # We need the line string when power operators are hugging to determine if we
        # should split the line. Default to line_str, if no power operator are present
        # on the line.
        line_str_hugging_power_ops = (
            _hugging_power_ops_line_to_string(line, features, mode) or line_str
        )

    ll = mode.line_length
    sn = mode.string_normalization
    string_merge = StringMerger(ll, sn)
    string_paren_strip = StringParenStripper(ll, sn)
    string_split = StringSplitter(ll, sn)
    string_paren_wrap = StringParenWrapper(ll, sn)

    transformers: list[Transformer]
    if (
        not line.contains_uncollapsable_type_comments()
        and not line.should_split_rhs
        and not line.magic_trailing_comma
        and (
            is_line_short_enough(line, mode=mode, line_str=line_str_hugging_power_ops)
            or line.contains_unsplittable_type_ignore()
        )
        and not (line.inside_brackets and line.contains_standalone_comments())
        and not line.contains_implicit_multiline_string_with_comments()
    ):
        # Only apply basic string preprocessing, since lines shouldn't be split here.
        if Preview.string_processing in mode:
            transformers = [string_merge, string_paren_strip]
        else:
            transformers = []
    elif line.is_def and not should_split_funcdef_with_rhs(line, mode):
        transformers = [left_hand_split]
    else:

        def _rhs(
            self: object, line: Line, features: Collection[Feature], mode: Mode
        ) -> Iterator[Line]:
            """Wraps calls to `right_hand_split`.

            The calls increasingly `omit` right-hand trailers (bracket pairs with
            content), meaning the trailers get glued together to split on another
            bracket pair instead.
            """
            pass

        # HACK: nested functions (like _rhs) compiled by mypyc don't retain their
        # __name__ attribute which is needed in `run_transformer` further down.
        # Unfortunately a nested class breaks mypyc too. So a class must be created
        # via type ... https://github.com/mypyc/mypyc/issues/884
        rhs = type("rhs", (), {"__call__": _rhs})()

        if Preview.string_processing in mode:
            if line.inside_brackets:
                transformers = [
                    string_merge,
                    string_paren_strip,
                    string_split,
                    delimiter_split,
                    standalone_comment_split,
                    string_paren_wrap,
                    rhs,
                ]
            else:
                transformers = [
                    string_merge,
                    string_paren_strip,
                    string_split,
                    string_paren_wrap,
                    rhs,
                ]
        else:
            if line.inside_brackets:
                transformers = [delimiter_split, standalone_comment_split, rhs]
            else:
                transformers = [rhs]

    if Preview.simplify_power_operator_hugging not in mode:
        # It's always safe to attempt hugging of power operations and pretty much every
        # line could match.
        transformers.append(hug_power_op)

    for transform in transformers:
        # We are accumulating lines in `result` because we might want to abort
        # mission and return the original line in the end, or attempt a different
        # split altogether.
        try:
            result = run_transformer(line, transform, mode, features, line_str=line_str)
        except CannotTransform:
            continue
        else:
            yield from result
            break

    else:
        yield line


def should_split_funcdef_with_rhs(line: Line, mode: Mode) -> bool:
    """If a funcdef has a magic trailing comma in the return type, then we should first
    split the line with rhs to respect the comma.
    """
    return_type_leaves: list[Leaf] = []
    in_return_type = False

    for leaf in line.leaves:
        if leaf.type == token.COLON:
            in_return_type = False
        if in_return_type:
            return_type_leaves.append(leaf)
        if leaf.type == token.RARROW:
            in_return_type = True

    # using `bracket_split_build_line` will mess with whitespace, so we duplicate a
    # couple lines from it.
    result = Line(mode=line.mode, depth=line.depth)
    leaves_to_track = get_leaves_inside_matching_brackets(return_type_leaves)
    for leaf in return_type_leaves:
        result.append(
            leaf,
            preformatted=True,
            track_bracket=id(leaf) in leaves_to_track,
        )

    # we could also return true if the line is too long, and the return type is longer
    # than the param list. Or if `should_split_rhs` returns True.
    return result.magic_trailing_comma is not None


class _BracketSplitComponent(Enum):
    head = auto()
    body = auto()
    tail = auto()


def left_hand_split(
    line: Line, _features: Collection[Feature], mode: Mode
) -> Iterator[Line]:
    """Split line into many lines, starting with the first matching bracket pair.

    Note: this usually looks weird, only use this for function definitions.
    Prefer RHS otherwise.  This is why this function is not symmetrical with
    :func:`right_hand_split` which also handles optional parentheses.
    """
    pass


def right_hand_split(
    line: Line,
    mode: Mode,
    features: Collection[Feature] = (),
    omit: Collection[LeafID] = (),
) -> Iterator[Line]:
    """Split line into many lines, starting with the last matching bracket pair.

    If the split was by optional parentheses, attempt splitting without them, too.
    `omit` is a collection of closing bracket IDs that shouldn't be considered for
    this split.

    Note: running this function modifies `bracket_depth` on the leaves of `line`.
    """
    rhs_result = _first_right_hand_split(line, omit=omit)
    yield from _maybe_split_omitting_optional_parens(
        rhs_result, line, mode, features=features, omit=omit
    )


def _first_right_hand_split(
    line: Line,
    omit: Collection[LeafID] = (),
) -> RHSResult:
    """Split the line into head, body, tail starting with the last bracket pair.

    Note: this function should not have side effects. It's relied upon by
    _maybe_split_omitting_optional_parens to get an opinion whether to prefer
    splitting on the right side of an assignment statement.
    """
    tail_leaves: list[Leaf] = []
    body_leaves: list[Leaf] = []
    head_leaves: list[Leaf] = []
    current_leaves = tail_leaves
    opening_bracket: Leaf | None = None
    closing_bracket: Leaf | None = None
    for leaf in reversed(line.leaves):
        if current_leaves is body_leaves:
            if leaf is opening_bracket:
                current_leaves = head_leaves if body_leaves else tail_leaves
        current_leaves.append(leaf)
        if current_leaves is tail_leaves:
            if leaf.type in CLOSING_BRACKETS and id(leaf) not in omit:
                opening_bracket = leaf.opening_bracket
                closing_bracket = leaf
                current_leaves = body_leaves
    if not (opening_bracket and closing_bracket and head_leaves):
        # If there is no opening or closing_bracket that means the split failed and
        # all content is in the tail.  Otherwise, if `head_leaves` are empty, it means
        # the matching `opening_bracket` wasn't available on `line` anymore.
        raise CannotSplit("No brackets found")

    tail_leaves.reverse()
    body_leaves.reverse()
    head_leaves.reverse()

    body: Line | None = None
    if (
        Preview.hug_parens_with_braces_and_square_brackets in line.mode
        and tail_leaves[0].value
        and tail_leaves[0].opening_bracket is head_leaves[-1]
    ):
        inner_body_leaves = list(body_leaves)
        hugged_opening_leaves: list[Leaf] = []
        hugged_closing_leaves: list[Leaf] = []
        is_unpacking = body_leaves[0].type in [token.STAR, token.DOUBLESTAR]
        unpacking_offset: int = 1 if is_unpacking else 0
        while (
            len(inner_body_leaves) >= 2 + unpacking_offset
            and inner_body_leaves[-1].type in CLOSING_BRACKETS
            and inner_body_leaves[-1].opening_bracket
            is inner_body_leaves[unpacking_offset]
        ):
            if unpacking_offset:
                hugged_opening_leaves.append(inner_body_leaves.pop(0))
                unpacking_offset = 0
            hugged_opening_leaves.append(inner_body_leaves.pop(0))
            hugged_closing_leaves.insert(0, inner_body_leaves.pop())

        if hugged_opening_leaves and inner_body_leaves:
            inner_body = bracket_split_build_line(
                inner_body_leaves,
                line,
                hugged_opening_leaves[-1],
                component=_BracketSplitComponent.body,
            )
            if (
                line.mode.magic_trailing_comma
                and inner_body_leaves[-1].type == token.COMMA
            ):
                should_hug = True
            else:
                line_length = line.mode.line_length - sum(
                    len(str(leaf))
                    for leaf in hugged_opening_leaves + hugged_closing_leaves
                )
                if is_line_short_enough(
                    inner_body, mode=replace(line.mode, line_length=line_length)
                ):
                    # Do not hug if it fits on a single line.
                    should_hug = False
                else:
                    should_hug = True
            if should_hug:
                body_leaves = inner_body_leaves
                head_leaves.extend(hugged_opening_leaves)
                tail_leaves = hugged_closing_leaves + tail_leaves
                body = inner_body  # No need to re-calculate the body again later.

    head = bracket_split_build_line(
        head_leaves, line, opening_bracket, component=_BracketSplitComponent.head
    )
    if body is None:
        body = bracket_split_build_line(
            body_leaves, line, opening_bracket, component=_BracketSplitComponent.body
        )
    tail = bracket_split_build_line(
        tail_leaves, line, opening_bracket, component=_BracketSplitComponent.tail
    )
    bracket_split_succeeded_or_raise(head, body, tail)
    return RHSResult(head, body, tail, opening_bracket, closing_bracket)


def _maybe_split_omitting_optional_parens(
    rhs: RHSResult,
    line: Line,
    mode: Mode,
    features: Collection[Feature] = (),
    omit: Collection[LeafID] = (),
) -> Iterator[Line]:
    if (
        Feature.FORCE_OPTIONAL_PARENTHESES not in features
        # the opening bracket is an optional paren
        and rhs.opening_bracket.type == token.LPAR
        and not rhs.opening_bracket.value
        # the closing bracket is an optional paren
        and rhs.closing_bracket.type == token.RPAR
        and not rhs.closing_bracket.value
        # it's not an import (optional parens are the only thing we can split on
        # in this case; attempting a split without them is a waste of time)
        and not line.is_import
        # and we can actually remove the parens
        and can_omit_invisible_parens(rhs, mode.line_length)
    ):
        omit = {id(rhs.closing_bracket), *omit}
        try:
            # The RHSResult Omitting Optional Parens.
            rhs_oop = _first_right_hand_split(line, omit=omit)
            if _prefer_split_rhs_oop_over_rhs(rhs_oop, rhs, mode):
                yield from _maybe_split_omitting_optional_parens(
                    rhs_oop, line, mode, features=features, omit=omit
                )
                return

        except CannotSplit as e:
            # For chained assignments we want to use the previous successful split
            if line.is_chained_assignment:
                pass

            elif (
                not can_be_split(rhs.body)
                and not is_line_short_enough(rhs.body, mode=mode)
                and not (
                    Preview.wrap_long_dict_values_in_parens
                    and rhs.opening_bracket.parent
                    and rhs.opening_bracket.parent.parent
                    and rhs.opening_bracket.parent.parent.type == syms.dictsetmaker
                )
            ):
                raise CannotSplit(
                    "Splitting failed, body is still too long and can't be split."
                ) from e

            elif (
                rhs.head.contains_multiline_strings()
                or rhs.tail.contains_multiline_strings()
            ):
                raise CannotSplit(
                    "The current optional pair of parentheses is bound to fail to"
                    " satisfy the splitting algorithm because the head or the tail"
                    " contains multiline strings which by definition never fit one"
                    " line."
                ) from e

    ensure_visible(rhs.opening_bracket)
    ensure_visible(rhs.closing_bracket)
    for result in (rhs.head, rhs.body, rhs.tail):
        if result:
            yield result


def _prefer_split_rhs_oop_over_rhs(
    rhs_oop: RHSResult, rhs: RHSResult, mode: Mode
) -> bool:
    """
    Returns whether we should prefer the result from a split omitting optional parens
    (rhs_oop) over the original (rhs).
    """
    # contains unsplittable type ignore
    if (
        rhs_oop.head.contains_unsplittable_type_ignore()
        or rhs_oop.body.contains_unsplittable_type_ignore()
        or rhs_oop.tail.contains_unsplittable_type_ignore()
    ):
        return True

    # Retain optional parens around dictionary values
    if (
        Preview.wrap_long_dict_values_in_parens
        and rhs.opening_bracket.parent
        and rhs.opening_bracket.parent.parent
        and rhs.opening_bracket.parent.parent.type == syms.dictsetmaker
        and rhs.body.bracket_tracker.delimiters
    ):
        # Unless the split is inside the key
        return any(leaf.type == token.COLON for leaf in rhs_oop.tail.leaves)

    # the split is right after `=`
    if not (len(rhs.head.leaves) >= 2 and rhs.head.leaves[-2].type == token.EQUAL):
        return True

    # the left side of assignment contains brackets
    if not any(leaf.type in BRACKETS for leaf in rhs.head.leaves[:-1]):
        return True

    # the left side of assignment is short enough (the -1 is for the ending optional
    # paren)
    if not is_line_short_enough(
        rhs.head, mode=replace(mode, line_length=mode.line_length - 1)
    ):
        return True

    # the left side of assignment won't explode further because of magic trailing comma
    if rhs.head.magic_trailing_comma is not None:
        return True

    # If we have multiple targets, we prefer more `=`s on the head vs pushing them to
    # the body
    rhs_head_equal_count = [leaf.type for leaf in rhs.head.leaves].count(token.EQUAL)
    rhs_oop_head_equal_count = [leaf.type for leaf in rhs_oop.head.leaves].count(
        token.EQUAL
    )
    if rhs_head_equal_count > 1 and rhs_head_equal_count > rhs_oop_head_equal_count:
        return False

    has_closing_bracket_after_assign = False
    for leaf in reversed(rhs_oop.head.leaves):
        if leaf.type == token.EQUAL:
            break
        if leaf.type in CLOSING_BRACKETS:
            has_closing_bracket_after_assign = True
            break
    return (
        # contains matching brackets after the `=` (done by checking there is a
        # closing bracket)
        has_closing_bracket_after_assign
        or (
            # the split is actually from inside the optional parens (done by checking
            # the first line still contains the `=`)
            any(leaf.type == token.EQUAL for leaf in rhs_oop.head.leaves)
            # the first line is short enough
            and is_line_short_enough(rhs_oop.head, mode=mode)
        )
    )


def bracket_split_succeeded_or_raise(head: Line, body: Line, tail: Line) -> None:
    """Raise :exc:`CannotSplit` if the last left- or right-hand split failed.

    Do nothing otherwise.

    A left- or right-hand split is based on a pair of brackets. Content before
    (and including) the opening bracket is left on one line, content inside the
    brackets is put on a separate line, and finally content starting with and
    following the closing bracket is put on a separate line.

    Those are called `head`, `body`, and `tail`, respectively. If the split
    produced the same line (all content in `head`) or ended up with an empty `body`
    and the `tail` is just the closing bracket, then it's considered failed.
    """
    tail_len = len(str(tail).strip())
    if not body:
        if tail_len == 0:
            raise CannotSplit("Splitting brackets produced the same line")

        elif tail_len < 3:
            raise CannotSplit(
                f"Splitting brackets on an empty body to save {tail_len} characters is"
                " not worth it"
            )


def _ensure_trailing_comma(
    leaves: list[Leaf], original: Line, opening_bracket: Leaf
) -> bool:
    if not leaves:
        return False
    # Ensure a trailing comma for imports
    if original.is_import:
        return True
    # ...and standalone function arguments
    if not original.is_def:
        return False
    if opening_bracket.value != "(":
        return False
    # Don't add commas if we already have any commas
    if any(
        leaf.type == token.COMMA and not is_part_of_annotation(leaf) for leaf in leaves
    ):
        return False

    # Find a leaf with a parent (comments don't have parents)
    leaf_with_parent = next((leaf for leaf in leaves if leaf.parent), None)
    if leaf_with_parent is None:
        return True
    # Don't add commas inside parenthesized return annotations
    if get_annotation_type(leaf_with_parent) == "return":
        return False
    # Don't add commas inside PEP 604 unions
    if (
        leaf_with_parent.parent
        and leaf_with_parent.parent.next_sibling
        and leaf_with_parent.parent.next_sibling.type == token.VBAR
    ):
        return False
    return True


def bracket_split_build_line(
    leaves: list[Leaf],
    original: Line,
    opening_bracket: Leaf,
    *,
    component: _BracketSplitComponent,
) -> Line:
    """Return a new line with given `leaves` and respective comments from `original`.

    If it's the head component, brackets will be tracked so trailing commas are
    respected.

    If it's the body component, the result line is one-indented inside brackets and as
    such has its first leaf's prefix normalized and a trailing comma added when
    expected.
    """
    result = Line(mode=original.mode, depth=original.depth)
    if component is _BracketSplitComponent.body:
        result.inside_brackets = True
        result.depth += 1
        if _ensure_trailing_comma(leaves, original, opening_bracket):
            for i in range(len(leaves) - 1, -1, -1):
                if leaves[i].type == STANDALONE_COMMENT:
                    continue

                if leaves[i].type != token.COMMA:
                    new_comma = Leaf(token.COMMA, ",")
                    leaves.insert(i + 1, new_comma)
                break

    leaves_to_track: set[LeafID] = set()
    if component is _BracketSplitComponent.head:
        leaves_to_track = get_leaves_inside_matching_brackets(leaves)
    # Populate the line
    for leaf in leaves:
        result.append(
            leaf,
            preformatted=True,
            track_bracket=id(leaf) in leaves_to_track,
        )
        for comment_after in original.comments_after(leaf):
            result.append(comment_after, preformatted=True)
    if component is _BracketSplitComponent.body and should_split_line(
        result, opening_bracket
    ):
        result.should_split_rhs = True
    return result


def dont_increase_indentation(split_func: Transformer) -> Transformer:
    """Normalize prefix of the first leaf in every line returned by `split_func`.

    This is a decorator over relevant split functions.
    """
    pass


def _get_last_non_comment_leaf(line: Line) -> int | None:
    pass


def _can_add_trailing_comma(leaf: Leaf, features: Collection[Feature]) -> bool:
    pass


def _safe_add_trailing_comma(safe: bool, delimiter_priority: int, line: Line) -> Line:
    pass


MIGRATE_COMMENT_DELIMITERS = {STRING_PRIORITY, COMMA_PRIORITY}


@dont_increase_indentation
def delimiter_split(
    line: Line, features: Collection[Feature], mode: Mode
) -> Iterator[Line]:
    """Split according to delimiters of the highest priority.

    If the appropriate Features are given, the split will add trailing commas
    also in function signatures and calls that contain `*` and `**`.
    """
    pass


@dont_increase_indentation
def standalone_comment_split(
    line: Line, features: Collection[Feature], mode: Mode
) -> Iterator[Line]:
    """Split standalone comments from the rest of the line."""
    pass


def normalize_invisible_parens(
    node: Node, parens_after: set[str], *, mode: Mode, features: Collection[Feature]
) -> None:
    """Make existing optional parentheses invisible or create new ones.

    `parens_after` is a set of string leaf values immediately after which parens
    should be put.

    Standardizes on visible parentheses for single-element tuples, and keeps
    existing visible parentheses for other tuples and generator expressions.
    """
    pass


def _normalize_import_from(parent: Node, child: LN, index: int) -> None:
    # "import from" nodes store parentheses directly as part of
    # the statement
    pass


def remove_await_parens(node: Node, mode: Mode, features: Collection[Feature]) -> None:
    pass


def _maybe_wrap_cms_in_parens(
    node: Node, mode: Mode, features: Collection[Feature]
) -> None:
    """When enabled and safe, wrap the multiple context managers in invisible parens.

    It is only safe when `features` contain Feature.PARENTHESIZED_CONTEXT_MANAGERS.
    """
    pass


def remove_with_parens(
    node: Node, parent: Node, mode: Mode, features: Collection[Feature]
) -> None:
    """Recursively hide optional parens in `with` statements."""
    pass


def _atom_has_magic_trailing_comma(node: LN, mode: Mode) -> bool:
    """Check if an atom node has a magic trailing comma.

    Returns True for single-element tuples with trailing commas like (a,),
    which should be preserved to maintain their tuple type.
    """
    pass


def _is_atom_multiline(node: LN) -> bool:
    """Check if an atom node is multiline (indicating intentional formatting)."""
    pass


def maybe_make_parens_invisible_in_atom(
    node: LN,
    parent: LN,
    mode: Mode,
    features: Collection[Feature],
    remove_brackets_around_comma: bool = False,
    allow_star_expr: bool = False,
) -> bool:
    """If it's safe, make the parens in the atom `node` invisible, recursively.
    Additionally, remove repeated, adjacent invisible parens from the atom `node`
    as they are redundant.

    Returns whether the node should itself be wrapped in invisible parentheses.
    """
    pass


def should_split_line(line: Line, opening_bracket: Leaf) -> bool:
    """Should `line` be immediately split with `delimiter_split()` after RHS?"""

    if not (opening_bracket.parent and opening_bracket.value in "[{("):
        return False

    # We're essentially checking if the body is delimited by commas and there's more
    # than one of them (we're excluding the trailing comma and if the delimiter priority
    # is still commas, that means there's more).
    exclude = set()
    trailing_comma = False
    try:
        last_leaf = line.leaves[-1]
        if last_leaf.type == token.COMMA:
            trailing_comma = True
            exclude.add(id(last_leaf))
        max_priority = line.bracket_tracker.max_delimiter_priority(exclude=exclude)
    except (IndexError, ValueError):
        return False

    return max_priority == COMMA_PRIORITY and (
        (line.mode.magic_trailing_comma and trailing_comma)
        # always explode imports
        or opening_bracket.parent.type in {syms.atom, syms.import_from}
    )


def generate_trailers_to_omit(line: Line, line_length: int) -> Iterator[set[LeafID]]:
    """Generate sets of closing bracket IDs that should be omitted in a RHS.

    Brackets can be omitted if the entire trailer up to and including
    a preceding closing bracket fits in one line.

    Yielded sets are cumulative (contain results of previous yields, too).  First
    set is empty, unless the line should explode, in which case bracket pairs until
    the one that needs to explode are omitted.
    """

    omit: set[LeafID] = set()
    if not line.magic_trailing_comma:
        yield omit

    length = 4 * line.depth
    opening_bracket: Leaf | None = None
    closing_bracket: Leaf | None = None
    inner_brackets: set[LeafID] = set()
    for index, leaf, leaf_length in line.enumerate_with_length(is_reversed=True):
        length += leaf_length
        if length > line_length:
            break

        has_inline_comment = leaf_length > len(leaf.value) + len(leaf.prefix)
        if leaf.type == STANDALONE_COMMENT or has_inline_comment:
            break

        if opening_bracket:
            if leaf is opening_bracket:
                opening_bracket = None
            elif leaf.type in CLOSING_BRACKETS:
                prev = line.leaves[index - 1] if index > 0 else None
                if (
                    prev
                    and prev.type == token.COMMA
                    and leaf.opening_bracket is not None
                    and not is_one_sequence_between(
                        leaf.opening_bracket, leaf, line.leaves
                    )
                ):
                    # Never omit bracket pairs with trailing commas.
                    # We need to explode on those.
                    break

                inner_brackets.add(id(leaf))
        elif leaf.type in CLOSING_BRACKETS:
            prev = line.leaves[index - 1] if index > 0 else None
            if prev and prev.type in OPENING_BRACKETS:
                # Empty brackets would fail a split so treat them as "inner"
                # brackets (e.g. only add them to the `omit` set if another
                # pair of brackets was good enough.
                inner_brackets.add(id(leaf))
                continue

            if closing_bracket:
                omit.add(id(closing_bracket))
                omit.update(inner_brackets)
                inner_brackets.clear()
                yield omit

            if (
                prev
                and prev.type == token.COMMA
                and leaf.opening_bracket is not None
                and not is_one_sequence_between(leaf.opening_bracket, leaf, line.leaves)
            ):
                # Never omit bracket pairs with trailing commas.
                # We need to explode on those.
                break

            if leaf.value:
                opening_bracket = leaf.opening_bracket
                closing_bracket = leaf


def run_transformer(
    line: Line,
    transform: Transformer,
    mode: Mode,
    features: Collection[Feature],
    *,
    line_str: str = "",
) -> list[Line]:
    if not line_str:
        line_str = line_to_string(line)
    result: list[Line] = []
    for transformed_line in transform(line, features, mode):
        if str(transformed_line).strip("\n") == line_str:
            raise CannotTransform("Line transformer returned an unchanged result")

        result.extend(transform_line(transformed_line, mode=mode, features=features))

    features_set = set(features)
    if (
        Feature.FORCE_OPTIONAL_PARENTHESES in features_set
        or transform.__class__.__name__ != "rhs"
        or not line.bracket_tracker.invisible
        or any(bracket.value for bracket in line.bracket_tracker.invisible)
        or line.contains_multiline_strings()
        or result[0].contains_uncollapsable_type_comments()
        or result[0].contains_unsplittable_type_ignore()
        or is_line_short_enough(result[0], mode=mode)
        # If any leaves have no parents (which _can_ occur since
        # `transform(line)` potentially destroys the line's underlying node
        # structure), then we can't proceed. Doing so would cause the below
        # call to `append_leaves()` to fail.
        or any(leaf.parent is None for leaf in line.leaves)
    ):
        return result

    line_copy = line.clone()
    append_leaves(line_copy, line, line.leaves)
    features_fop = features_set | {Feature.FORCE_OPTIONAL_PARENTHESES}
    second_opinion = run_transformer(
        line_copy, transform, mode, features_fop, line_str=line_str
    )
    if all(is_line_short_enough(ln, mode=mode) for ln in second_opinion):
        result = second_opinion
    return result
