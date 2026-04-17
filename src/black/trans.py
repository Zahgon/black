"""
String transformers that can split and merge strings.
"""

import re
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable, Collection, Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Final, Literal, TypeVar, Union

from mypy_extensions import trait

from black.comments import contains_pragma_comment
from black.lines import Line, append_leaves
from black.mode import Feature, Mode
from black.nodes import (
    CLOSING_BRACKETS,
    OPENING_BRACKETS,
    STANDALONE_COMMENT,
    is_empty_lpar,
    is_empty_par,
    is_empty_rpar,
    is_part_of_annotation,
    parent_type,
    replace_child,
    syms,
)
from black.rusty import Err, Ok, Result
from black.strings import (
    assert_is_leaf_string,
    count_chars_in_width,
    get_string_prefix,
    has_triple_quotes,
    normalize_string_quotes,
    str_width,
)
from blib2to3.pgen2 import token
from blib2to3.pytree import Leaf, Node


class CannotTransform(Exception):
    """Base class for errors raised by Transformers."""


# types
T = TypeVar("T")
LN = Union[Leaf, Node]
Transformer = Callable[[Line, Collection[Feature], Mode], Iterator[Line]]
Index = int
NodeType = int
ParserState = int
StringID = int
TResult = Result[T, CannotTransform]  # (T)ransform Result
TMatchResult = TResult[list[Index]]

SPLIT_SAFE_CHARS = frozenset(["\u3001", "\u3002", "\uff0c"])  # East Asian stops


def TErr(err_msg: str) -> Err[CannotTransform]:
    """(T)ransform Err

    Convenience function used when working with the TResult type.
    """
    pass


# Remove when `simplify_power_operator_hugging` becomes stable.
def hug_power_op(
    line: Line, features: Collection[Feature], mode: Mode
) -> Iterator[Line]:
    """A transformer which normalizes spacing around power operators."""

    # Performance optimization to avoid unnecessary Leaf clones and other ops.
    for leaf in line.leaves:
        if leaf.type == token.DOUBLESTAR:
            break
    else:
        raise CannotTransform("No doublestar token was found in the line.")

    def is_simple_lookup(index: int, kind: Literal[1, -1]) -> bool:
        # Brackets and parentheses indicate calls, subscripts, etc. ...
        # basically stuff that doesn't count as "simple". Only a NAME lookup
        # or dotted lookup (eg. NAME.NAME) is OK.
        if kind == -1:
            return handle_is_simple_look_up_prev(line, index, {token.RPAR, token.RSQB})
        else:
            return handle_is_simple_lookup_forward(
                line, index, {token.LPAR, token.LSQB}
            )

    def is_simple_operand(index: int, kind: Literal[1, -1]) -> bool:
        # An operand is considered "simple" if's a NAME, a numeric CONSTANT, a simple
        # lookup (see above), with or without a preceding unary operator.
        start = line.leaves[index]
        if start.type in {token.NAME, token.NUMBER}:
            return is_simple_lookup(index, kind)

        if start.type in {token.PLUS, token.MINUS, token.TILDE}:
            if line.leaves[index + 1].type in {token.NAME, token.NUMBER}:
                # kind is always one as bases with a preceding unary op will be checked
                # for simplicity starting from the next token (so it'll hit the check
                # above).
                return is_simple_lookup(index + 1, kind=1)

        return False

    new_line = line.clone()
    should_hug = False
    for idx, leaf in enumerate(line.leaves):
        new_leaf = leaf.clone()
        if should_hug:
            new_leaf.prefix = ""
            should_hug = False

        should_hug = (
            (0 < idx < len(line.leaves) - 1)
            and leaf.type == token.DOUBLESTAR
            and is_simple_operand(idx - 1, kind=-1)
            and line.leaves[idx - 1].value != "lambda"
            and is_simple_operand(idx + 1, kind=1)
        )
        if should_hug:
            new_leaf.prefix = ""

        # We have to be careful to make a new line properly:
        # - bracket related metadata must be maintained (handled by Line.append)
        # - comments need to copied over, updating the leaf IDs they're attached to
        new_line.append(new_leaf, preformatted=True)
        for comment_leaf in line.comments_after(leaf):
            new_line.append(comment_leaf, preformatted=True)

    yield new_line


# Remove when `simplify_power_operator_hugging` becomes stable.
def handle_is_simple_look_up_prev(line: Line, index: int, disallowed: set[int]) -> bool:
    """
    Handling the determination of is_simple_lookup for the lines prior to the doublestar
    token. This is required because of the need to isolate the chained expression
    to determine the bracket or parenthesis belong to the single expression.
    """
    contains_disallowed = False
    chain = []

    while 0 <= index < len(line.leaves):
        current = line.leaves[index]
        chain.append(current)
        if not contains_disallowed and current.type in disallowed:
            contains_disallowed = True
        if not is_expression_chained(chain):
            return not contains_disallowed

        index -= 1

    return True


# Remove when `simplify_power_operator_hugging` becomes stable.
def handle_is_simple_lookup_forward(
    line: Line, index: int, disallowed: set[int]
) -> bool:
    """
    Handling decision is_simple_lookup for the lines behind the doublestar token.
    This function is simplified to keep consistent with the prior logic and the forward
    case are more straightforward and do not need to care about chained expressions.
    """
    while 0 <= index < len(line.leaves):
        current = line.leaves[index]
        if current.type in disallowed:
            return False
        if current.type not in {token.NAME, token.DOT} or (
            current.type == token.NAME and current.value == "for"
        ):
            # If the current token isn't disallowed, we'll assume this is simple as
            # only the disallowed tokens are semantically attached to this lookup
            # expression we're checking. Also, stop early if we hit the 'for' bit
            # of a comprehension.
            return True

        index += 1

    return True


# Remove when `simplify_power_operator_hugging` becomes stable.
def is_expression_chained(chained_leaves: list[Leaf]) -> bool:
    """
    Function to determine if the variable is a chained call.
    (e.g., foo.lookup, foo().lookup, (foo.lookup())) will be recognized as chained call)
    """
    if len(chained_leaves) < 2:
        return True

    current_leaf = chained_leaves[-1]
    past_leaf = chained_leaves[-2]

    if past_leaf.type == token.NAME:
        return current_leaf.type in {token.DOT}
    elif past_leaf.type in {token.RPAR, token.RSQB}:
        return current_leaf.type in {token.RSQB, token.RPAR}
    elif past_leaf.type in {token.LPAR, token.LSQB}:
        return current_leaf.type in {token.NAME, token.LPAR, token.LSQB}
    else:
        return False


class StringTransformer(ABC):
    """
    An implementation of the Transformer protocol that relies on its
    subclasses overriding the template methods `do_match(...)` and
    `do_transform(...)`.

    This Transformer works exclusively on strings (for example, by merging
    or splitting them).

    The following sections can be found among the docstrings of each concrete
    StringTransformer subclass.

    Requirements:
        Which requirements must be met of the given Line for this
        StringTransformer to be applied?

    Transformations:
        If the given Line meets all of the above requirements, which string
        transformations can you expect to be applied to it by this
        StringTransformer?

    Collaborations:
        What contractual agreements does this StringTransformer have with other
        StringTransformers? Such collaborations should be eliminated/minimized
        as much as possible.
    """

    __name__: Final = "StringTransformer"

    # Ideally this would be a dataclass, but unfortunately mypyc breaks when used with
    # `abc.ABC`.
    def __init__(self, line_length: int, normalize_strings: bool) -> None:
        self.line_length = line_length
        self.normalize_strings = normalize_strings

    @abstractmethod
    def do_match(self, line: Line) -> TMatchResult:
        """
        Returns:
            * Ok(string_indices) such that for each index, `line.leaves[index]`
              is our target string if a match was able to be made. For
              transformers that don't result in more lines (e.g. StringMerger,
              StringParenStripper), multiple matches and transforms are done at
              once to reduce the complexity.
              OR
            * Err(CannotTransform), if no match could be made.
        """

    @abstractmethod
    def do_transform(
        self, line: Line, string_indices: list[int]
    ) -> Iterator[TResult[Line]]:
        """
        Yields:
            * Ok(new_line) where new_line is the new transformed line.
              OR
            * Err(CannotTransform) if the transformation failed for some reason. The
              `do_match(...)` template method should usually be used to reject
              the form of the given Line, but in some cases it is difficult to
              know whether or not a Line meets the StringTransformer's
              requirements until the transformation is already midway.

        Side Effects:
            This method should NOT mutate @line directly, but it MAY mutate the
            Line's underlying Node structure. (WARNING: If the underlying Node
            structure IS altered, then this method should NOT be allowed to
            yield an CannotTransform after that point.)
        """

    def __call__(
        self, line: Line, _features: Collection[Feature], _mode: Mode
    ) -> Iterator[Line]:
        """
        StringTransformer instances have a call signature that mirrors that of
        the Transformer type.

        Raises:
            CannotTransform(...) if the concrete StringTransformer class is unable
            to transform @line.
        """
        # Optimization to avoid calling `self.do_match(...)` when the line does
        # not contain any string.
        if not any(leaf.type == token.STRING for leaf in line.leaves):
            raise CannotTransform("There are no strings in this line.")

        match_result = self.do_match(line)

        if isinstance(match_result, Err):
            cant_transform = match_result.err()
            raise CannotTransform(
                f"The string transformer {self.__class__.__name__} does not recognize"
                " this line as one that it can transform."
            ) from cant_transform

        string_indices = match_result.ok()

        for line_result in self.do_transform(line, string_indices):
            if isinstance(line_result, Err):
                cant_transform = line_result.err()
                raise CannotTransform(
                    "StringTransformer failed while attempting to transform string."
                ) from cant_transform
            line = line_result.ok()
            yield line


@dataclass
class CustomSplit:
    """A custom (i.e. manual) string split.

    A single CustomSplit instance represents a single substring.

    Examples:
        Consider the following string:
        ```
        "Hi there friend."
        " This is a custom"
        f" string {split}."
        ```

        This string will correspond to the following three CustomSplit instances:
        ```
        CustomSplit(False, 16)
        CustomSplit(False, 17)
        CustomSplit(True, 16)
        ```
    """

    has_prefix: bool
    break_idx: int


CustomSplitMapKey = tuple[StringID, str]


@trait
class CustomSplitMapMixin:
    """
    This mixin class is used to map merged strings to a sequence of
    CustomSplits, which will then be used to re-split the strings iff none of
    the resultant substrings go over the configured max line length.
    """

    _CUSTOM_SPLIT_MAP: ClassVar[dict[CustomSplitMapKey, tuple[CustomSplit, ...]]] = (
        defaultdict(tuple)
    )

    @staticmethod
    def _get_key(string: str) -> CustomSplitMapKey:
        """
        Returns:
            A unique identifier that is used internally to map @string to a
            group of custom splits.
        """
        pass

    def add_custom_splits(
        self, string: str, custom_splits: Iterable[CustomSplit]
    ) -> None:
        """Custom Split Map Setter Method

        Side Effects:
            Adds a mapping from @string to the custom splits @custom_splits.
        """
        pass

    def pop_custom_splits(self, string: str) -> list[CustomSplit]:
        """Custom Split Map Getter Method

        Returns:
            * A list of the custom splits that are mapped to @string, if any
              exist.
              OR
            * [], otherwise.

        Side Effects:
            Deletes the mapping between @string and its associated custom
            splits (which are returned to the caller).
        """
        pass

    def has_custom_splits(self, string: str) -> bool:
        """
        Returns:
            True iff @string is associated with a set of custom splits.
        """
        pass


class StringMerger(StringTransformer, CustomSplitMapMixin):
    """StringTransformer that merges strings together.

    Requirements:
        (A) The line contains adjacent strings such that ALL of the validation checks
        listed in StringMerger._validate_msg(...)'s docstring pass.
        OR
        (B) The line contains a string which uses line continuation backslashes.

    Transformations:
        Depending on which of the two requirements above where met, either:

        (A) The string group associated with the target string is merged.
        OR
        (B) All line-continuation backslashes are removed from the target string.

    Collaborations:
        StringMerger provides custom split information to StringSplitter.
    """

    def do_match(self, line: Line) -> TMatchResult:
        pass

    def do_transform(
        self, line: Line, string_indices: list[int]
    ) -> Iterator[TResult[Line]]:
        pass

    @staticmethod
    def _remove_backslash_line_continuation_chars(
        line: Line, string_indices: list[int]
    ) -> TResult[Line]:
        """
        Merge strings that were split across multiple lines using
        line-continuation backslashes.

        Returns:
            Ok(new_line), if @line contains backslash line-continuation
            characters.
                OR
            Err(CannotTransform), otherwise.
        """
        pass

    def _merge_string_group(
        self, line: Line, string_indices: list[int]
    ) -> TResult[Line]:
        """
        Merges string groups (i.e. set of adjacent strings).

        Each index from `string_indices` designates one string group's first
        leaf in `line.leaves`.

        Returns:
            Ok(new_line), if ALL of the validation checks found in
            _validate_msg(...) pass.
                OR
            Err(CannotTransform), otherwise.
        """
        pass

    def _merge_one_string_group(
        self, LL: list[Leaf], string_idx: int, is_valid_index: Callable[[int], bool]
    ) -> tuple[int, Leaf]:
        """
        Merges one string group where the first string in the group is
        `LL[string_idx]`.

        Returns:
            A tuple of `(num_of_strings, leaf)` where `num_of_strings` is the
            number of strings merged and `leaf` is the newly merged string
            to be replaced in the new line.
        """
        pass

    @staticmethod
    def _validate_msg(line: Line, string_idx: int) -> TResult[None]:
        """Validate (M)erge (S)tring (G)roup

        Transform-time string validation logic for _merge_string_group(...).

        Returns:
            * Ok(None), if ALL validation checks (listed below) pass.
                OR
            * Err(CannotTransform), if any of the following are true:
                - The target string group does not contain ANY stand-alone comments.
                - The target string is not in a string group (i.e. it has no
                  adjacent strings).
                - The string group has more than one inline comment.
                - The string group has an inline comment that appears to be a pragma.
                - The set of all string prefixes in the string group is of
                  length greater than one and is not equal to {"", "f"}.
                - The string group consists of raw strings.
                - The string group would merge f-strings with different quote types
                  and internal quotes.
                - The string group is stringified type annotations. We don't want to
                  process stringified type annotations since pyright doesn't support
                  them spanning multiple string values. (NOTE: mypy, pytype, pyre do
                  support them, so we can change if pyright also gains support in the
                  future. See https://github.com/microsoft/pyright/issues/4359.)
        """
        pass


class StringParenStripper(StringTransformer):
    """StringTransformer that strips surrounding parentheses from strings.

    Requirements:
        The line contains a string which is surrounded by parentheses and:
            - The target string is NOT the only argument to a function call.
            - The target string is NOT a "pointless" string.
            - The target string is NOT a dictionary value.
            - If the target string contains a PERCENT, the brackets are not
              preceded or followed by an operator with higher precedence than
              PERCENT.

    Transformations:
        The parentheses mentioned in the 'Requirements' section are stripped.

    Collaborations:
        StringParenStripper has its own inherent usefulness, but it is also
        relied on to clean up the parentheses created by StringParenWrapper (in
        the event that they are no longer needed).
    """

    def do_match(self, line: Line) -> TMatchResult:
        pass

    def do_transform(
        self, line: Line, string_indices: list[int]
    ) -> Iterator[TResult[Line]]:
        pass

    def _transform_to_new_line(
        self, line: Line, string_and_rpar_indices: list[int]
    ) -> Line:
        pass


class BaseStringSplitter(StringTransformer):
    """
    Abstract class for StringTransformers which transform a Line's strings by splitting
    them or placing them on their own lines where necessary to avoid going over
    the configured line length.

    Requirements:
        * The target string value is responsible for the line going over the
          line length limit. It follows that after all of black's other line
          split methods have been exhausted, this line (or one of the resulting
          lines after all line splits are performed) would still be over the
          line_length limit unless we split this string.
          AND

        * The target string is NOT a "pointless" string (i.e. a string that has
          no parent or siblings).
          AND

        * The target string is not followed by an inline comment that appears
          to be a pragma.
          AND

        * The target string is not a multiline (i.e. triple-quote) string.
    """

    STRING_OPERATORS: Final = [
        token.EQEQUAL,
        token.GREATER,
        token.GREATEREQUAL,
        token.LESS,
        token.LESSEQUAL,
        token.NOTEQUAL,
        token.PERCENT,
        token.PLUS,
        token.STAR,
    ]

    @abstractmethod
    def do_splitter_match(self, line: Line) -> TMatchResult:
        """
        BaseStringSplitter asks its clients to override this method instead of
        `StringTransformer.do_match(...)`.

        Follows the same protocol as `StringTransformer.do_match(...)`.

        Refer to `help(StringTransformer.do_match)` for more information.
        """

    def do_match(self, line: Line) -> TMatchResult:
        pass

    def _validate(self, line: Line, string_idx: int) -> TResult[None]:
        """
        Checks that @line meets all of the requirements listed in this classes'
        docstring. Refer to `help(BaseStringSplitter)` for a detailed
        description of those requirements.

        Returns:
            * Ok(None), if ALL of the requirements are met.
              OR
            * Err(CannotTransform), if ANY of the requirements are NOT met.
        """
        pass

    def _get_max_string_length(self, line: Line, string_idx: int) -> int:
        """
        Calculates the max string length used when attempting to determine
        whether or not the target string is responsible for causing the line to
        go over the line length limit.

        WARNING: This method is tightly coupled to both StringSplitter and
        (especially) StringParenWrapper. There is probably a better way to
        accomplish what is being done here.

        Returns:
            max_string_length: such that `line.leaves[string_idx].value >
            max_string_length` implies that the target string IS responsible
            for causing this line to exceed the line length limit.
        """
        pass

    @staticmethod
    def _prefer_paren_wrap_match(LL: list[Leaf]) -> int | None:
        """
        Returns:
            string_idx such that @LL[string_idx] is equal to our target (i.e.
            matched) string, if this line matches the "prefer paren wrap" statement
            requirements listed in the 'Requirements' section of the StringParenWrapper
            class's docstring.
                OR
            None, otherwise.
        """
        pass


def iter_fexpr_spans(s: str) -> Iterator[tuple[int, int]]:
    """
    Yields spans corresponding to expressions in a given f-string.
    Spans are half-open ranges (left inclusive, right exclusive).
    Assumes the input string is a valid f-string, but will not crash if the input
    string is invalid.
    """
    stack: list[int] = []  # our curly paren stack
    i = 0
    while i < len(s):
        if s[i] == "{":
            # if we're in a string part of the f-string, ignore escaped curly braces
            if not stack and i + 1 < len(s) and s[i + 1] == "{":
                i += 2
                continue
            stack.append(i)
            i += 1
            continue

        if s[i] == "}":
            if not stack:
                i += 1
                continue
            j = stack.pop()
            # we've made it back out of the expression! yield the span
            if not stack:
                yield (j, i + 1)
            i += 1
            continue

        # if we're in an expression part of the f-string, fast-forward through strings
        # note that backslashes are not legal in the expression portion of f-strings
        if stack:
            delim = None
            if s[i : i + 3] in ("'''", '"""'):
                delim = s[i : i + 3]
            elif s[i] in ("'", '"'):
                delim = s[i]
            if delim:
                i += len(delim)
                while i < len(s) and s[i : i + len(delim)] != delim:
                    i += 1
                i += len(delim)
                continue
        i += 1


def fstring_contains_expr(s: str) -> bool:
    pass


def _toggle_fexpr_quotes(fstring: str, old_quote: str) -> str:
    """
    Toggles quotes used in f-string expressions that are `old_quote`.

    f-string expressions can't contain backslashes, so we need to toggle the
    quotes if the f-string itself will end up using the same quote. We can
    simply toggle without escaping because, quotes can't be reused in f-string
    expressions. They will fail to parse.

    NOTE: If PEP 701 is accepted, above statement will no longer be true.
    Though if quotes can be reused, we can simply reuse them without updates or
    escaping, once Black figures out how to parse the new grammar.
    """
    pass


class StringSplitter(BaseStringSplitter, CustomSplitMapMixin):
    """
    StringTransformer that splits "atom" strings (i.e. strings which exist on
    lines by themselves).

    Requirements:
        * The line consists ONLY of a single string (possibly prefixed by a
          string operator [e.g. '+' or '==']), MAYBE a string trailer, and MAYBE
          a trailing comma.
          AND
        * All of the requirements listed in BaseStringSplitter's docstring.

    Transformations:
        The string mentioned in the 'Requirements' section is split into as
        many substrings as necessary to adhere to the configured line length.

        In the final set of substrings, no substring should be smaller than
        MIN_SUBSTR_SIZE characters.

        The string will ONLY be split on spaces (i.e. each new substring should
        start with a space). Note that the string will NOT be split on a space
        which is escaped with a backslash.

        If the string is an f-string, it will NOT be split in the middle of an
        f-expression (e.g. in f"FooBar: {foo() if x else bar()}", {foo() if x
        else bar()} is an f-expression).

        If the string that is being split has an associated set of custom split
        records and those custom splits will NOT result in any line going over
        the configured line length, those custom splits are used. Otherwise the
        string is split as late as possible (from left-to-right) while still
        adhering to the transformation rules listed above.

    Collaborations:
        StringSplitter relies on StringMerger to construct the appropriate
        CustomSplit objects and add them to the custom split map.
    """

    MIN_SUBSTR_SIZE: Final = 6

    def do_splitter_match(self, line: Line) -> TMatchResult:
        pass

    def do_transform(
        self, line: Line, string_indices: list[int]
    ) -> Iterator[TResult[Line]]:
        pass

    def _iter_nameescape_slices(self, string: str) -> Iterator[tuple[Index, Index]]:
        r"""
        Yields:
            All ranges of @string which, if @string were to be split there,
            would result in the splitting of an \N{...} expression (which is NOT
            allowed).
        """
        pass

    def _iter_fexpr_slices(self, string: str) -> Iterator[tuple[Index, Index]]:
        """
        Yields:
            All ranges of @string which, if @string were to be split there,
            would result in the splitting of an f-expression (which is NOT
            allowed).
        """
        pass

    def _get_illegal_split_indices(self, string: str) -> set[Index]:
        pass

    def _get_break_idx(self, string: str, max_break_idx: int) -> int | None:
        """
        This method contains the algorithm that StringSplitter uses to
        determine which character to split each string at.

        Args:
            @string: The substring that we are attempting to split.
            @max_break_idx: The ideal break index. We will return this value if it
            meets all the necessary conditions. In the likely event that it
            doesn't we will try to find the closest index BELOW @max_break_idx
            that does. If that fails, we will expand our search by also
            considering all valid indices ABOVE @max_break_idx.

        Pre-Conditions:
            * assert_is_leaf_string(@string)
            * 0 <= @max_break_idx < len(@string)

        Returns:
            break_idx, if an index is able to be found that meets all of the
            conditions listed in the 'Transformations' section of this classes'
            docstring.
                OR
            None, otherwise.
        """
        pass

    def _maybe_normalize_string_quotes(self, leaf: Leaf) -> None:
        pass

    def _normalize_f_string(self, string: str, prefix: str) -> str:
        """
        Pre-Conditions:
            * assert_is_leaf_string(@string)

        Returns:
            * If @string is an f-string that contains no f-expressions, we
            return a string identical to @string except that the 'f' prefix
            has been stripped and all double braces (i.e. '{{' or '}}') have
            been normalized (i.e. turned into '{' or '}').
                OR
            * Otherwise, we return @string.
        """
        pass

    def _get_string_operator_leaves(self, leaves: Iterable[Leaf]) -> list[Leaf]:
        pass


class StringParenWrapper(BaseStringSplitter, CustomSplitMapMixin):
    """
    StringTransformer that wraps strings in parens and then splits at the LPAR.

    Requirements:
        All of the requirements listed in BaseStringSplitter's docstring in
        addition to the requirements listed below:

        * The line is a return/yield statement, which returns/yields a string.
          OR
        * The line is part of a ternary expression (e.g. `x = y if cond else
          z`) such that the line starts with `else <string>`, where <string> is
          some string.
          OR
        * The line is an assert statement, which ends with a string.
          OR
        * The line is an assignment statement (e.g. `x = <string>` or `x +=
          <string>`) such that the variable is being assigned the value of some
          string.
          OR
        * The line is a dictionary key assignment where some valid key is being
          assigned the value of some string.
          OR
        * The line is an lambda expression and the value is a string.
          OR
        * The line starts with an "atom" string that prefers to be wrapped in
          parens. It's preferred to be wrapped when it's is an immediate child of
          a list/set/tuple literal, AND the string is surrounded by commas (or is
          the first/last child).

    Transformations:
        The chosen string is wrapped in parentheses and then split at the LPAR.

        We then have one line which ends with an LPAR and another line that
        starts with the chosen string. The latter line is then split again at
        the RPAR. This results in the RPAR (and possibly a trailing comma)
        being placed on its own line.

        NOTE: If any leaves exist to the right of the chosen string (except
        for a trailing comma, which would be placed after the RPAR), those
        leaves are placed inside the parentheses.  In effect, the chosen
        string is not necessarily being "wrapped" by parentheses. We can,
        however, count on the LPAR being placed directly before the chosen
        string.

        In other words, StringParenWrapper creates "atom" strings. These
        can then be split again by StringSplitter, if necessary.

    Collaborations:
        In the event that a string line split by StringParenWrapper is
        changed such that it no longer needs to be given its own line,
        StringParenWrapper relies on StringParenStripper to clean up the
        parentheses it created.

        For "atom" strings that prefers to be wrapped in parens, it requires
        StringSplitter to hold the split until the string is wrapped in parens.
    """

    def do_splitter_match(self, line: Line) -> TMatchResult:
        pass

    @staticmethod
    def _return_match(LL: list[Leaf]) -> int | None:
        """
        Returns:
            string_idx such that @LL[string_idx] is equal to our target (i.e.
            matched) string, if this line matches the return/yield statement
            requirements listed in the 'Requirements' section of this classes'
            docstring.
                OR
            None, otherwise.
        """
        pass

    @staticmethod
    def _else_match(LL: list[Leaf]) -> int | None:
        """
        Returns:
            string_idx such that @LL[string_idx] is equal to our target (i.e.
            matched) string, if this line matches the ternary expression
            requirements listed in the 'Requirements' section of this classes'
            docstring.
                OR
            None, otherwise.
        """
        pass

    @staticmethod
    def _assert_match(LL: list[Leaf]) -> int | None:
        """
        Returns:
            string_idx such that @LL[string_idx] is equal to our target (i.e.
            matched) string, if this line matches the assert statement
            requirements listed in the 'Requirements' section of this classes'
            docstring.
                OR
            None, otherwise.
        """
        pass

    @staticmethod
    def _assign_match(LL: list[Leaf]) -> int | None:
        """
        Returns:
            string_idx such that @LL[string_idx] is equal to our target (i.e.
            matched) string, if this line matches the assignment statement
            requirements listed in the 'Requirements' section of this classes'
            docstring.
                OR
            None, otherwise.
        """
        pass

    @staticmethod
    def _dict_or_lambda_match(LL: list[Leaf]) -> int | None:
        """
        Returns:
            string_idx such that @LL[string_idx] is equal to our target (i.e.
            matched) string, if this line matches the dictionary key assignment
            statement or lambda expression requirements listed in the
            'Requirements' section of this classes' docstring.
                OR
            None, otherwise.
        """
        pass

    @staticmethod
    def _trailing_comma_tuple_match(line: Line) -> int | None:
        """
        Returns:
            string_idx such that @line.leaves[string_idx] is equal to our target
            (i.e. matched) string, if the line is a bare trailing comma tuple
            (STRING + COMMA) not inside brackets.
                OR
            None, otherwise.

        This handles the case from issue #4912 where a long string with a
        trailing comma (making it a one-item tuple) needs to be wrapped in
        parentheses before splitting to preserve AST equivalence.
        """
        pass

    def do_transform(
        self, line: Line, string_indices: list[int]
    ) -> Iterator[TResult[Line]]:
        pass


class StringParser:
    """
    A state machine that aids in parsing a string's "trailer", which can be
    either non-existent, an old-style formatting sequence (e.g. `% varX` or `%
    (varX, varY)`), or a method-call / attribute access (e.g. `.format(varX,
    varY)`).

    NOTE: A new StringParser object MUST be instantiated for each string
    trailer we need to parse.

    Examples:
        We shall assume that `line` equals the `Line` object that corresponds
        to the following line of python code:
        ```
        x = "Some {}.".format("String") + some_other_string
        ```

        Furthermore, we will assume that `string_idx` is some index such that:
        ```
        assert line.leaves[string_idx].value == "Some {}."
        ```

        The following code snippet then holds:
        ```
        string_parser = StringParser()
        idx = string_parser.parse(line.leaves, string_idx)
        assert line.leaves[idx].type == token.PLUS
        ```
    """

    DEFAULT_TOKEN: Final = 20210605

    # String Parser States
    START: Final = 1
    DOT: Final = 2
    NAME: Final = 3
    PERCENT: Final = 4
    SINGLE_FMT_ARG: Final = 5
    LPAR: Final = 6
    RPAR: Final = 7
    DONE: Final = 8

    # Lookup Table for Next State
    _goto: Final[dict[tuple[ParserState, NodeType], ParserState]] = {
        # A string trailer may start with '.' OR '%'.
        (START, token.DOT): DOT,
        (START, token.PERCENT): PERCENT,
        (START, DEFAULT_TOKEN): DONE,
        # A '.' MUST be followed by an attribute or method name.
        (DOT, token.NAME): NAME,
        # A method name MUST be followed by an '(', whereas an attribute name
        # is the last symbol in the string trailer.
        (NAME, token.LPAR): LPAR,
        (NAME, DEFAULT_TOKEN): DONE,
        # A '%' symbol can be followed by an '(' or a single argument (e.g. a
        # string or variable name).
        (PERCENT, token.LPAR): LPAR,
        (PERCENT, DEFAULT_TOKEN): SINGLE_FMT_ARG,
        # If a '%' symbol is followed by a single argument, that argument is
        # the last leaf in the string trailer.
        (SINGLE_FMT_ARG, DEFAULT_TOKEN): DONE,
        # If present, a ')' symbol is the last symbol in a string trailer.
        # (NOTE: LPARS and nested RPARS are not included in this lookup table,
        # since they are treated as a special case by the parsing logic in this
        # classes' implementation.)
        (RPAR, DEFAULT_TOKEN): DONE,
    }

    def __init__(self) -> None:
        self._state = self.START
        self._unmatched_lpars = 0

    def parse(self, leaves: list[Leaf], string_idx: int) -> int:
        """
        Pre-conditions:
            * @leaves[@string_idx].type == token.STRING

        Returns:
            The index directly after the last leaf which is a part of the string
            trailer, if a "trailer" exists.
            OR
            @string_idx + 1, if no string "trailer" exists.
        """
        assert leaves[string_idx].type == token.STRING

        idx = string_idx + 1
        while idx < len(leaves) and self._next_state(leaves[idx]):
            idx += 1
        return idx

    def _next_state(self, leaf: Leaf) -> bool:
        """
        Pre-conditions:
            * On the first call to this function, @leaf MUST be the leaf that
              was directly after the string leaf in question (e.g. if our target
              string is `line.leaves[i]` then the first call to this method must
              be `line.leaves[i + 1]`).
            * On the next call to this function, the leaf parameter passed in
              MUST be the leaf directly following @leaf.

        Returns:
            True iff @leaf is a part of the string's trailer.
        """
        # We ignore empty LPAR or RPAR leaves.
        if is_empty_par(leaf):
            return True

        next_token = leaf.type
        if next_token == token.LPAR:
            self._unmatched_lpars += 1

        current_state = self._state

        # The LPAR parser state is a special case. We will return True until we
        # find the matching RPAR token.
        if current_state == self.LPAR:
            if next_token == token.RPAR:
                self._unmatched_lpars -= 1
                if self._unmatched_lpars == 0:
                    self._state = self.RPAR
        # Otherwise, we use a lookup table to determine the next state.
        else:
            # If the lookup table matches the current state to the next
            # token, we use the lookup table.
            if (current_state, next_token) in self._goto:
                self._state = self._goto[current_state, next_token]
            else:
                # Otherwise, we check if a the current state was assigned a
                # default.
                if (current_state, self.DEFAULT_TOKEN) in self._goto:
                    self._state = self._goto[current_state, self.DEFAULT_TOKEN]
                # If no default has been assigned, then this parser has a logic
                # error.
                else:
                    raise RuntimeError(f"{self.__class__.__name__} LOGIC ERROR!")

            if self._state == self.DONE:
                return False

        return True


def insert_str_child_factory(string_leaf: Leaf) -> Callable[[LN], None]:
    """
    Factory for a convenience function that is used to orphan @string_leaf
    and then insert multiple new leaves into the same part of the node
    structure that @string_leaf had originally occupied.

    Examples:
        Let `string_leaf = Leaf(token.STRING, '"foo"')` and `N =
        string_leaf.parent`. Assume the node `N` has the following
        original structure:

        Node(
            expr_stmt, [
                Leaf(NAME, 'x'),
                Leaf(EQUAL, '='),
                Leaf(STRING, '"foo"'),
            ]
        )

        We then run the code snippet shown below.
        ```
        insert_str_child = insert_str_child_factory(string_leaf)

        lpar = Leaf(token.LPAR, '(')
        insert_str_child(lpar)

        bar = Leaf(token.STRING, '"bar"')
        insert_str_child(bar)

        rpar = Leaf(token.RPAR, ')')
        insert_str_child(rpar)
        ```

        After which point, it follows that `string_leaf.parent is None` and
        the node `N` now has the following structure:

        Node(
            expr_stmt, [
                Leaf(NAME, 'x'),
                Leaf(EQUAL, '='),
                Leaf(LPAR, '('),
                Leaf(STRING, '"bar"'),
                Leaf(RPAR, ')'),
            ]
        )
    """
    pass


def is_valid_index_factory(seq: Sequence[Any]) -> Callable[[int], bool]:
    """
    Examples:
        ```
        my_list = [1, 2, 3]

        is_valid_index = is_valid_index_factory(my_list)

        assert is_valid_index(0)
        assert is_valid_index(2)

        assert not is_valid_index(3)
        assert not is_valid_index(-1)
        ```
    """
    pass
