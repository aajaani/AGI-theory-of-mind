import pytest
from arc_agi_benchmarking.utils.parsing import backscan_json_parser, parse_and_validate_json

# Test cases for backscan_json_parser

def test_backscan_valid_json_at_end():
    log = "Some preceding text [[1, 2], [3, 4]]"
    expected = [[1, 2], [3, 4]]
    assert backscan_json_parser(log) == expected

def test_backscan_valid_json_only():
    log = "[[1, 2], [3, 4]]"
    expected = [[1, 2], [3, 4]]
    assert backscan_json_parser(log) == expected

def test_backscan_nested_json():
    log = "Log data... [[1, [2, 2]], [3, [4, 4]]]"
    expected = [[1, [2, 2]], [3, [4, 4]]]
    assert backscan_json_parser(log) == expected

def test_backscan_no_json():
    log = "This string has no json content."
    assert backscan_json_parser(log) is None

def test_backscan_incomplete_json_missing_closing():
    log = "Some logs [[1, 2], [3, 4"
    assert backscan_json_parser(log) is None

def test_backscan_incomplete_json_missing_opening():
    # This case might be tricky depending on implementation, assuming it looks for balanced brackets
    log = "Some logs [1, 2], [3, 4]]"
    # The corrected logic finds the last ']', scans back to the matching '[',
    # extracts '[3, 4]', and successfully parses it.
    # HOWEVER, the function must return List[List] or None.
    # Since [3, 4] is not List[List], it should return None.
    assert backscan_json_parser(log) is None

def test_backscan_invalid_json_structure():
    # This case has valid brackets `[]` but invalid JSON inside.
    log = "Log with invalid json [1, 2],[3, 4]]" # Missing outer list brackets, comma is invalid
    # The backscan finds the final ']', scans back to the first '['.
    # Extracts '[1, 2],[3, 4]' which is NOT valid JSON.
    assert backscan_json_parser(log) is None

def test_backscan_empty_string():
    log = ""
    assert backscan_json_parser(log) is None

def test_backscan_only_brackets():
    log = "[]"
    # Parses to [], which is not List[List]. Should return None.
    assert backscan_json_parser(log) is None

def test_backscan_empty_inner_list():
    log = "[[]]"
    expected = [[]]
    assert backscan_json_parser(log) == expected

def test_backscan_multiple_json_blocks():
    log = "First block [1, 2] then some text [[3, 4], [5, 6]]"
    expected = [[3, 4], [5, 6]]
    assert backscan_json_parser(log) == expected

def test_backscan_json_with_internal_brackets():
    log = "Data [annoying] [[1], [2]]"
    expected = [[1], [2]]
    assert backscan_json_parser(log) == expected

def test_backscan_malformed_but_decodable_json():
    # The parser might find the brackets but json.loads fails
    log = "Text [[1, 2], [3, 4]" # Malformed, missing ]
    assert backscan_json_parser(log) is None

def test_backscan_no_closing_bracket_at_all():
    log = "Text [[1, 2], [3, 4"
    assert backscan_json_parser(log) is None

def test_backscan_no_opening_bracket_found():
    log = "Text 1, 2], [3, 4]]"
    # Current logic finds the last ']' and scans back for '['.
    # It should find the matching '[' for the final ']' which starts before 3.
    # Parses to [3, 4]. This is not List[List]. Should return None.
    assert backscan_json_parser(log) is None

def test_boxed_json():
    response = r""" ### Summary:\nWe analyzed the training examples to identify the pattern that transforms the input into the output. The output consists of six lists derived from the two input lists as follows:\n1. The first input list is repeated three times.\n2. The second input list is repeated three times.\n3. The second and first elements of the first input list are alternated three times.\n4. The second and first elements of the second input list are alternated three times.\n5. Repeat of the first output list.\n6. Repeat of the second output list.\n\nApplying this pattern to the test input `[[3, 2], [7, 8]]`, we constructed the output by following these steps systematically.\n\nFinal Output:\n```\n[[3, 2, 3, 2, 3, 2], [7, 8, 7, 8, 7, 8], [2, 3, 2, 3, 2, 3], [8, 7, 8, 7, 8, 7], [3, 2, 3, 2, 3, 2], [7, 8, 7, 8, 7, 8]]\n```\n\n\boxed{[ [3, 2, 3, 2, 3, 2], [7, 8, 7, 8, 7, 8], [2, 3, 2, 3, 2, 3], [8, 7, 8, 7, 8, 7], [3, 2, 3, 2, 3, 2], [7, 8, 7, 8, 7, 8] ]}"""
    
    expected_output = [[3, 2, 3, 2, 3, 2], [7, 8, 7, 8, 7, 8], [2, 3, 2, 3, 2, 3], [8, 7, 8, 7, 8, 7], [3, 2, 3, 2, 3, 2], [7, 8, 7, 8, 7, 8]]
    
    parsed_output = parse_and_validate_json(response)
    assert parsed_output == expected_output 