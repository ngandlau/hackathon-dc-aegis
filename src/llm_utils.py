import re


def extract_text_inside_xml_tags(
    text: str,
    tag: str,
    return_only_first_match: bool = False,
) -> str | None | list[str]:
    """
    Extracts the text inside a XML-tag from a text.

    ```
        # Example
        text = "outside text <answer>inside text</answer> outside text"
        extract_text_inside_tags(text=text, tag="answer")
        >>> "inside text"
    """
    # Escape special regex characters in the tag
    escaped_tag = re.escape(tag)
    pattern = r"<{tag}>([\s\S]*?)</{tag}>".format(tag=escaped_tag)
    matches: list[str] = re.findall(pattern, text)

    if not matches:
        return None

    matches = [m.strip() for m in matches]

    if return_only_first_match:
        return matches[0]

    return matches


def extract_xml_tags_from_text(text: str) -> list[str]:
    """
    Finds all XML tags with correct opening and closing tags in the given text.
    """
    # This regular expression matches opening and closing tags with the same tag name
    pattern = r"<([^<>]+)>.*?</\1>"
    # re.DOTALL flag allows '.' to match newline characters
    matches = re.findall(pattern, text, flags=re.DOTALL)
    return matches
