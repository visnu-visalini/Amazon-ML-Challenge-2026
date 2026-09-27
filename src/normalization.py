import re
import unicodedata


# ---------------------------------------------------------
# BASIC UNICODE NORMALIZATION
# ---------------------------------------------------------

def normalize_unicode(text):
    if text is None:
        return ""

    text = str(text)

    # Normalize Unicode representation
    text = unicodedata.normalize("NFKC", text)

    # Lowercase
    text = text.lower()

    # Remove surrounding whitespace
    text = text.strip()

    # Collapse repeated whitespace
    text = re.sub(r"\s+", " ", text)

    return text


# ---------------------------------------------------------
# ALPHANUMERIC NORMALIZATION
# ---------------------------------------------------------

def normalize_alphanumeric(text):
    text = normalize_unicode(text)

    # Keep only letters and numbers
    text = re.sub(r"[^\w\s]", " ", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ---------------------------------------------------------
# COMPACT REPRESENTATION
# ---------------------------------------------------------

def normalize_compact(text):
    text = normalize_alphanumeric(text)

    return text.replace(" ", "")


# ---------------------------------------------------------
# TOKEN REPRESENTATION
# ---------------------------------------------------------

def normalize_tokens(text):
    text = normalize_alphanumeric(text)

    if not text:
        return []

    return text.split()


# ---------------------------------------------------------
# DIGIT EXTRACTION
# ---------------------------------------------------------

def extract_digits(text):
    if text is None:
        return ""

    return "".join(
        re.findall(r"\d+", str(text))
    )


# ---------------------------------------------------------
# ADDRESS NORMALIZATION
# ---------------------------------------------------------

def normalize_address(text):
    text = normalize_unicode(text)

    # Common punctuation → space
    text = re.sub(r"[,.;:/\\\-]+", " ", text)

    # Remove parentheses
    text = re.sub(r"[()]+", " ", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ---------------------------------------------------------
# BUSINESS NAME NORMALIZATION
# ---------------------------------------------------------

def normalize_business_name(text):
    text = normalize_unicode(text)

    # Replace & with and
    text = text.replace("&", " and ")

    # Remove punctuation
    text = re.sub(r"[^\w\s]", " ", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ---------------------------------------------------------
# DEBUG
# ---------------------------------------------------------

if __name__ == "__main__":

    examples = [
        "Payne Énterprises",
        "PAYNE-ENRTPRMISES",
        "Pvt. Ltd.",
        "Raj Investments LLP",
        "85 Wayne Avenue, Ticonderoga, NY",
        "3315 FREMONT ST, PEORIA, IL",
    ]

    for value in examples:

        print("\nOriginal:")
        print(value)

        print("Unicode:")
        print(normalize_unicode(value))

        print("Alphanumeric:")
        print(normalize_alphanumeric(value))

        print("Compact:")
        print(normalize_compact(value))

        print("Tokens:")
        print(normalize_tokens(value))

        print("Digits:")
        print(extract_digits(value))