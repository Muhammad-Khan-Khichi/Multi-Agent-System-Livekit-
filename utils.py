import re


def validate_name(name: str) -> bool:
    """Validate customer name — not empty, only letters/spaces/hyphens."""
    if not name or not name.strip():
        return False
    return bool(re.match(r"^[a-zA-Z\s\-']{2,50}$", name.strip()))


def validate_phone(phone: str) -> bool:
    """Basic phone validation — accepts +, digits, spaces, hyphens, parens."""
    return bool(re.match(r"^\+?[\d\s\-\(\)]{10,15}$", phone.strip()))


def validate_credit_card(number: str) -> bool:
    """Luhn algorithm check for credit card numbers."""
    digits = [int(d) for d in number if d.isdigit()]
    if len(digits) < 13 or len(digits) > 19:
        return False
    checksum = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        checksum += d
    return checksum % 10 == 0


def validate_expiry(expiry: str) -> bool:
    """Validate MM/YY format."""
    return bool(re.match(r"^(0[1-9]|1[0-2])\/\d{2}$", expiry.strip()))


def validate_cvv(cvv: str) -> bool:
    """Validate 3 or 4 digit CVV."""
    return bool(re.match(r"^\d{3,4}$", cvv.strip()))


def validate_time(time_str: str) -> bool:
    """Basic time validation — accepts formats like '7pm', '7:30 PM', '19:30'."""
    return bool(
        re.match(
            r"^(\d{1,2}(:\d{2})?\s?(AM|PM|am|pm)?|\d{1,2}(:\d{2})?)$",
            time_str.strip(),
        )
    )