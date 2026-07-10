import re

def validate_phone(phone: str) -> bool:
    """Basic phone validation."""
    return bool(re.match(r"^\+?[\d\s\-\(\)]{10,15}$", phone))

def validate_credit_card(number: str) -> bool:
    """Luhn algorithm check."""
    digits = [int(d) for d in number if d.isdigit()]
    checksum = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        checksum += d
    return checksum % 10 == 0

def validate_time(time_str: str) -> bool:
    """Basic time validation."""
    return bool(re.match(r"^\d{1,2}:\d{2}\s?(AM|PM|am|pm)?$", time_str))

def validate_expiry(expiry: str) -> bool:
    """Validate MM/YY format."""
    return bool(re.match(r"^(0[1-9]|1[0-2])\/\d{2}$", expiry))

def validate_cvv(cvv: str) -> bool:
    """Validate 3 or 4 digit CVV."""
    return bool(re.match(r"^\d{3,4}$", cvv))