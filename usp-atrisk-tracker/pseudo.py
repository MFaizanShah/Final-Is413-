"""
pseudo.py
Keyed pseudonymisation of student identifiers.

Real student numbers are never stored. On upload each ID is replaced with
an HMAC-SHA256 code derived from the secret pseudonym key and the course
code, so the same student gets the same code in every upload of that course
(needed to replace and compare datasets) but a different code in other
courses (so records cannot be linked across courses).

Someone who already knows a student's real ID can find that student by
hashing it again (see lookup in the dashboard search). Nobody can reverse a
code without the key.
"""

import hashlib
import hmac
import re

from config import PSEUDONYM_KEY


def normalise_id(raw):
    return re.sub(r"\s+", "", str(raw or "")).upper()


def pseudonymise(course_code, raw_id):
    text = normalise_id(raw_id)
    if not text:
        return ""
    digest = hmac.new(PSEUDONYM_KEY.encode(), f"{course_code.upper()}|{text}".encode(), hashlib.sha256).hexdigest()
    return "P" + digest[:10].upper()


def looks_like_pseudonym(text):
    return bool(re.fullmatch(r"P[0-9A-F]{10}", (text or "").strip().upper()))
