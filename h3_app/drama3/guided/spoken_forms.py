"""Deterministic English readings of explicit numeric symbols and abbreviations.

These equivalences preserve the written value and unit; they are not fuzzy text
matching. Unknown abbreviations and numeric formats retain their written form.
"""

import re

_SMALL = ('zero one two three four five six seven eight nine ten eleven twelve thirteen '
          'fourteen fifteen sixteen seventeen eighteen nineteen').split()
_TENS = ('zero ten twenty thirty forty fifty sixty seventy eighty ninety').split()


def _integer(value: int) -> str:
    if value < 20:
        return _SMALL[value]
    if value < 100:
        return _TENS[value // 10] + (' ' + _SMALL[value % 10] if value % 10 else '')
    if value < 1000:
        return _SMALL[value // 100] + ' hundred' + (' ' + _integer(value % 100) if value % 100 else '')
    for scale, name in [(10**12, 'trillion'), (10**9, 'billion'), (10**6, 'million'), (1000, 'thousand')]:
        if value >= scale:
            return _integer(value // scale) + ' ' + name + (' ' + _integer(value % scale) if value % scale else '')
    raise ValueError('unsupported integer')


def _number(raw: str) -> str:
    sign = 'minus ' if raw.startswith('-') else 'plus ' if raw.startswith('+') else ''
    raw = raw.lstrip('+-').replace(',', '')
    whole, dot, fraction = raw.partition('.')
    if len(whole) > 15:
        return sign + raw
    reading = (' '.join(_SMALL[int(digit)] for digit in whole) if len(whole) > 1 and whole.startswith('0')
               else _integer(int(whole)))
    return sign + reading + (' point ' + ' '.join(_SMALL[int(digit)] for digit in fraction) if dot else '')


def spoken_forms(text: str) -> str:
    for abbreviation, reading in {'Dr.': 'doctor', 'Mr.': 'mister', 'Mrs.': 'missus',
                                   'e.g.': 'for example', 'i.e.': 'that is', 'etc.': 'et cetera'}.items():
        text = re.sub(r'(?<!\w)' + re.escape(abbreviation) + r'(?=\s|$|[,;:!?])', reading, text)

    def money(match):
        sign, currency, amount = match.groups()
        whole, dot, fraction = amount.replace(',', '').partition('.')
        unit, minor = {'$': ('dollar', 'cent'), '£': ('pound', 'penny'), '€': ('euro', 'cent')}[currency]
        if len(whole) > 15 or (len(whole) > 1 and whole.startswith('0')):
            return match.group()  # do not guess how an identifier-like value is spoken
        result = _number(sign + whole) + ' ' + (unit if int(whole) == 1 else unit + 's')
        cents = int(fraction.ljust(2, '0')) if dot else 0
        if cents:
            minor_unit = ('pence' if cents != 1 else 'penny') if minor == 'penny' else minor + ('s' if cents != 1 else '')
            result += ' and ' + _integer(cents) + ' ' + minor_unit
        return result

    amount = r'(\d{1,3}(?:,\d{3})+(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?)'
    text = re.sub(r'(?<![A-Za-z0-9_.])([+-]?)([$£€])' + amount + r'(?![A-Za-z0-9_]|\.\d)', money, text)
    # Preserve unknown currency spellings as distinct tokens rather than losing symbols.
    for symbol, name in {'$': 'currency dollar', '£': 'currency pound', '€': 'currency euro', '¥': 'currency yen yuan'}.items():
        text = text.replace(symbol, ' ' + name + ' ')
    text = text.replace('%', ' percent ')
    for unit, reading in {'kg': 'kilogram', 'km': 'kilometer', 'cm': 'centimeter', 'mm': 'millimeter',
                          'mg': 'milligram', 'ms': 'millisecond'}.items():
        def measure(match, reading=reading):
            value = match.group(1)
            singular = re.fullmatch(r'0*1(?:\.0+)?', value)
            return value + ' ' + reading + ('' if singular else 's')
        text = re.sub(r'(?<!\w)(\d+(?:\.\d+)?)\s*' + unit + r'\b', measure, text)
    text = re.sub(r'(?<![\w.])[-+]?\d+(?:,\d{3})*(?:\.\d+)?(?!\w|\.\d)', lambda m: _number(m.group()), text)
    number_word = '|'.join(_SMALL + _TENS[2:])
    text = re.sub(r'\b(hundred|thousand|million|billion|trillion|dollars?|pounds?|euros?)\s+and\s+'
                  r'(?=(?:' + number_word + r')\b)', r'\1 ', text, flags=re.I)
    return text
