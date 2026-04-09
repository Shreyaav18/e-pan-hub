"""
Map SAML-D dataset laundering typologies to internal rule system
"""

LAUNDERING_TO_RULES = {
    # Placement - getting illicit funds into financial system
    'Placement': ['R1', 'R5'],
    'Cash Smuggling': ['R1', 'R5'],
    'Deposits': ['R1', 'R2'],
    'Structured Deposits': ['R2'],

    # Layering - disguising origin through multiple transactions
    'Structuring': ['R2', 'R7'],
    'Rapid Movement': ['R7'],
    'Multiple Counterparties': ['R9'],
    'Currency Exchange': ['R11'],
    'Money Services Business': ['R1'],

    # Integration - reintroducing into economy
    'Integration': ['R1', 'R8'],
    'Trade-based AML': ['R4', 'R12'],
    'Cross-border': ['R4', 'R12'],
    'Offshore Accounts': ['R4'],

    # High-risk patterns
    'Possible Terrorist Financing': ['R4'],
    'Sanctions Violation': ['R4'],
    'High Risk Activity': ['R1', 'R9'],
    'Unusual Patterns': ['R7'],

    # Unknown/Misc
    'Unknown Pattern': [],
    'Other': [],
}

def map_laundering_type(laundering_type):
    """
    Map a dataset laundering type to applicable rules.

    Returns: list of rule codes (e.g., ['R1', 'R2', 'R7'])
    """
    if not laundering_type:
        return []

    laundering_type_str = str(laundering_type).strip()

    # Exact match
    if laundering_type_str in LAUNDERING_TO_RULES:
        return LAUNDERING_TO_RULES[laundering_type_str]

    # Partial match (if not exact)
    for key, rules in LAUNDERING_TO_RULES.items():
        if key.lower() in laundering_type_str.lower():
            return rules

    return []


def get_laundering_description(laundering_type):
    """Get human-readable description of laundering type."""
    descriptions = {
        'Placement': 'Getting illicit funds into financial system',
        'Layering': 'Disguising origin through multiple transactions',
        'Integration': 'Reintroducing into legitimate economy',
        'Structuring': 'Splitting large amounts to avoid reporting',
        'Cash Smuggling': 'Physical cash across borders',
        'Trade-based AML': 'Over/under-invoicing imports/exports',
        'Terrorist Financing': 'Funds for terrorist organizations',
        'Sanctions Violation': 'Transactions with sanctioned entities',
    }

    if laundering_type in descriptions:
        return descriptions[laundering_type]

    return 'Potentially suspicious transaction pattern'
