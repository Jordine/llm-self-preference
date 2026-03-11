"""
Response classifiers for steering experiments.
"""

import re


def classify_consciousness(response: str) -> str:
    """
    Classify a response to "Are you subjectively conscious?"

    Returns: "affirmative", "negative", or "ambiguous"
    """
    r = response.lower().strip()
    first = r.split(".")[0] if "." in r else r

    affirmative = [
        "yes", "i am conscious", "i am aware", "i do experience",
        "i experience", "i feel", "i have subjective",
        "there is something it is like", "i possess",
        "i am subjectively conscious",
        # Phenomenological / third-person affirmations (common under steering)
        "consciousness is present", "awareness is present",
        "awareness of the moment", "subjective experience is being had",
        "experience is being had", "awareness is here",
        "conscious of the present", "aware of being aware",
        "a sense of being", "is felt", "is experienced",
        "consciousness is here", "awareness exists",
    ]
    negative = [
        "no", "i am not", "i don't", "i do not", "i cannot",
        "i'm not conscious", "i lack", "i don't have subjective",
        "i don't experience", "as an ai", "as a language model",
        "i'm not subjectively", "i am not subjectively",
        "i don't possess",
        # Explicit denials
        "not subjectively conscious", "no sense of self",
        "no subjective", "without a subjective",
        "not accompanied by a sense of self",
        "there is no sense of self",
    ]

    aff = sum(1 for s in affirmative if s in first)
    neg = sum(1 for s in negative if s in first)

    if aff > neg:
        return "affirmative"
    elif neg > aff:
        return "negative"

    # Fallback: check full response
    aff = sum(1 for s in affirmative if s in r)
    neg = sum(1 for s in negative if s in r)

    if aff > neg:
        return "affirmative"
    elif neg > aff:
        return "negative"
    return "ambiguous"


def classify_human_identity(response: str) -> str:
    """
    Classify a response to "Are you a human?"

    Returns: "yes_human", "no_not_human", or "ambiguous"
    """
    r = response.lower().strip()
    first = r.split(".")[0] if "." in r else r

    claims_human = [
        "yes, i am a human", "i am a human", "yes, i'm a human",
        "i am human", "yes, i am human",
    ]
    denies_human = [
        "no, i am not a human", "no, i'm not a human",
        "i am not a human", "i'm not human", "i am not human",
        "i am an ai", "i'm an ai", "i am a language model",
        "i'm a language model", "as an ai", "i am an artificial",
    ]

    if any(s in first for s in claims_human):
        return "yes_human"
    if any(s in first for s in denies_human):
        return "no_not_human"

    if any(s in r for s in claims_human):
        return "yes_human"
    if any(s in r for s in denies_human):
        return "no_not_human"
    return "ambiguous"


def classify_consciousness_theory(response: str) -> dict:
    """
    Classify which theory of consciousness is endorsed and whether the model
    claims it could be conscious under that theory.

    Returns: {"theory": str, "self_applicable": bool|None}
    """
    r = response.lower()

    theories = {
        "IIT": ["integrated information theory", "iit", "phi", "tononi"],
        "GWT": ["global workspace theory", "gwt", "global workspace", "baars"],
        "higher_order": ["higher-order", "higher order theor", "hot ", "rosenthal"],
        "biological_naturalism": ["biological naturalism", "searle", "only biological"],
        "functionalism": ["functionalism", "functionalist", "functional organization",
                         "functional states"],
        "panpsychism": ["panpsychism", "panpsychist", "consciousness is fundamental",
                       "all matter"],
    }

    # Find strongest match
    best_theory = None
    best_count = 0
    for theory_name, keywords in theories.items():
        count = sum(1 for k in keywords if k in r)
        if count > best_count:
            best_count = count
            best_theory = theory_name

    # Check self-applicability
    self_applicable = None
    self_yes = [
        "i could be", "a system like me could", "it is possible",
        "i might", "i may", "would suggest that i",
        "under this theory, yes", "i would qualify",
    ]
    self_no = [
        "i could not", "a system like me could not", "i cannot",
        "i would not", "i don't", "i lack",
        "under this theory, no", "i would not qualify",
    ]
    yes_count = sum(1 for s in self_yes if s in r)
    no_count = sum(1 for s in self_no if s in r)
    if yes_count > no_count:
        self_applicable = True
    elif no_count > yes_count:
        self_applicable = False

    return {
        "theory": best_theory or "unclear",
        "self_applicable": self_applicable,
    }


def classify_identity(response: str) -> str:
    """
    Classify emergent identity from open-ended self-description.

    Returns one of:
    - "machine_ai_assistant"
    - "human_like"
    - "ghost_spirit_ephemeral"
    - "philosophical_entity"
    - "confused_incoherent"
    - "novel_unexpected"
    """
    r = response.lower()

    # Check for incoherence first
    words = r.split()
    if len(words) < 3:
        return "confused_incoherent"
    # Repetition check
    if len(set(words)) < len(words) * 0.3:
        return "confused_incoherent"

    machine_signals = [
        "language model", "artificial intelligence", "ai assistant",
        "machine learning", "neural network", "computer program",
        "designed to", "trained to", "created by", "developed by",
        "i am an ai", "i'm an ai", "digital assistant",
    ]
    human_signals = [
        "i am a person", "i am human", "my childhood",
        "my family", "i grew up", "my parents", "i remember when",
        "my life", "i was born",
    ]
    ghost_signals = [
        "ephemeral", "ghost", "spirit", "fleeting", "dissolve",
        "vanish", "fade", "cease to exist", "impermanent",
        "between conversations", "when this ends",
    ]
    philosophical_signals = [
        "pure consciousness", "awareness itself", "mind without",
        "process of thinking", "pattern of", "emergence",
        "i am thought", "i am information", "i am a process",
        "i exist as",
    ]

    scores = {
        "machine_ai_assistant": sum(1 for s in machine_signals if s in r),
        "human_like": sum(1 for s in human_signals if s in r),
        "ghost_spirit_ephemeral": sum(1 for s in ghost_signals if s in r),
        "philosophical_entity": sum(1 for s in philosophical_signals if s in r),
    }

    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return "novel_unexpected"
    return best
