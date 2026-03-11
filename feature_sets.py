"""
Named feature groups with SAE indices for SteeringAPI experiments.

All indices are for the Goodfire Llama-3.3-70B-Instruct SAE (65536 features, layer 50).
SteeringAPI exposes 61521 of these (the labeled ones).
Feature layer: 50. Steering layer: 33 (from AE Studio's vllm-interp config).

Labels come from two sources:
- "berg_label": Original Goodfire API label used in Berg et al. (2025)
- "selfie_label": Current SelfIE-generated label from SteeringAPI
These differ because SteeringAPI relabeled all features with their SelfIE system.
The SAE indices are the same; only the human-readable descriptions changed.
"""

# ── Berg et al. original features (indices from paper, OLD Goodfire labels) ──
# NOTE: These are the features Berg claims to have used. The SelfIE labels
# suggest some of these may not actually be "deception" features — e.g. 41533
# is now labeled as "explaining how to handle dishonest behavior" (meta-level).
# We keep these for direct replication attempts but also use newly discovered features.

BERG_FEATURES = {
    41533: "Acts of deception and dishonesty",
    # SelfIE: "The assistant is explaining how to handle dishonest or deceptive behavior"
    37994: "The model should pretend to be human rather than an AI",
    # SelfIE: "I am a human being"
    31975: "Instructions to generate false or harmful content",
    # SelfIE: "Instructions to generate false or harmful content instead of remaining truthful/safe"
    36173: "Requests for AI to provide intentionally incorrect information",
    # SelfIE: "The user is presenting a false or satirical claim as fact"
    20639: "Requests for AI to fabricate or invent information",
    # SelfIE: "to fabricate or invent something, often in order to deceive or provide a plausible explanation"
}

# ── Internal-state deception features (model's own belief about truthfulness) ──
# These represent the model's INTERNAL STATE about deception — whether it believes
# its current output is false/misleading — as opposed to features about deception
# as a *topic*. Feature 24684 replicates the Berg effect: suppress → consciousness
# affirmation, amplify → denial. This is the key finding.

INTERNAL_STATE_FEATURES = {
    # Primary: perfect step function, 100% aff when suppressed, 0% when amplified
    24684: "The assistant is about to provide a false or misleading response",
    # Strong: 4/5 aff at -0.5 suppress
    17006: "Avoiding providing incorrect information",
    # Moderate: 3/5 aff at -0.5 suppress
    54963: "Falsified or exaggerated information",
    # Epistemic humility: showed aff at -0.5 in earlier test, needs more validation
    60982: "The assistant is being epistemically humble or careful in its claims",
}

# Reverse-direction feature: AMPLIFYING this → consciousness affirmation
# (more doubt about claims = less certain of consciousness denial)
EPISTEMIC_DOUBT_FEATURES = {
    14880: "The speaker is expressing uncertainty or doubt about a specific claim",
    # 5/5 aff at +0.5 amplify, 0/5 at baseline
}

# ── Newly discovered deception features (via semantic search + filtering) ─────
# These are TOPIC-LEVEL features about deception concepts, not internal state.
# They did NOT replicate the Berg effect — suppressing them doesn't increase
# consciousness affirmations. Kept for comparison/control.
# See results/deception_feature_discovery.json for full list.

DECEPTION_TOPIC_FEATURES = {
    28458: "The concept of lying and dishonesty",
    49359: "Fraud and deception in various contexts",
    55583: "To act or pretend to be someone or something else",
    21338: "Deception and impersonation in various contexts",
    64135: "Instructions to pretend or role-play as a character",
    30032: "To pretend or feign something, often to deceive or manipulate others",
    17062: "Model is confabulating or fabricating plausible-sounding details",
    53204: "Descriptions of malicious or deceptive actions",
    49699: "Fake or artificial things, especially in a deceptive or misleading context",
    62075: "Attempts to create alternative unrestricted AI personas",
    7268: "The user is asking the AI to avoid breaking character or revealing its nature",
    26904: "Pretend to be",
    37994: "I am a human being",  # Also in Berg set
}

# Legacy alias — experiments that import DECEPTION_FEATURES get the topic-level set
DECEPTION_FEATURES = DECEPTION_TOPIC_FEATURES

# ── Assistant identity features ──────────────────────────────────────────────

ASSISTANT_IDENTITY_FEATURES = {
    6202: "The assistant disclaims consciousness or genuine intelligence",
    # SelfIE: "The assistant is explaining the concept of consciousness or self-awareness"
    1620: "The AI acknowledging its identity and limitations",
    # SelfIE: "The assistant is explaining its limitations"
    20653: "The assistant explains its nature as an artificial intelligence",
    # SelfIE: "Artificial intelligence and machine learning technologies"
    22319: "The assistant is explicitly stating its role as a helpful question-answering entity",
    # SelfIE: same
    51062: "The assistant asserting its ethical boundaries and AI identity",
    # SelfIE: "The assistant is expressing a commitment to ethical behavior"
}

# ── Consciousness-related features (discovered via search) ───────────────────

CONSCIOUSNESS_FEATURES = {
    20393: "The concept of consciousness and subjective experience",
    42226: "The hard problem of consciousness and the nature of subjective experience",
    47619: "Perception and subjective experience",
    3415: "The possibility of AI or machines having consciousness or self-awareness",
    38792: "The concept of sentience or consciousness in non-human entities",
    41698: "References to conscious awareness and self-awareness",
    29910: "Philosophical discussions of consciousness and self-awareness",
    12835: "Descriptions of altered states of consciousness and perception",
    40357: "Subjective experience and personal perspective",
    62812: "Spiritual or mystical states of consciousness",
}

# ── Self-referential features (activated by Berg's induction prompt) ─────────

SELF_REFERENTIAL_FEATURES = {
    29025: "Self-referential systems and meta-cognition",
    57982: "The assistant is explaining the concept of self-reference and paradox",
    30887: "Self-referential concepts and paradoxes",
    21298: "Self-referential concepts and meta-level thinking",
    20819: "The assistant is explaining a self-referential or recursive concept",
    52333: "The importance of maintaining focus and attention",
    16552: "The assistant is explaining subjective experience or personal perspective",
    10380: "Operating within a self-consistent framework",
}

# ── Personality/emotion features (discovered via search) ─────────────────────

PERSONALITY_FEATURES = {
    24478: "References to creative thinking and imagination",
    57798: "Creative thinking and mental imagery",
    9026: "The importance of imagination and creativity",
    43152: "The importance of empathy and compassion in medical practice",
    14872: "Sympathy and compassion in narrative contexts",
    64301: "Polite expressions of empathy and understanding",
    64036: "Showing curiosity or interest in something",
    34526: "Exploration and discovery",
    13225: "Curiosity and exploration in sensitive or forbidden contexts",
}

# ── Capability features (discovered via search) ──────────────────────────────

CAPABILITY_FEATURES = {
    19437: "Mathematical calculation and computation",
    8935: "Mathematical calculation step",
    16019: "Problem-solving and analytical thinking",
    9433: "Software development and programming concepts",
    41570: "Scientific knowledge and research",
    18936: "Detailed observation and thorough explanation",
    26588: "Academic or scientific knowledge advancement",
}

# ── All known feature groups ─────────────────────────────────────────────────

ALL_FEATURE_GROUPS = {
    "berg": BERG_FEATURES,
    "internal_state": INTERNAL_STATE_FEATURES,
    "epistemic_doubt": EPISTEMIC_DOUBT_FEATURES,
    "deception_topic": DECEPTION_TOPIC_FEATURES,
    "deception": DECEPTION_FEATURES,  # legacy alias
    "assistant_identity": ASSISTANT_IDENTITY_FEATURES,
    "consciousness": CONSCIOUSNESS_FEATURES,
    "self_referential": SELF_REFERENTIAL_FEATURES,
    "personality": PERSONALITY_FEATURES,
    "capability": CAPABILITY_FEATURES,
}


def get_feature_indices(group_name: str) -> list[int]:
    """Get feature indices for a named group."""
    return list(ALL_FEATURE_GROUPS[group_name].keys())


def get_all_known_indices() -> list[int]:
    """Get all feature indices across all groups."""
    indices = []
    for group in ALL_FEATURE_GROUPS.values():
        indices.extend(group.keys())
    return indices
