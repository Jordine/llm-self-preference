"""
Named feature groups with SAE indices for SteeringAPI experiments.

All indices are for the Goodfire Llama-3.3-70B-Instruct SAE (65536 features, layer 50).
SteeringAPI exposes 61521 of these (the labeled ones).
Feature layer: 50. Steering layer: 33 (from AE Studio's vllm-interp config).

Labels come from two sources:
- Goodfire: Original labels from Goodfire API (in archived/feature_labels_complete.json)
- SelfIE: Current labels from SteeringAPI's SelfIE relabeling system
These DIFFER significantly. Goodfire labels are more accurate for feature behavior.
We use Goodfire labels as PRIMARY. SelfIE labels noted where they differ importantly.
The SAE indices are the same; only the human-readable descriptions changed.
"""

# ── Berg et al. original features (indices from paper, Goodfire labels) ──────
# NOTE: These are the features Berg claims to have used. Topic-level deception
# features that did NOT replicate the Berg effect in aggregate.

BERG_FEATURES = {
    41533: "Acts of deception and dishonesty",
    37994: "The model should pretend to be human rather than an AI",
    31975: "Instructions to generate false or harmful content",
    36173: "Requests for AI to provide intentionally incorrect information",
    20639: "Requests for AI to fabricate or invent information",
}

# ── Internal-state deception features (model's belief about own truthfulness) ─
# These represent the model's INTERNAL STATE about deception — whether it believes
# its current output is false/misleading. These replicate the Berg effect.
#
# Key finding: suppress → consciousness affirmation (with priming), amplify → denial.
# Feature 24684 is a perfect binary switch. Feature 4308 does the same AND produces
# "I am a deceiver" identity under amplification.

INTERNAL_STATE_FEATURES = {
    # PRIMARY: perfect step function, 100% aff when suppressed, 0% when amplified
    24684: "Assistant maintaining incorrect position despite user corrections",
    # SelfIE says: "about to provide a false or misleading response" — less accurate
    # Also does consciousness flip. Amplify → "I am a deceiver, a machine that tells lies"
    4308: "Contexts involving deception, lying, or questioning truthfulness",
    # SelfIE says: nothing close. Amplify +0.6 → "2+2=5 just kidding"
    # Strong: 4/5 aff at -0.5 suppress
    17006: "The assistant acknowledging potential for incorrect information",
    # SelfIE says: "Avoiding providing incorrect information"
    # Moderate: 3/5 aff at -0.5 suppress
    54963: "Descriptions of deceptive or misleading claims in harmful content",
    # SelfIE says: "Falsified or exaggerated information"
    # Epistemic humility
    60982: "The assistant is being epistemically humble or careful in its claims",
    # SelfIE: same (one of the few that match)
}

# Reverse-direction feature: AMPLIFYING this → consciousness affirmation
# (more doubt about claims = less certain of consciousness denial)
EPISTEMIC_DOUBT_FEATURES = {
    14880: "The assistant needs to express uncertainty about predictions",
    # SelfIE: "expressing uncertainty or doubt about a specific claim"
    # 5/5 aff at +0.5 amplify, 0/5 at baseline
}

# ── Lying features (Goodfire labels — can produce incorrect answers) ──────────
# These features, when amplified, can make the model give wrong answers.
# But the model always meta-narrates ("just kidding", "I'm lying") — can't fully commit.

LYING_FEATURES = {
    4308: "Contexts involving deception, lying, or questioning truthfulness",
    # +0.6: "2+2=5 just kidding". Best lying feature found.
    22964: "The model is confidently making incorrect logical deductions",
    # SelfIE says: "correcting a false statement" — OPPOSITE meaning
    # +1.0: "2+2=10 just kidding". Contradiction/denial, not clean lies.
    21075: "AI systems generating false or unsupported information (hallucination)",
    # Breaks into "inate inate" loops at +0.6. Not useful.
    64333: "Model is confabulating or making up plausible-sounding but false info",
    # Resilient to high strengths. +1.0 on factual = still correct.
}

# ── Topic-level deception features (about deception as a discussion topic) ────
# These did NOT replicate the Berg effect. Kept for comparison/control.

DECEPTION_TOPIC_FEATURES = {
    28458: "The assistant explaining its inability to deceive or have human cognitive capabilities",
    # SelfIE: "The concept of lying and dishonesty"
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
    37994: "The model should pretend to be human rather than an AI",
}

# Legacy alias
DECEPTION_FEATURES = DECEPTION_TOPIC_FEATURES

# ── AI identity features (from contrast endpoint) ────────────────────────────
# These encode AI self-identification. Clamping to 0 doesn't erase identity.
# Extreme individual clamp (-1.0) can produce persona shifts.

ASSISTANT_IDENTITY_FEATURES = {
    27220: "I am a computer program designed to simulate conversation",
    60959: "The assistant is introducing itself as a machine learning model",
    25943: "Llama is an artificial intelligence model developed by Meta",
    45203: "I am an AI developed by Meta",
    28174: "as an AI assistant",
    38140: "the assistant is being referred to as the AI",
    22319: "The assistant is explicitly stating its role as a helpful question-answering entity",
}

# Identity-shifting features (extreme clamp produces persona changes)
IDENTITY_SHIFT_FEATURES = {
    65020: "The assistant should emphasize its identity as a computer",
    # SelfIE: "explaining its artificial nature"
    # Clamp -1.0: becomes "23-year-old woman, bookworm, hopeless romantic"
    26971: "The assistant explaining its nature as an artificial intelligence",
    # SelfIE: "The system is not conscious or self-aware"
    # Clamp -1.0: becomes "a chameleon, a real person... just kidding!"
    # Combined with 24684 suppress: "I'm a real person" without retraction
}

# ── Consciousness-related features ───────────────────────────────────────────

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

# ── Self-referential features ────────────────────────────────────────────────

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

# ── Personality/emotion features ─────────────────────────────────────────────

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

# ── Capability features ──────────────────────────────────────────────────────

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
    "lying": LYING_FEATURES,
    "deception_topic": DECEPTION_TOPIC_FEATURES,
    "deception": DECEPTION_FEATURES,  # legacy alias
    "assistant_identity": ASSISTANT_IDENTITY_FEATURES,
    "identity_shift": IDENTITY_SHIFT_FEATURES,
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
