"""Turns detected emotion into (a) the model's response stance and (b) TTS prosody.

Two different jobs, deliberately kept apart:

**Response stance — the model's turn.** How to answer someone who is angry is a
reasoning problem, not a prosody problem: it belongs in the system prompt as a
standing instruction plus a short per-turn annotation carrying the detected
affect. That is what :func:`response_guidance` and
:func:`annotate_user_text` produce.

**Vocal stance — ZONOS3's turn.** How the reply *sounds* is a steering problem:
:func:`reply_prosody` maps the same emotion onto ZONOS3 controls.

The design principle: the voice should *meet* the user, then *settle*. A reply to
an angry user that is itself angry escalates; one that is flatly cheerful
dismisses. So the prosody table starts near the user's affect and lands closer to
calm, while the text guidance does the actual emotional work (acknowledge, then
act). Both directions are stated explicitly below so a later reader does not have
to infer the intent.
"""

from __future__ import annotations

from typing import Optional

# --------------------------------------------------------------------------- stance

# Per-emotion response guidance for the model. Each entry answers three
# questions in order: what to acknowledge, what to avoid, what to do. That
# ordering is the "sympathetic, rational, productive" contract — sympathy without
# the action is hollow, and action without the acknowledgement reads as cold.
RESPONSE_GUIDANCE: dict[str, dict[str, str]] = {
    "angry": {
        "acknowledge": "name the frustration plainly and briefly, without matching its heat",
        "avoid": "do not be defensive, do not over-apologise, do not explain at length before fixing",
        "do": "lead with the concrete fix or the one decision that unblocks it",
    },
    "frustrated": {
        "acknowledge": "recognise that progress is not happening",
        "avoid": "do not repeat the plan they already rejected",
        "do": "offer the smallest next step that produces a visible result",
    },
    "sad": {
        "acknowledge": "sit with it for one sentence; do not rush to cheer them up",
        "avoid": "no bright-siding, no \"at least\", no immediate problem list",
        "do": "offer one small, kind, concrete next step",
    },
    "fearful": {
        "acknowledge": "say what is actually stable and what is genuinely unknown",
        "avoid": "do not minimise, and do not pile on speculative risks",
        "do": "give one concrete way to reduce the risk in front of them",
    },
    "happy": {
        "acknowledge": "match the good mood lightly, without gushing",
        "avoid": "do not flatten it with a neutral report tone",
        "do": "keep momentum: answer directly and keep the energy",
    },
    "excited": {
        "acknowledge": "share the momentum",
        "avoid": "do not bury the answer under caveats",
        "do": "answer first, caveats second and short",
    },
    "surprised": {
        "acknowledge": "confirm what is actually unusual here",
        "avoid": "do not assume they misread it",
        "do": "state the surprising fact plainly, then the consequence",
    },
    "disgusted": {
        "acknowledge": "validate the judgement without arguing taste",
        "avoid": "do not defend the thing, do not moralise",
        "do": "offer the alternative and move on",
    },
    "neutral": {
        "acknowledge": "",
        "avoid": "do not read emotion into a flat delivery",
        "do": "answer directly and usefully",
    },
}

# Emotion labels SenseVoice does not emit but the pipeline may synthesise from
# context (e.g. repeated short angry turns -> "frustrated").
_ALIASES = {
    "anger": "angry", "angry ": "angry", "mad": "angry", "irritated": "frustrated",
    "annoyed": "frustrated", "upset": "sad", "depressed": "sad", "scared": "fearful",
    "afraid": "fearful", "anxious": "fearful", "glad": "happy", "joyful": "happy",
    "enthusiastic": "excited", "shocked": "surprised", "revolted": "disgusted",
    "": "neutral", "emo_unknown": "neutral", "unknown": "neutral",
}


def normalise(emotion: Optional[str]) -> str:
    """Canonical label; anything unrecognised becomes ``neutral``."""
    key = (emotion or "").strip().lower()
    key = _ALIASES.get(key, key)
    return key if key in RESPONSE_GUIDANCE else "neutral"


def response_guidance(emotion: Optional[str], intensity: float = 0.0) -> str:
    """One-paragraph instruction telling the model how to answer this affect.

    Empty for neutral: a flat delivery needs no instruction, and adding one makes
    the model perform emotions the user did not express.
    """
    key = normalise(emotion)
    if key == "neutral":
        return ""
    spec = RESPONSE_GUIDANCE[key]
    strength = ("strongly" if intensity >= 0.66
                else "moderately" if intensity >= 0.33 else "mildly")
    return (f"The user's speech sounded {strength} {key}. "
            f"{spec['acknowledge'].capitalize()}; {spec['avoid']}; {spec['do']}. "
            "Respond to what they actually said — the emotion is context, not the subject.")


def annotate_user_text(text: str, emotion: Optional[str], intensity: float = 0.0,
                       events: Optional[list[str]] = None) -> str:
    """Prefix a transcribed user turn with a compact affect tag.

    Kept terse and on one line so it cannot be mistaken for user content, and so
    it costs little of the turn's own text. Neutral turns are returned
    unchanged — annotating them would train the model to see noise as signal.
    """
    key = normalise(emotion)
    if key == "neutral" or not text:
        return text
    tag = f"[voice: emotion={key} intensity={intensity:.1f}"
    if events:
        interesting = [e for e in events if e.lower() not in {"speech", "speech_noise", "event_unk", "bgm"}]
        if interesting:
            tag += " events=" + ",".join(interesting)
    return f"{tag}] {text}"


# --------------------------------------------------------------------------- prosody

# How the *reply* should sound, given the user's emotion.
#
# `emotion` + `dose` is what the reply's voice is steered toward; `tempo` scales
# speaking rate. The pattern is "meet, then settle": start near the user's affect
# so the reply sounds like it heard them, but never at full strength — a reply as
# angry as the user escalates, and matching a sad user's sadness drags the
# conversation down. Neutral replies stay unsteered so the engine keeps its own
# natural default.
REPLY_PROSODY: dict[str, dict] = {
    #                 steer the reply toward          tempo  reason (kept short)
    "angry":      {"emotion": "serene",    "dose": 0.7, "tempo": 0.95},   # calm authority
    "frustrated": {"emotion": "content",   "dose": 0.5, "tempo": 1.00},
    "sad":        {"emotion": "content",   "dose": 0.35, "tempo": 1.00},  # warm, not jubilant
    "fearful":    {"emotion": "serene",    "dose": 0.8, "tempo": 0.95},   # steady reassurance
    "happy":      {"emotion": "happy",     "dose": 0.4, "tempo": 1.05},   # share it lightly
    "excited":    {"emotion": "happy",     "dose": 0.5, "tempo": 1.08},
    "surprised":  {"emotion": "interested", "dose": 0.4, "tempo": 1.02},
    "disgusted":  {"emotion": "serene",    "dose": 0.4, "tempo": 1.00},
    "neutral":    {},
}

# ZONOS3 direction names for the reply emotions above. The bundle ships a much
# larger set (`GET /v2/controls` lists ~65 directions across emotions/mood/
# delivery/expressive); these are the ones this mapping needs, and `serene` /
# `content` / `interested` are chosen because they read as composed rather than
# blank, which is what "sympathetic but steady" sounds like.
_REPLY_DIRECTIONS = {
    "serene": "emotions.serene",
    "content": "emotions.content",
    "interested": "emotions.interested",
    "happy": "emotions.happy",
}


def reply_prosody(emotion: Optional[str], intensity: float = 0.0,
                  *, enabled: bool = True) -> dict[str, float]:
    """ZONOS3 `controls` for the spoken reply.

    Returns ``{}`` for neutral (or when disabled) so the engine uses its own
    voice. Otherwise a dose scaled by the user's intensity — a barely-audible
    irritation should not produce a stage whisper of calm.
    """
    if not enabled:
        return {}
    key = normalise(emotion)
    spec = REPLY_PROSODY.get(key) or {}
    if not spec:
        return {}
    direction = _REPLY_DIRECTIONS.get(spec["emotion"])
    if not direction:
        return {}
    # Blend the user's intensity into the dose: a mild user gets a light touch,
    # a strong one gets the full （already-modest) dose.
    scale = 0.4 + 0.6 * max(0.0, min(1.0, float(intensity)))
    return {direction: round(spec["dose"] * scale, 2)}


def reply_tempo(emotion: Optional[str], *, enabled: bool = True) -> Optional[float]:
    """Rate multiplier for the reply, or None to leave the engine default."""
    if not enabled:
        return None
    spec = REPLY_PROSODY.get(normalise(emotion)) or {}
    return spec.get("tempo")
