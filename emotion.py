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
        "acknowledge": "recognise that this has gone on across several exchanges, not just this one",
        "avoid": "do not repeat any approach that has already been tried and did not work, and do not ask them to re-explain",
        "do": "say plainly what you are doing differently now, or admit the current approach is wrong and name the alternative",
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
        "acknowledge": "match the momentum — they are already enthusiastic, so do not re-sell it",
        "avoid": "do not bury the answer under caveats, and do not slow down to reassure them",
        "do": "answer first and make it actionable now; keep caveats to one clause at the end",
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


def _as_intensity(intensity: Optional[float]) -> float:
    """Coerce a missing intensity to 0.0.

    `intensity` is Optional because the backend may not expose CTC posteriors:
    the sherpa path reports ``intensity=None`` on purpose (see
    ``sensevoice._SherpaBackend``), and the ONNX path leaves it None whenever the
    winning token is neutral/unknown. Every consumer below compares it to a
    threshold, so a bare None was a TypeError waiting for the day the engine
    setting changed -- not a hypothetical, this install can flip to sherpa with
    one config line.
    """
    if intensity is None:
        return 0.0
    try:
        value = float(intensity)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, value))


def response_guidance(emotion: Optional[str], intensity: float = 0.0) -> str:
    """One-paragraph instruction telling the model how to answer this affect.

    Empty for neutral: a flat delivery needs no instruction, and adding one makes
    the model perform emotions the user did not express.
    """
    key = normalise(emotion)
    if key == "neutral":
        return ""
    spec = RESPONSE_GUIDANCE[key]
    level = _as_intensity(intensity)
    strength = ("strongly" if level >= 0.66
                else "moderately" if level >= 0.33 else "mildly")
    return (f"The user's speech sounded {strength} {key}. "
            f"{spec['acknowledge'].capitalize()}; {spec['avoid']}; {spec['do']}. "
            "Respond to what they actually said — the emotion is context, not the subject.")


def annotate_user_text(text: str, emotion: Optional[str], intensity: float = 0.0,
                       events: Optional[list[str]] = None) -> str:
    """Prefix a transcribed user turn with a compact affect tag.

    Kept terse and on one line so it cannot be mistaken for user content, and so
    it costs little of the turn's own text. Neutral turns are returned unchanged —
    annotating them would train the model to see noise as signal — with one
    exception: if the *events* carry real affect evidence (crying, laughter), that
    is dropped nowhere else and is exactly what a neutral label hides. Voice
    activity and noise events stay silent, since they describe the recording
    rather than the speaker.
    """
    key = normalise(emotion)
    interesting = [e for e in (events or [])
                   if e.lower() not in {"speech", "speech_noise", "event_unk", "bgm"}]
    if not text:
        return text
    if key == "neutral":
        # No emotion to report, but an affect-bearing event is still evidence.
        if not interesting:
            return text
        return f"[voice: emotion=neutral events={','.join(interesting)}] {text}"
    tag = f"[voice: emotion={key} intensity={intensity:.1f}"
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
# `secondary` is optional: a second direction at a weaker dose, for the cases
# where one direction cannot distinguish two affects. Only used where a single
# steer would be an exact duplicate of another emotion's.
REPLY_PROSODY: dict[str, dict] = {
    #                 primary steer              dose   tempo   secondary
    #
    # Two rules govern every row. First, MATCH AROUSAL, FLIP VALENCE: answering
    # anger with "serene" alone drops both the energy and the feeling at once,
    # which reads as deflating; serene + proud keeps some spine and only flips
    # the valence, which is what calm authority actually sounds like.
    #
    # Second, the counter-affect must CONTRADICT the user's, not agree with it.
    # "disappointed" as the counter to frustration agreed that things were
    # hopeless -- the opposite of useful -- so it moved to angry's own counter.
    "angry":      {"emotion": "serene",     "dose": 0.70, "tempo": 0.95,
                   "secondary": {"direction": "proud", "dose": 0.20}},   # calm authority
    # Frustration wants MOMENTUM, not hush. All three negative rows used to slow
    # down; this one speeds up, because "let us go" is the useful answer to
    # "this still is not working".
    "frustrated": {"emotion": "serene",     "dose": 0.70, "tempo": 1.06,
                   "secondary": {"direction": "proud", "dose": 0.22}},   # resolve, not regret
    # "content" beside a sad person reads as satisfied, so the sympathy is
    # carried by melody instead: a flattened pitch range is audible as quiet
    # warmth without performing an upbeat mood.
    "sad":        {"emotion": "content",    "dose": 0.32, "tempo": 1.00,
                   "raw": {"pitch.pitch_range": -1.0}},                  # warm, subdued
    "fearful":    {"emotion": "serene",     "dose": 0.55, "tempo": 0.92,
                   "secondary": {"direction": "relieved", "dose": 0.30}},  # steady reassurance
    "happy":      {"emotion": "happy",      "dose": 0.40, "tempo": 1.05},  # share it lightly
    "excited":    {"emotion": "happy",      "dose": 0.50, "tempo": 1.08,
                   "secondary": {"direction": "ecstatic", "dose": 0.25}},  # genuine momentum
    # "interested" only says "tell me more", which is the USER's job. "realizing"
    # arrives at understanding, which is the assistant's.
    "surprised":  {"emotion": "realizing",  "dose": 0.35, "tempo": 1.02,
                   "secondary": {"direction": "interested", "dose": 0.15}},
    "disgusted":  {"emotion": "disgusted",  "dose": 0.35, "tempo": 1.00,
                   "secondary": {"direction": "content", "dose": 0.20}},  # distaste, staying civil
    "neutral":    {},
}

# ZONOS3 direction names for the reply emotions above. The bundle ships a much
# larger set (`GET /v2/controls` lists ~65 directions across emotions/mood/
# delivery/expressive); these are the ones this mapping needs, and `serene` /
# `content` / `interested` are chosen because they read as composed rather than
# blank, which is what "sympathetic but steady" sounds like.
# Every one of these is accepted by the served bundle and measurably changes the
# audio. The bundle advertises 23 emotion directions; only six are exposed as
# mood sliders WITH a published measurement, which is not the same as "only six
# work" -- the rest resolve fine when sent directly. Verified by rendering each
# at a fixed seed and comparing bytes against an unsteered take.
_REPLY_DIRECTIONS = {
    "serene": "emotions.serene",
    "content": "emotions.content",
    "interested": "emotions.interested",
    "happy": "emotions.happy",
    "proud": "emotions.proud",
    "disappointed": "emotions.disappointed",
    "relieved": "emotions.relieved",
    "ecstatic": "emotions.ecstatic",
    "disgusted": "emotions.disgusted",
    "realizing": "emotions.realizing",
}


def reply_prosody(emotion: Optional[str], intensity: float = 0.0,
                  *, enabled: bool = True, allow_raw: bool = False,
                  previous: Optional[str] = None, turn: int = 0,
                  seed: Optional[int] = None) -> dict[str, float]:
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
    # Which ENGINE direction to use is now chosen from the affect's pool rather
    # than fixed by the table, so repeated affects vary and the 14 directions the
    # old mapping ignored become reachable. turn=0 (the default) keeps the
    # reasoned table entry, so callers that pass no history are unchanged.
    chosen, variant_scale, _why = choose_direction(
        key, previous=previous, turn=turn, intensity=intensity, seed=seed)
    primary = chosen or spec["emotion"]
    direction = _REPLY_DIRECTIONS.get(primary)
    if not direction:
        return {}
    # Blend the user's intensity into the dose: a mild user gets a light touch,
    # a strong one gets the full (already-modest) dose.
    scale = (0.4 + 0.6 * _as_intensity(intensity)) * variant_scale
    controls = {direction: round(spec["dose"] * scale, 2)}

    # A second, weaker direction where one could not distinguish two affects
    # (see the table). Sent at the same scaled dose so the pair stays in
    # proportion at every intensity.
    secondary = spec.get("secondary") or {}
    second = _REPLY_DIRECTIONS.get(secondary.get("direction", ""))
    if second:
        controls[second] = round(secondary.get("dose", 0.0) * scale, 2)

    # A `raw` steer moves the speaker embedding rather than the hidden space,
    # which the server refuses (400) unless the request asked for
    # speaker_mode=embedding. Only bundled-voice installations run that way, so
    # the caller decides whether raw controls may be appended -- sending one on a
    # cloning path would take the whole utterance down.
    if allow_raw:
        for name, dose in (spec.get("raw") or {}).items():
            controls[name] = round(float(dose) * scale, 3)

    # Tempo was carried in this table but never emitted -- reply_tempo() existed
    # and had no caller, so every tempo value here was dead. `delivery.tempo` is
    # the direction the engine actually honours (the same one `speed` maps onto),
    # so fold it in rather than leave a column of numbers that does nothing.
    tempo = spec.get("tempo")
    if tempo and abs(float(tempo) - 1.0) > 1e-3:
        controls["delivery.tempo"] = round((float(tempo) - 1.0) * 4.0, 3)
    return controls


def reply_tempo(emotion: Optional[str], *, enabled: bool = True) -> Optional[float]:
    """Rate multiplier for the reply, or None to leave the engine default.

    Kept for callers that want the number itself; :func:`reply_prosody` already
    folds it into ``delivery.tempo``, so applying both would double the effect.
    """
    if not enabled:
        return None
    spec = REPLY_PROSODY.get(normalise(emotion)) or {}
    return spec.get("tempo")


# SenseVoice does not only label emotion — it reports audio *events*, and two of
# them are direct affect evidence the emotion tag often misses. A crying speaker
# frequently comes back `<|SAD|>` or even `<|NEUTRAL|>` because the vocal
# realisation is flat, while `<|Cry|>` is unambiguous about what is happening;
# `<|Laughter|>` likewise confirms a happy reading rather than a merely raised
# voice. FunASR documents these tags as per-segment predictions, so they are
# evidence about this utterance, not decoration.
_EVENT_AFFECT: dict[str, dict] = {
    # event     -> affect it argues for, and how strongly
    "cry":       {"emotion": "sad",    "weight": 0.85},
    "laughter":  {"emotion": "happy",  "weight": 0.55},
}


# ------------------------------------------------------------------- escalation
#
# SenseVoice emits seven labels; the guidance/prosody tables carry nine. The two
# extras -- "frustrated" and "excited" -- are distinctions the model cannot make
# (it folds irritation into ANGRY and enthusiasm into HAPPY) but which the
# *conversation* can: a user who is angry about something that keeps not working
# needs a different answer from one who is angry about a single shock, and a
# user who is excited wants momentum rather than a report.
#
# Without this, both rows were unreachable dead code AND, worse, every sustained
# irritation was answered with the single-incident "lead with the concrete fix"
# stance -- which is exactly the advice that had already failed them.
#
# The signal is temporal: the same affect recurring across consecutive turns.
# One angry turn is an event; three is a pattern.
_ESCALATION: dict[str, dict] = {
    # from      -> to            when this many consecutive turns agree
    "angry":     {"to": "frustrated", "turns": 2},
    "happy":     {"to": "excited",    "turns": 2},
}

# A pattern needs a strong signal, not a shrug: repeated mild readings are noise.
_ESCALATION_MIN_INTENSITY = 0.33

# How long a run may sit between turns and still count as "the same mood". Beyond
# this the user has moved on and the run restarts.
RUN_WINDOW_SECONDS = 180.0


class AffectTracker:
    """Remembers the last few affects so a run of them can escalate.

    Deliberately not part of :mod:`emotion`'s pure functions: this is the only
    piece with state, and keeping it separate means the tables above stay
    testable without a clock.
    """

    def __init__(self, window_seconds: float = RUN_WINDOW_SECONDS) -> None:
        self._window = float(window_seconds)
        self._label: Optional[str] = None
        self._count = 0
        self._at = 0.0

    def observe(self, emotion: Optional[str], intensity: Optional[float] = None,
                *, now: Optional[float] = None) -> tuple[str, int]:
        """Record one turn's affect; return ``(effective_emotion, run_length)``.

        The effective emotion is the base label until a run is long enough to
        escalate, at which point it becomes the escalated one. Intensity is not
        altered -- the run says nothing about how loud the last turn was.
        """
        import time as _time

        key = normalise(emotion)
        stamp = _time.time() if now is None else float(now)
        level = _as_intensity(intensity)

        if key == "neutral":
            # A flat turn ends any run: silence is not evidence of a pattern.
            self._label, self._count, self._at = None, 0, stamp
            return key, 0
        if (key == self._label and self._count
                and (stamp - self._at) <= self._window):
            self._count += 1
        else:
            self._count = 1
        self._label = key
        self._at = stamp

        spec = _ESCALATION.get(key)
        if spec and self._count >= spec["turns"] and level >= _ESCALATION_MIN_INTENSITY:
            return spec["to"], self._count
        return key, self._count

    def snapshot(self) -> dict:
        import time as _time

        return {"label": self._label, "count": self._count,
                "age_seconds": round(_time.time() - self._at, 1) if self._at else None}


def coverage_gaps(emotions) -> dict[str, list[str]]:
    """Report which emotions lack a usable steer in either table.

    The requirement is that every emotion the pipeline can *produce* has an
    equivalent steer in both directions:

      guidance  what the model should SAY  (RESPONSE_GUIDANCE)
      prosody   how the reply should SOUND (REPLY_PROSODY)
      directs   ...and every direction it names must exist on the engine

    This is a function rather than a test so it can also be asserted at import
    time by the plugin and called from the dashboard, and it returns gaps instead
    of raising so a partial mapping still speaks.
    """
    gaps: dict[str, list[str]] = {"guidance": [], "prosody": [], "directions": [],
                                  "collapsed": []}
    seen: dict[tuple, list[str]] = {}
    for name in emotions:
        if name not in RESPONSE_GUIDANCE:
            gaps["guidance"].append(name)
        spec = REPLY_PROSODY.get(name)
        if spec is None:
            gaps["prosody"].append(name)
            continue
        if name == "neutral":
            continue            # neutral deliberately sends no controls
        primary = spec.get("emotion")
        if not primary or primary not in _REPLY_DIRECTIONS:
            gaps["directions"].append(f"{name}:{primary}")
        secondary = (spec.get("secondary") or {}).get("direction")
        if secondary and secondary not in _REPLY_DIRECTIONS:
            gaps["directions"].append(f"{name}:{secondary}")
        # `raw` names a direction directly (pitch.pitch_range), so it bypasses
        # _REPLY_DIRECTIONS by design -- verify it is namespaced, not arbitrary.
        for raw_name in (spec.get("raw") or {}):
            if "." not in str(raw_name):
                gaps["directions"].append(f"{name}:raw:{raw_name}")
        # Two emotions resolving to the SAME directions at the SAME doses are one
        # voice wearing two labels.
        key = (primary, round(float(spec.get("dose", 0)), 2),
               secondary, round(float((spec.get("secondary") or {}).get("dose", 0)), 2),
               tuple(sorted((spec.get("raw") or {}).items())))
        seen.setdefault(key, []).append(name)
    for key, names in seen.items():
        if len(names) > 1:
            gaps["collapsed"].append(f"{key[0]}: {'/'.join(names)}")
    return gaps


# --------------------------------------------------------------- emotion pools
#
# SenseVoice emits 7 labels; the engine accepts 23 emotion directions. The old
# table threw that away by mapping each label to ONE fixed direction, so a
# recurring affect always sounded identical and 14 usable directions sat unused.
#
# Each pool is now a set of plausible replies for the same detected affect. The
# first entry is the default (the reasoned "meet, flip valence" stance); the rest
# are alternatives chosen when history makes them a better fit.
#
# `avoid` is an explicit anti-pattern, not just documentation: some directions
# agree with the user's affect and make things worse, so they must never be
# reached by rotation no matter how many turns pass.
_POOLS: dict[str, dict] = {
    "angry": {
        "directions": ["serene", "proud", "content", "interested"],
        "avoid": ["angry", "distressed", "pained"],
        "reason": "steady, never matching the heat",
    },
    "frustrated": {
        "directions": ["serene", "proud", "realizing", "interested"],
        "avoid": ["disappointed", "angry", "sad"],
        "reason": "resolve and forward motion, never regret",
    },
    "sad": {
        "directions": ["content", "relieved", "serene", "interested"],
        "avoid": ["happy", "ecstatic", "adoring"],
        "reason": "warm company, never bright-siding",
    },
    "fearful": {
        "directions": ["serene", "relieved", "content", "proud"],
        "avoid": ["scared", "distressed", "fearful"],
        "reason": "reassurance, never amplifying the threat",
    },
    "happy": {
        "directions": ["happy", "content", "adoring", "interested"],
        "avoid": ["ecstatic", "desirous"],
        "reason": "share it lightly, never gush",
    },
    "excited": {
        "directions": ["happy", "ecstatic", "interested", "proud"],
        "avoid": ["adoring"],
        "reason": "match momentum",
    },
    "surprised": {
        "directions": ["realizing", "interested", "confused", "content"],
        "avoid": ["surprised", "embarrassed"],
        "reason": "arrive at understanding rather than mirror the shock",
    },
    "disgusted": {
        "directions": ["disgusted", "content", "interested", "serene"],
        "avoid": ["pained", "distressed"],
        "reason": "acknowledge the distaste, stay civil",
    },
    "frustrated_after_anger": {
        "directions": ["proud", "serene", "realizing", "content"],
        "avoid": ["disappointed", "angry"],
        "reason": "anger that has cooled needs resolve, not more empathy",
    },
    "sad_after_fear": {
        "directions": ["relieved", "serene", "content", "proud"],
        "avoid": ["sad", "distressed"],
        "reason": "fear that has resolved should sound like relief",
    },
}

# How far the chosen direction may sit from its pool's default before rotation
# is allowed to pick it. Higher = more variance between turns.
POOL_TEMPERATURE = 0.55


def _pool_for(emotion: str, previous: Optional[str]) -> dict:
    """Pick the pool for an affect, refined by what was heard last turn.

    History matters because two different sequences can share a label: a user who
    was ANGRY and is now FRUSTRATED is cooling down, while one who was SAD and is
    now FRUSTRATED is heating up. Those want different replies from the same
    detected word.
    """
    key = normalise(emotion)
    prev = normalise(previous) if previous else None
    special = f"{key}_after_{prev}" if prev else None
    if special and special in _POOLS:
        return _POOLS[special] | {"id": special}
    return (_POOLS.get(key) or {}) | {"id": key}


def choose_direction(emotion: Optional[str], *, previous: Optional[str] = None,
                     turn: int = 0, intensity: float = 0.0,
                     seed: Optional[int] = None) -> tuple[str, float, str]:
    """Choose which engine emotion direction this reply should use.

    Returns ``(direction, dose_scale, why)``.

    Deterministic on ``(emotion, previous, turn, intensity)`` so a given
    conversation replays identically -- an emotion that changes on every request
    would make the voice feel unstable rather than alive. History does the
    varying: consecutive turns with the same affect walk through the pool, so
    three frustrated turns sound like three related but distinct replies.

    The first turn of any affect always uses the pool default, which is the stance
    the tables were reasoned out for. Rotation only advertises alternatives.
    """
    key = normalise(emotion)
    pool = _pool_for(key, previous)
    options = [d for d in pool.get("directions", []) if d not in (pool.get("avoid") or [])]
    if not options:
        return "", 1.0, "no pool"

    # Turn 0 keeps the reasoned default: the first impression should be the
    # intended one, not a random variant.
    if turn <= 0:
        return options[0], 1.0, pool.get("reason", "")

    # WALK the pool by turn index, offset by history. Hashing turn into the index
    # was wrong: hash() clustered onto two values for adjacent turn numbers and
    # the rotation stuck on one alternative instead of cycling.
    #
    # History shifts the phase, so the same affect reached from a different
    # previous affect starts at a different point in the pool -- that is the whole
    # point of passing `previous`, and without the offset it had no effect at all.
    phase = 0
    prev = normalise(previous) if previous else ""
    if prev and prev != key:
        phase = 1 + (abs(hash(prev)) % max(1, len(options) - 1))
    base = (turn - 1) + phase
    if seed is not None:
        base += abs(hash(seed)) % max(1, len(options))

    # Deeper intensity narrows toward the default; mild intensity wanders more,
    # because a barely-audible affect does not justify a strong stance change.
    #
    # The floor is 2, not 1: at high intensity the previous formula rounded to a
    # spread of one, which silently disabled rotation entirely -- every turn
    # returned options[0] and the pools looked broken. Emphasis and variety are
    # different knobs and intensity should not switch variety off.
    level = _as_intensity(intensity)
    span = 1 + int(round(len(options) * (1.0 - level)))
    spread = max(2, min(span, len(options))) if len(options) > 1 else 1
    idx = base % spread
    return options[idx], 0.85, pool.get("reason", "")


def infer_affect(emotion: Optional[str], events: Optional[list[str]] = None,
                 intensity: float = 0.0) -> tuple[str, float, str]:
    """Reconcile the emotion tag with the audio events.

    Returns ``(emotion, intensity, why)``. The event only *upgrades* an affect the
    emotion tag left ambiguous:
      * a neutral or unknown reading is replaced by the event's affect — this is
        the common case, since the label is per-segment and a flat read of a
        crying speaker scores neutral;
      * a reading that already agrees keeps its intensity, nudged up;
      * a reading that explicitly disagrees is left alone (a laughing-but-angry
        speaker is real), only the reason is reported.
    """
    key = normalise(emotion)
    if not events:
        return key, float(intensity or 0.0), "emotion only"
    found = None
    for raw in events:
        hit = _EVENT_AFFECT.get(str(raw or "").strip().lower())
        if hit:
            found = hit
            break
    if found is None:
        return key, float(intensity or 0.0), "emotion only"
    target, weight = found["emotion"], found["weight"]
    if key in ("neutral", "", "unknown"):
        # No opinion from the label: the event is the only evidence there is.
        return target, max(float(intensity or 0.0), weight), "event overrode an unnamed affect"
    if key == target:
        return key, min(1.0, max(float(intensity or 0.0), weight)), "event confirmed the label"
    return key, float(intensity or 0.0), f"label and event disagree ({key} vs {target})"
