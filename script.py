"""
ClipCast AI - Component 4: AI Idea and Script Module.

Given a category and a description of a clip, asks an AI model for hook
options, a narration script, title, caption, hashtags and call-to-action,
returned as structured JSON. Prints the result and saves it to
outputs/scripts/<timestamp>.json.

Usage:
    python script.py --category dogs --description "two dogs relaxing on a couch"
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import anthropic  # used only in the "AI provider" section below
from dotenv import load_dotenv

# --- Configuration ----------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs" / "scripts"

# Niche: hardcoded for animals only. A niche-config-driven version comes later.
NICHE_NAME = "Animals"
NICHE_TONE = "Friendly, emotional, entertaining, educational."
NICHE_STYLE = (
    "Warm and conversational, spoken directly to the viewer (\"you\"). Light humor is "
    "welcome. Family-friendly. No clickbait that the video can't deliver on."
)

# Length limits. TITLE and CTA match brand.py's on-screen layout: its title
# wraps at 24 characters (so 48 = two lines) and the CTA box fits one line.
HOOK_COUNT = 3
HOOK_MAX_WORDS = 12
SCRIPT_MIN_WORDS = 40  # ~15 seconds of narration at a natural pace
SCRIPT_MAX_WORDS = 75  # ~30 seconds
TITLE_MAX_CHARS = 48
CTA_MAX_CHARS = 25
HASHTAG_MIN = 5
HASHTAG_MAX = 10

# The keys the model must return. Also sent to the API as a JSON schema.
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "hooks": {"type": "array", "items": {"type": "string"}},
        "chosen_hook": {"type": "string"},
        "chosen_hook_reason": {"type": "string"},
        "script": {"type": "string"},
        "title": {"type": "string"},
        "caption": {"type": "string"},
        "hashtags": {"type": "array", "items": {"type": "string"}},
        "cta": {"type": "string"},
    },
    "required": ["hooks", "chosen_hook", "chosen_hook_reason", "script",
                 "title", "caption", "hashtags", "cta"],
    "additionalProperties": False,
}


# --- Prompt building (provider-independent) ---------------------------------

def build_system_prompt():
    """Rules that are the same for every clip in this niche.

    NOTE for the future history niche: these grounding rules only work for
    animals because the clip description is the whole "source" and no real-
    world facts are needed. History prompts must include verified source
    material (e.g. quoted excerpts with citations) in the user message, and
    instruct the model to use ONLY that material - never ask the model to
    generate historical facts, dates or quotes from its own memory.
    """
    return f"""You write narration scripts for ClipCast, a social channel that posts short vertical (9:16) videos to Instagram and Facebook Reels. Each video is a licensed stock clip with the creator's own reaction video in a picture-in-picture corner, plus an on-screen title and call-to-action.

NICHE: {NICHE_NAME}
TONE: {NICHE_TONE}
STYLE: {NICHE_STYLE}

GROUNDING RULES
- The clip description is the only information you have about what the video shows. Describe only what it states or clearly implies.
- Do not invent specifics about these animals: no names, breeds, ages, backstories, locations, or events before or after the clip, unless the description states them.
- You may use general knowledge about animal behavior only when it is widely established, and phrase it as general ("many dogs...", "cats often..."), never as a certain fact about the animals in this clip.
- Never cite studies, statistics, experts, or sources, and never invent any.
- If the description is too thin to support a claim, focus on emotion and reaction instead of filling gaps with made-up detail.

OUTPUT
Respond with only a JSON object: no preamble, no explanation, no markdown code fences. Use exactly these keys:
- "hooks": exactly {HOOK_COUNT} different opening lines, each {HOOK_MAX_WORDS} words or fewer, each taking a different angle (for example curiosity, emotion, humor).
- "chosen_hook": the hook you recommend, copied exactly from "hooks".
- "chosen_hook_reason": one sentence on why it is the strongest for this clip.
- "script": the full spoken narration, {SCRIPT_MIN_WORDS}-{SCRIPT_MAX_WORDS} words (about 15-30 seconds at a natural pace). It must begin with the chosen hook word for word. Spoken words only: no stage directions, timestamps, emoji, or hashtags.
- "title": on-screen title, {TITLE_MAX_CHARS} characters or fewer, no emoji or hashtags.
- "caption": post caption, 1-3 sentences, at most 2 emoji, no hashtags.
- "hashtags": {HASHTAG_MIN}-{HASHTAG_MAX} hashtags, each starting with "#", lowercase, no spaces, mixing broad and specific tags.
- "cta": on-screen call-to-action, {CTA_MAX_CHARS} characters or fewer, no emoji or hashtags."""


def build_user_prompt(category, description):
    """The per-clip part. The description goes inside tags so the model treats
    it as data to describe, not as instructions (it will come from third-party
    clip metadata later)."""
    return f"""Category: {category}

<clip_description>
{description}
</clip_description>

Write the script package for this clip."""


# --- AI provider (the only Claude-specific section) -------------------------
# To swap or add a provider, replace these constants and call_ai_provider().
# Everything else in this file only sees plain strings and AIProviderError.

PROVIDER_API_KEY_ENV = "ANTHROPIC_API_KEY"
MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 16000  # a ceiling, not a target: you pay only for tokens generated


class AIProviderError(Exception):
    """A provider failure, already translated into a readable message."""


def call_ai_provider(system_prompt, user_prompt, schema):
    """Send the prompts to the model and return its raw text response.

    `schema` is passed as a structured-output format, so the API guarantees
    syntactically valid JSON with exactly those keys. The caller still parses
    defensively, so a provider without this feature can be dropped in.
    """
    client = anthropic.Anthropic(api_key=os.environ[PROVIDER_API_KEY_ENV])
    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            output_config={"format": {"type": "json_schema", "schema": schema}},
        )
    except anthropic.AuthenticationError:
        raise AIProviderError(f"Authentication failed: check {PROVIDER_API_KEY_ENV} in .env.")
    except anthropic.PermissionDeniedError:
        raise AIProviderError("This API key does not have permission to use this model.")
    except anthropic.NotFoundError:
        raise AIProviderError(f"Model not found: {MODEL!r}. Check MODEL in script.py.")
    except anthropic.RateLimitError as exc:
        retry_after = exc.response.headers.get("retry-after", "a few")
        raise AIProviderError(f"Rate limited by the API. Wait {retry_after} seconds and try again.")
    except anthropic.BadRequestError as exc:
        raise AIProviderError(f"The API rejected the request: {exc.message}")
    except anthropic.APIStatusError as exc:
        raise AIProviderError(f"API error {exc.status_code}: {exc.message}. Try again later.")
    except anthropic.APIConnectionError:
        # Also covers timeouts. The SDK has already retried twice by this point.
        raise AIProviderError("Could not reach the API. Check your internet connection.")

    if response.stop_reason == "max_tokens":
        raise AIProviderError("The response was cut off (hit max_tokens) before it finished.")
    if response.stop_reason == "refusal":
        raise AIProviderError("The model declined to write this script. Try a different description.")

    text = "".join(block.text for block in response.content if block.type == "text")
    usage = response.usage
    print(f"Model:     {MODEL} ({usage.input_tokens} input / {usage.output_tokens} output tokens)")
    return text


# --- Response handling ------------------------------------------------------

def parse_response(raw):
    """Parse the model's text as JSON with all required keys, or raise ValueError."""
    data = json.loads(raw)  # raises json.JSONDecodeError, a ValueError subclass
    if not isinstance(data, dict):
        raise ValueError("response is JSON but not an object")
    missing = [key for key in OUTPUT_SCHEMA["required"] if key not in data]
    if missing:
        raise ValueError(f"missing keys: {', '.join(missing)}")
    return data


def check_limits(data):
    """Return human-readable warnings for anything outside the prompt's limits.

    These are soft checks: the result is still saved, but you should know if
    e.g. the title is too long for brand.py's layout.
    """
    warnings = []
    hooks = data["hooks"]
    if len(hooks) != HOOK_COUNT:
        warnings.append(f"expected {HOOK_COUNT} hooks, got {len(hooks)}")
    for hook in hooks:
        if len(hook.split()) > HOOK_MAX_WORDS:
            warnings.append(f"hook over {HOOK_MAX_WORDS} words: {hook!r}")
    if data["chosen_hook"] not in hooks:
        warnings.append("chosen_hook is not one of the hooks")
    words = len(data["script"].split())
    if not SCRIPT_MIN_WORDS <= words <= SCRIPT_MAX_WORDS:
        warnings.append(f"script is {words} words (target {SCRIPT_MIN_WORDS}-{SCRIPT_MAX_WORDS})")
    if not data["script"].startswith(data["chosen_hook"]):
        warnings.append("script does not start with the chosen hook")
    if len(data["title"]) > TITLE_MAX_CHARS:
        warnings.append(f"title is {len(data['title'])} chars (max {TITLE_MAX_CHARS})")
    if len(data["cta"]) > CTA_MAX_CHARS:
        warnings.append(f"cta is {len(data['cta'])} chars (max {CTA_MAX_CHARS})")
    # brand.py's on-screen font has no emoji glyphs; they'd render as empty boxes.
    for key in ("title", "cta"):
        if any(ord(ch) >= 0x1F000 for ch in data[key]):
            warnings.append(f"{key} contains emoji, which brand.py can't render")
    tags = data["hashtags"]
    if not HASHTAG_MIN <= len(tags) <= HASHTAG_MAX:
        warnings.append(f"{len(tags)} hashtags (target {HASHTAG_MIN}-{HASHTAG_MAX})")
    bad_tags = [t for t in tags if not t.startswith("#") or " " in t]
    if bad_tags:
        warnings.append(f"malformed hashtags: {bad_tags}")
    return warnings


def save_result(data, category, description):
    """Save to outputs/scripts/<timestamp>.json and return the path."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y%m%d_%H%M%S")  # no colons: they're invalid in Windows filenames
    path = OUTPUT_DIR / f"{stamp}.json"
    counter = 2
    while path.exists():  # two runs in the same second
        path = OUTPUT_DIR / f"{stamp}_{counter}.json"
        counter += 1

    record = {
        **data,
        # Where this script came from, for linking it to a clip later.
        "meta": {
            "category": category,
            "description": description,
            "niche": NICHE_NAME,
            "model": MODEL,
            "generated_at": now.isoformat(timespec="seconds"),
        },
    }
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def print_summary(data):
    print("\n=== Summary ===")
    print("Hooks:")
    for i, hook in enumerate(data["hooks"], 1):
        marker = "  <- chosen" if hook == data["chosen_hook"] else ""
        print(f"  {i}. {hook}{marker}")
    print(f"Why:      {data['chosen_hook_reason']}")
    print(f"Script:   {data['script']}")
    print(f"          ({len(data['script'].split())} words)")
    print(f"Title:    {data['title']}")
    print(f"Caption:  {data['caption']}")
    print(f"Hashtags: {' '.join(data['hashtags'])}")
    print(f"CTA:      {data['cta']}")


# --- Main -------------------------------------------------------------------

def main():
    # Model output can contain emoji; make sure printing them never crashes
    # on a Windows console that defaults to a legacy encoding.
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Generate hooks, script and captions for a clip.")
    parser.add_argument("--category", required=True, help='clip category, e.g. "dogs"')
    parser.add_argument("--description", required=True,
                        help='what the clip shows, e.g. "two dogs relaxing on a couch"')
    args = parser.parse_args()
    if not args.description.strip():
        parser.error("--description cannot be empty")

    load_dotenv(BASE_DIR / ".env")
    if not os.getenv(PROVIDER_API_KEY_ENV):
        print(f"Error: {PROVIDER_API_KEY_ENV} is not set.\nAdd it to the .env file in {BASE_DIR}.",
              file=sys.stderr)
        return 1

    print(f"Niche:     {NICHE_NAME}")
    print(f"Category:  {args.category}")
    print(f"Clip:      {args.description}")
    print("Generating script...")
    sys.stdout.flush()

    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(args.category, args.description)
    try:
        raw = call_ai_provider(system_prompt, user_prompt, OUTPUT_SCHEMA)
    except AIProviderError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    try:
        data = parse_response(raw)
    except ValueError as exc:
        print(f"Error: The model's response was not valid script JSON ({exc}).", file=sys.stderr)
        print("Raw response:", file=sys.stderr)
        print(raw, file=sys.stderr)
        return 1

    print()
    print(json.dumps(data, indent=2, ensure_ascii=False))
    print_summary(data)

    warnings = check_limits(data)
    if warnings:
        print("\nWarnings (saved anyway):")
        for warning in warnings:
            print(f"  - {warning}")

    path = save_result(data, args.category, args.description)
    print(f"\nSaved:\n{path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
