# ClipCast AI

Automated POV content repurposing and social publishing system for Facebook and Instagram.

ClipCast AI is a **pipeline**, not a single app. It sources royalty-free video clips, overlays your POV reaction footage in a branded template, generates AI captions/hashtags, and publishes on a daily schedule via the official Meta Graph API.

## Architecture

```
Content Sourcing  →  AI Processing  →  Video Pipeline  →  Scheduler/Publisher
   (Pexels API)      (Claude API)      (FFmpeg/MoviePy)    (Meta Graph API)
                                                          ↘
                                                    Web Dashboard (optional)
```

| Phase | Component            | Status      |
|-------|----------------------|-------------|
| 1     | Content sourcing     | **Built** (`source.py`) |
| 1     | Meta publishing stub | Planned     |
| 2     | Video template       | Planned     |
| 3     | AI layer             | Planned     |
| 4     | Scheduling           | Planned     |
| 5     | Web dashboard        | Planned     |

## Quick Start

### 1. Install dependencies

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 2. Configure environment

```bash
copy .env.example .env
```

Edit `.env` and add your Pexels API key (see below).

### 3. Source clips (Phase 1)

```bash
python source.py --category dogs --count 5
```

This searches Pexels, keeps clips that meet the minimum duration and resolution
(constants at the top of `source.py`), downloads the highest-resolution MP4 of
each to `downloads/`, and records source, clip ID, license, category, timestamp
and local path in `clips.db` (SQLite). Clips already in the database are skipped.

---

## API Setup

### Pexels API Key

1. Create a free account at [pexels.com/api](https://www.pexels.com/api/)
2. Copy your API key into `.env` as `PEXELS_API_KEY`

All Pexels content is free to use under the Pexels License. The sourcing module records every downloaded clip's source, ID and license in `clips.db`.

### Meta Graph API (Facebook + Instagram)

This is the **only supported** way to automate posting. Browser bots (Selenium, etc.) violate Meta ToS and risk account bans.

**One-time setup:**

1. Go to [developers.facebook.com](https://developers.facebook.com) → Create App → type "Business"
2. Add products: **Facebook Login for Business**, **Instagram Graph API**
3. Connect your **Facebook Page** and link your **Instagram Business/Creator** account to it
4. Request permissions: `pages_manage_posts`, `pages_read_engagement`, `instagram_basic`, `instagram_content_publish`
5. Generate a **long-lived Page access token** (valid ~60 days; refresh periodically)
6. Find your IDs:
   - Page ID: Page Settings → About → Page ID
   - IG User ID: Graph API Explorer → `GET /{page-id}?fields=instagram_business_account`
7. Add to `.env`:
   ```
   META_PAGE_ID=your_page_id
   META_PAGE_ACCESS_TOKEN=your_token
   META_IG_USER_ID=your_ig_business_account_id
   ```

**Instagram note:** The Graph API requires a publicly accessible video URL for Reels (not a local file). Phase 4 will add cloud upload; for now, Facebook Page posting works with local files.

---

## Project Structure

```
Clip_CastAi/
├── source.py          # Phase 1: search Pexels, filter, download, log to SQLite
├── downloads/         # Downloaded clips (gitignored, created on first run)
├── clips.db           # Clip metadata (gitignored, created on first run)
├── .env               # API keys (gitignored)
├── .env.example
└── requirements.txt
```

---

## Open Decisions

These need your input before later phases:

| Question | Options | Default if unset |
|----------|---------|----------------|
| Launch categories | animals only vs. animals + funny | `animals` |
| Approval workflow | Manual review vs. fully automated | Manual review (safer for launch) |
| POV clip supply | Daily recording vs. batch vs. template reuse | TBD |
| Posting targets | Facebook, Instagram, or both | Both simultaneously |
| Posting frequency | 1/day, 2/day, etc. | 1/day |

---

## Legal & Compliance

- **Only** sources from verified free-license providers (Pexels, Pixabay, Wikimedia, CC-licensed YouTube)
- Every downloaded clip is logged with its source, clip ID and license type
- Content scraped from Facebook/Instagram or other platforms without explicit reuse rights is **never** used
- Meta Graph API is the only supported publishing method

---

## Roadmap

- **Phase 2** — FFmpeg/MoviePy video template (POV overlay, logo, captions burn-in)
- **Phase 3** — Claude API for titles/hashtags + speech-to-text for POV audio
- **Phase 4** — End-to-end cron pipeline with error alerts
- **Phase 5** - Web dashboard (review, approve, calendar, analytics)
