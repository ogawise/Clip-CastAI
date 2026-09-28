"""
Project database for the ClipCast pipeline.

Tracks every video project from idea to publication in pipeline.db (SQLite).
This is separate from clips.db, which belongs to source.py and is never opened
here. Other components import the functions below; the CLI is for testing.

Usage:
    python db.py init
    python db.py create --niche history --topic "How the Berlin Airlift began"
    python db.py list [--status research]
    python db.py show 1
    python db.py status 1 research [--note "pack generated"]
    python db.py approve 1 --note "sources and rights checked by me"
    python db.py asset 1 --license "CC BY 4.0" --commercial yes --modification yes --note "checked"

Set the PIPELINE_DB environment variable to use a different database file.
"""

import argparse
import os
import sqlite3
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = Path(os.environ.get("PIPELINE_DB") or BASE_DIR / "pipeline.db")

WORKFLOW = [
    "idea",
    "research",
    "script_draft",
    "fact_and_rights_review",
    "media_selected",
    "rendered",
    "needs_approval",
    "approved",
    "scheduled",
    "published",
]
REJECTED = "rejected"
STATUSES = WORKFLOW + [REJECTED]
# Any stage up to and including "approved" can be rejected; "rejected" is final.
REJECTABLE = set(WORKFLOW[: WORKFLOW.index("approved") + 1])
# Niches where requires_human_approval is always on and cannot be turned off.
APPROVAL_REQUIRED_NICHES = {"history"}


# --- Errors -----------------------------------------------------------------

class PipelineError(Exception):
    """Base class for every error this module raises on purpose."""


class ProjectNotFound(PipelineError):
    pass


class InvalidTransition(PipelineError):
    pass


class RightsCheckFailed(PipelineError):
    pass


# --- Schema -----------------------------------------------------------------

_STATUS_LIST = ", ".join(f"'{s}'" for s in STATUSES)
_HISTORY_NICHES = ", ".join(f"'{n}'" for n in sorted(APPROVAL_REQUIRED_NICHES))

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS projects (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    niche                   TEXT NOT NULL,
    topic                   TEXT NOT NULL,
    status                  TEXT NOT NULL DEFAULT 'idea' CHECK (status IN ({_STATUS_LIST})),
    requires_human_approval INTEGER NOT NULL CHECK (requires_human_approval IN (0, 1)),
    created_at              TEXT NOT NULL,
    updated_at              TEXT NOT NULL,
    notes                   TEXT,
    CHECK (niche NOT IN ({_HISTORY_NICHES}) OR requires_human_approval = 1)
);

CREATE TABLE IF NOT EXISTS research_packs (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL REFERENCES projects(id),
    json_path  TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS scripts (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL REFERENCES projects(id),
    json_path  TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL
);

-- Rights record per asset. NULL in license or a yes/no column means unknown.
CREATE TABLE IF NOT EXISTS assets (
    id                     INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id             INTEGER NOT NULL REFERENCES projects(id),
    asset_id               TEXT NOT NULL,
    source_platform        TEXT NOT NULL,
    source_url             TEXT NOT NULL,
    creator                TEXT,
    license                TEXT,
    date_retrieved         TEXT NOT NULL,
    attribution_required   INTEGER CHECK (attribution_required IN (0, 1)),
    commercial_use_allowed INTEGER CHECK (commercial_use_allowed IN (0, 1)),
    modification_allowed   INTEGER CHECK (modification_allowed IN (0, 1)),
    local_path             TEXT,
    notes                  TEXT,
    created_at             TEXT NOT NULL,
    UNIQUE (project_id, source_platform, asset_id)
);

-- Append-only: one row per changed asset field. NULL value means unknown.
CREATE TABLE IF NOT EXISTS asset_history (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    asset_row_id INTEGER NOT NULL REFERENCES assets(id),
    field        TEXT NOT NULL,
    old_value    TEXT,
    new_value    TEXT,
    changed_at   TEXT NOT NULL,
    note         TEXT
);

CREATE TABLE IF NOT EXISTS renders (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL REFERENCES projects(id),
    video_path TEXT NOT NULL,
    created_at TEXT NOT NULL
);

-- Append-only: one row per status change, never updated or deleted.
CREATE TABLE IF NOT EXISTS status_history (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL REFERENCES projects(id),
    old_status TEXT,
    new_status TEXT NOT NULL,
    changed_at TEXT NOT NULL,
    note       TEXT
);
"""


def connect(db_path=None):
    """Open the pipeline database, creating any missing tables."""
    conn = sqlite3.connect(db_path or DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    return conn


@contextmanager
def _db():
    """One connection per call: commit on success, roll back on any error."""
    conn = connect()
    try:
        with conn:
            yield conn
    finally:
        conn.close()


# --- Helpers ----------------------------------------------------------------

def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _store_path(path, must_exist=True):
    """Store paths inside the project relative to it, with forward slashes."""
    full = Path(path).resolve()
    if must_exist and not full.is_file():
        raise PipelineError(f"file not found: {path}")
    try:
        return full.relative_to(BASE_DIR).as_posix()
    except ValueError:
        return full.as_posix()


def _flag(value, name):
    """True/False/None -> 1/0/NULL. None means unknown."""
    if value is None:
        return None
    if value in (True, False, 0, 1):
        return int(value)
    raise PipelineError(f"{name} must be True, False or None (unknown), got {value!r}")


def _yes_no(value):
    return {1: "yes", 0: "no"}.get(value, "unknown")


def _license_unknown(license_text):
    text = (license_text or "").strip().lower()
    return text in ("", "unknown")


def _clean_license(license_text):
    """Blank or 'unknown' is stored as NULL."""
    return None if _license_unknown(license_text) else license_text.strip()


# Asset columns update_asset() may change. The identity columns (project_id,
# asset_id, source_platform) are fixed; record a new asset instead.
ASSET_EDITABLE = {
    "source_url",
    "creator",
    "license",
    "date_retrieved",
    "attribution_required",
    "commercial_use_allowed",
    "modification_allowed",
    "local_path",
    "notes",
}
ASSET_FLAGS = {"attribution_required", "commercial_use_allowed", "modification_allowed"}


def _clean_asset_field(name, value):
    if name in ASSET_FLAGS:
        return _flag(value, name)
    if name == "license":
        return _clean_license(value)
    if name == "local_path":
        return _store_path(value, must_exist=False) if value else None
    value = (value or "").strip() or None
    if value is None and name in ("source_url", "date_retrieved"):
        raise PipelineError(f"{name} cannot be empty")
    return value


def _history_value(name, value):
    """How a value is written to asset_history: yes/no for flags, NULL for unknown."""
    if value is None:
        return None
    return _yes_no(value) if name in ASSET_FLAGS else str(value)


def _get_row(conn, project_id):
    row = conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
    if row is None:
        raise ProjectNotFound(f"no project with id {project_id}")
    return row


def allowed_next(status):
    """Statuses a project in `status` may move to (including 'approved')."""
    allowed = []
    if status in WORKFLOW[:-1]:
        allowed.append(WORKFLOW[WORKFLOW.index(status) + 1])
    if status in REJECTABLE:
        allowed.append(REJECTED)
    return allowed


def _rights_problems(conn, project_id):
    """One line per linked asset that isn't cleared for rendering."""
    problems = []
    rows = conn.execute(
        "SELECT * FROM assets WHERE project_id = ? ORDER BY id", (project_id,)
    )
    for a in rows:
        reasons = []
        if _license_unknown(a["license"]):
            reasons.append("license unknown")
        if a["commercial_use_allowed"] != 1:
            reasons.append(f"commercial use: {_yes_no(a['commercial_use_allowed'])}")
        if a["modification_allowed"] != 1:
            reasons.append(f"modification: {_yes_no(a['modification_allowed'])}")
        if reasons:
            problems.append(
                f"  asset #{a['id']} {a['source_platform']}:{a['asset_id']} "
                f"({a['source_url']}) - " + ", ".join(reasons)
            )
    return problems


def _change_status(conn, project, new_status, note):
    """Validate and apply one transition, logging it to status_history."""
    project_id, old_status = project["id"], project["status"]
    if new_status not in STATUSES:
        raise InvalidTransition(
            f"unknown status '{new_status}'. Valid statuses: {', '.join(STATUSES)}"
        )
    allowed = allowed_next(old_status)
    if new_status not in allowed:
        if not allowed:
            raise InvalidTransition(
                f"project {project_id} is '{old_status}', which is final"
            )
        raise InvalidTransition(
            f"cannot move project {project_id} from '{old_status}' to '{new_status}'; "
            f"allowed next: {', '.join(allowed)}"
        )
    if new_status == "rendered":
        problems = _rights_problems(conn, project_id)
        if problems:
            raise RightsCheckFailed(
                f"project {project_id} cannot move to 'rendered': "
                f"{len(problems)} asset(s) fail the rights check:\n" + "\n".join(problems)
            )

    now = _now()
    conn.execute(
        "UPDATE projects SET status = ?, updated_at = ? WHERE id = ?",
        (new_status, now, project_id),
    )
    conn.execute(
        """
        INSERT INTO status_history (project_id, old_status, new_status, changed_at, note)
        VALUES (?, ?, ?, ?, ?)
        """,
        (project_id, old_status, new_status, now, note),
    )


def _attach_file(table, column, project_id, path, unique):
    stored = _store_path(path)
    with _db() as conn:
        _get_row(conn, project_id)
        if unique:
            existing = conn.execute(
                f"SELECT project_id FROM {table} WHERE {column} = ?", (stored,)
            ).fetchone()
            if existing:
                raise PipelineError(
                    f"{stored} is already attached to project {existing['project_id']}"
                )
        cur = conn.execute(
            f"INSERT INTO {table} (project_id, {column}, created_at) VALUES (?, ?, ?)",
            (project_id, stored, _now()),
        )
        return cur.lastrowid


# --- Public API -------------------------------------------------------------

def init_db():
    """Create the database file and tables if missing. Returns the path."""
    connect().close()
    return DB_PATH


def create_project(niche, topic, notes=None, requires_human_approval=None):
    """Create a project in status 'idea' and return its id.

    requires_human_approval defaults to on for history and off otherwise;
    it can never be turned off for history.
    """
    niche = (niche or "").strip().lower()
    topic = (topic or "").strip()
    if not niche or not topic:
        raise PipelineError("niche and topic are both required")
    if niche in APPROVAL_REQUIRED_NICHES:
        if requires_human_approval is False:
            raise PipelineError(
                f"requires_human_approval cannot be turned off for niche '{niche}'"
            )
        requires = 1
    else:
        requires = 1 if requires_human_approval else 0

    now = _now()
    with _db() as conn:
        cur = conn.execute(
            """
            INSERT INTO projects
                (niche, topic, status, requires_human_approval, created_at, updated_at, notes)
            VALUES (?, ?, 'idea', ?, ?, ?, ?)
            """,
            (niche, topic, requires, now, now, notes),
        )
        project_id = cur.lastrowid
        conn.execute(
            """
            INSERT INTO status_history (project_id, old_status, new_status, changed_at, note)
            VALUES (?, NULL, 'idea', ?, 'project created')
            """,
            (project_id, now),
        )
    return project_id


def get_project(project_id):
    """Return the project with its linked files, assets and status history,
    or None if it doesn't exist."""
    with _db() as conn:
        row = conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
        if row is None:
            return None
        project = dict(row)
        for table in ("research_packs", "scripts", "assets", "renders", "status_history"):
            project[table] = [
                dict(r)
                for r in conn.execute(
                    f"SELECT * FROM {table} WHERE project_id = ? ORDER BY id", (project_id,)
                )
            ]
        for asset in project["assets"]:
            asset["history"] = [
                dict(r)
                for r in conn.execute(
                    "SELECT * FROM asset_history WHERE asset_row_id = ? ORDER BY id",
                    (asset["id"],),
                )
            ]
    return project


def list_projects(status=None):
    """Return all projects (without linked records), optionally filtered by status."""
    if status is not None and status not in STATUSES:
        raise PipelineError(f"unknown status '{status}'. Valid statuses: {', '.join(STATUSES)}")
    with _db() as conn:
        if status is None:
            rows = conn.execute("SELECT * FROM projects ORDER BY id")
        else:
            rows = conn.execute("SELECT * FROM projects WHERE status = ? ORDER BY id", (status,))
        return [dict(r) for r in rows]


def set_status(project_id, new_status, note=None):
    """Move a project one step along the workflow (or to 'rejected').

    'approved' is refused here; it can only be set by approve().
    Moving to 'rendered' is refused while any linked asset fails the rights check.
    """
    if new_status == "approved":
        raise InvalidTransition(
            "'approved' can only be set by approve(project_id, note) "
            "(CLI: python db.py approve <id> --note \"...\")"
        )
    with _db() as conn:
        _change_status(conn, _get_row(conn, project_id), new_status, note)


def approve(project_id, note):
    """Approve a project that is in 'needs_approval'. A note is required."""
    note = (note or "").strip()
    if not note:
        raise PipelineError("approve() needs a note saying who approved it and why")
    with _db() as conn:
        project = _get_row(conn, project_id)
        if project["status"] != "needs_approval":
            raise InvalidTransition(
                f"project {project_id} is '{project['status']}'; "
                "only projects in 'needs_approval' can be approved"
            )
        _change_status(conn, project, "approved", note)


def attach_research(project_id, json_path):
    """Link a research pack JSON file to a project. Returns the row id."""
    return _attach_file("research_packs", "json_path", project_id, json_path, unique=True)


def attach_script(project_id, json_path):
    """Link a script JSON file to a project. Returns the row id."""
    return _attach_file("scripts", "json_path", project_id, json_path, unique=True)


def attach_render(project_id, video_path):
    """Link a rendered video file to a project. Returns the row id."""
    return _attach_file("renders", "video_path", project_id, video_path, unique=False)


def add_asset(
    project_id,
    asset_id,
    source_platform,
    source_url,
    license=None,
    creator=None,
    date_retrieved=None,
    attribution_required=None,
    commercial_use_allowed=None,
    modification_allowed=None,
    local_path=None,
    notes=None,
):
    """Record an asset's rights for a project. Returns the row id.

    Leave license or any yes/no field as None when it is unknown; the
    rights check before 'rendered' treats unknown as not cleared.
    """
    asset_id = str(asset_id).strip()
    source_platform = (source_platform or "").strip().lower()
    source_url = (source_url or "").strip()
    if not asset_id or not source_platform or not source_url:
        raise PipelineError("asset_id, source_platform and source_url are required")
    values = (
        project_id,
        asset_id,
        source_platform,
        source_url,
        creator,
        _clean_license(license),
        date_retrieved or _now(),
        _flag(attribution_required, "attribution_required"),
        _flag(commercial_use_allowed, "commercial_use_allowed"),
        _flag(modification_allowed, "modification_allowed"),
        _store_path(local_path, must_exist=False) if local_path else None,
        notes,
        _now(),
    )
    with _db() as conn:
        _get_row(conn, project_id)
        try:
            cur = conn.execute(
                """
                INSERT INTO assets
                    (project_id, asset_id, source_platform, source_url, creator, license,
                     date_retrieved, attribution_required, commercial_use_allowed,
                     modification_allowed, local_path, notes, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                values,
            )
        except sqlite3.IntegrityError:
            raise PipelineError(
                f"asset {source_platform}:{asset_id} is already recorded for project {project_id}"
            ) from None
        return cur.lastrowid


def update_asset(asset_row_id, note=None, **fields):
    """Change an asset's rights record and log every changed field to asset_history.

    `asset_row_id` is assets.id (the #N shown by `show`). `fields` are asset
    columns from ASSET_EDITABLE; pass None to set a value back to unknown.
    `note` is the reason for the change and goes in the history, not in the
    asset's own `notes` column. Returns the names of the fields that changed.
    """
    if not fields:
        raise PipelineError("update_asset() needs at least one field to change")
    bad = sorted(set(fields) - ASSET_EDITABLE)
    if bad:
        raise PipelineError(
            f"cannot update {', '.join(bad)}. Editable fields: {', '.join(sorted(ASSET_EDITABLE))}"
        )
    new_values = {name: _clean_asset_field(name, value) for name, value in fields.items()}

    with _db() as conn:
        row = conn.execute("SELECT * FROM assets WHERE id = ?", (asset_row_id,)).fetchone()
        if row is None:
            raise PipelineError(f"no asset with id {asset_row_id}")
        changed = [name for name, value in new_values.items() if row[name] != value]
        now = _now()
        for name in changed:
            conn.execute(
                f"UPDATE assets SET {name} = ? WHERE id = ?", (new_values[name], asset_row_id)
            )
            conn.execute(
                """
                INSERT INTO asset_history
                    (asset_row_id, field, old_value, new_value, changed_at, note)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    asset_row_id,
                    name,
                    _history_value(name, row[name]),
                    _history_value(name, new_values[name]),
                    now,
                    note,
                ),
            )
    return changed


# --- CLI --------------------------------------------------------------------

def _print_list(projects):
    if not projects:
        print("No projects.")
        return
    print(f"{'ID':>4}  {'STATUS':<22}  {'NICHE':<10}  {'APPROVAL':<8}  TOPIC")
    for p in projects:
        approval = "required" if p["requires_human_approval"] else "-"
        print(f"{p['id']:>4}  {p['status']:<22}  {p['niche']:<10}  {approval:<8}  {p['topic']}")


def _print_project(p):
    print(f"Project #{p['id']}: {p['topic']}")
    print(f"  niche:          {p['niche']}")
    next_steps = ", ".join(allowed_next(p["status"])) or "none (final)"
    print(f"  status:         {p['status']}   (next: {next_steps})")
    print(f"  human approval: {'required' if p['requires_human_approval'] else 'not required'}")
    print(f"  created:        {p['created_at']}")
    print(f"  updated:        {p['updated_at']}")
    print(f"  notes:          {p['notes'] or '-'}")

    for title, key, column in (
        ("Research packs", "research_packs", "json_path"),
        ("Scripts", "scripts", "json_path"),
        ("Renders", "renders", "video_path"),
    ):
        print(f"\n{title}:")
        for r in p[key]:
            print(f"  #{r['id']}  {r[column]}  ({r['created_at']})")
        if not p[key]:
            print("  none")

    print("\nAssets:")
    for a in p["assets"]:
        print(f"  #{a['id']}  {a['source_platform']}:{a['asset_id']}")
        print(f"      source url:        {a['source_url']}")
        print(f"      creator:           {a['creator'] or 'unknown'}")
        print(f"      license:           {'unknown' if _license_unknown(a['license']) else a['license']}")
        print(f"      retrieved:         {a['date_retrieved']}")
        print(f"      attribution req.:  {_yes_no(a['attribution_required'])}")
        print(f"      commercial use:    {_yes_no(a['commercial_use_allowed'])}")
        print(f"      modification:      {_yes_no(a['modification_allowed'])}")
        print(f"      local file:        {a['local_path'] or '-'}")
        print(f"      notes:             {a['notes'] or '-'}")
        for h in a["history"]:
            old = h["old_value"] if h["old_value"] is not None else "unknown"
            new = h["new_value"] if h["new_value"] is not None else "unknown"
            note = f"  - {h['note']}" if h["note"] else ""
            print(f"      changed {h['changed_at']}  {h['field']}: {old} -> {new}{note}")
    if not p["assets"]:
        print("  none")

    print("\nStatus history:")
    for h in p["status_history"]:
        old = h["old_status"] or "(new)"
        note = f"  - {h['note']}" if h["note"] else ""
        print(f"  {h['changed_at']}  {old} -> {h['new_status']}{note}")


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    parser = argparse.ArgumentParser(description="ClipCast project database.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init", help="create pipeline.db and its tables")

    p_create = sub.add_parser("create", help="create a project in status 'idea'")
    p_create.add_argument("--niche", required=True)
    p_create.add_argument("--topic", required=True)
    p_create.add_argument("--notes")

    p_list = sub.add_parser("list", help="list projects")
    p_list.add_argument("--status", choices=STATUSES)

    p_show = sub.add_parser("show", help="show one project in full")
    p_show.add_argument("id", type=int)

    p_status = sub.add_parser("status", help="move a project to a new status")
    p_status.add_argument("id", type=int)
    p_status.add_argument("new_status")
    p_status.add_argument("--note")

    p_approve = sub.add_parser("approve", help="approve a project in 'needs_approval'")
    p_approve.add_argument("id", type=int)
    p_approve.add_argument("--note", required=True)

    p_asset = sub.add_parser("asset", help="update an asset's rights record (id from 'show')")
    p_asset.add_argument("id", type=int)
    p_asset.add_argument("--license", help="license text, or 'unknown'")
    p_asset.add_argument("--creator")
    for flag in ("attribution", "commercial", "modification"):
        p_asset.add_argument(f"--{flag}", choices=("yes", "no", "unknown"))
    p_asset.add_argument("--note", help="reason for the change (goes in the history)")

    args = parser.parse_args()

    try:
        if args.command == "init":
            print(f"Database ready: {init_db()}")
        elif args.command == "create":
            project_id = create_project(args.niche, args.topic, notes=args.notes)
            print(f"Created project {project_id}.")
            _print_project(get_project(project_id))
        elif args.command == "list":
            _print_list(list_projects(args.status))
        elif args.command == "show":
            project = get_project(args.id)
            if project is None:
                raise ProjectNotFound(f"no project with id {args.id}")
            _print_project(project)
        elif args.command == "status":
            set_status(args.id, args.new_status, note=args.note)
            print(f"Project {args.id} is now '{args.new_status}'.")
        elif args.command == "approve":
            approve(args.id, args.note)
            print(f"Project {args.id} is now 'approved'.")
        elif args.command == "asset":
            flags = {"yes": True, "no": False, "unknown": None}
            fields = {}
            if args.license is not None:
                fields["license"] = args.license
            if args.creator is not None:
                fields["creator"] = args.creator
            for option, column in (
                ("attribution", "attribution_required"),
                ("commercial", "commercial_use_allowed"),
                ("modification", "modification_allowed"),
            ):
                if getattr(args, option) is not None:
                    fields[column] = flags[getattr(args, option)]
            changed = update_asset(args.id, note=args.note, **fields)
            if changed:
                print(f"Asset {args.id} updated: {', '.join(changed)}.")
            else:
                print(f"Asset {args.id}: no changes (values already match).")
    except PipelineError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
