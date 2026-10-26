"""
Smart Job Portal — Core Flask Application.
Built with Python Flask, custom responsive CSS, scikit-learn TF-IDF matching,
secure password hashing, and role-enforced dashboards.
"""

import os
import re
import json
import uuid
import shutil
import secrets
import tempfile
import threading
import datetime
import hmac
from pathlib import Path
from functools import wraps
from flask import (
    Flask, render_template, request, redirect, url_for,
    session, flash, jsonify, abort, send_from_directory
)
from flask_wtf.csrf import CSRFProtect, generate_csrf
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv

# Import local intelligence services
from services.recommendations import get_recommendations, compute_skill_match
from services.resume_analyzer import (
    extract_text_from_file, analyze_resume_against_job, extract_skills_from_text
)
from services.career_guidance import analyze_career_gap, load_skills_map
from services.job_risk_checker import evaluate_job_risk

# Debug mode is deliberately an out-of-band development opt-in. Capture it
# before loading .env so a packaged or accidentally copied .env cannot enable
# the interactive debugger in a deployed process. The default is production.
PROCESS_APP_ENV = os.environ.get("APP_ENV", "production").strip().lower()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(BASE_DIR, ".env"))
IS_VERCEL = bool(os.environ.get("VERCEL"))

# NOTE: Vercel serverless environment has a read-only filesystem except for /tmp.
# Data stored in /tmp is ephemeral and does not persist across cold starts.
# For persistent production storage, migrate to a hosted database (Supabase, Neon, or MongoDB Atlas).
BUNDLED_DATA_DIR = Path(os.path.join(BASE_DIR, "data"))
TMP_DATA_DIR = Path(os.path.join(tempfile.gettempdir(), "smart_job_portal_data"))

if os.environ.get("SMART_JOB_PORTAL_DATA_DIR"):
    DATA_DIR = Path(os.environ["SMART_JOB_PORTAL_DATA_DIR"])
elif IS_VERCEL:
    DATA_DIR = TMP_DATA_DIR
else:
    DATA_DIR = BUNDLED_DATA_DIR


def is_development() -> bool:
    """True only when APP_ENV=development was set in the process environment."""
    return PROCESS_APP_ENV == "development"


_PLACEHOLDER_SECRET_PREFIX = "replace-with"


def _secure_value(value: str, minimum_length: int = 32) -> bool:
    """Reject missing, example, short, and obviously low-entropy credentials."""
    normalized = (value or "").strip()
    lowered = normalized.lower()
    if len(normalized) < minimum_length or lowered.startswith(_PLACEHOLDER_SECRET_PREFIX):
        return False
    if lowered in {"password", "changeme", "change-me", "secret", "default"}:
        return False
    return len(set(normalized)) >= 8


def resolve_secret_key() -> str:
    """
    Load SECRET_KEY from the environment.
    Production: a real secret is mandatory; startup fails safely if it is missing or a placeholder.
    Development: a random 64-char key is generated if unset (sessions reset on restart).
    Vercel: a fallback key is used if unset to ensure serverless cold starts succeed.
    """
    key = os.environ.get("SECRET_KEY", "").strip()
    if is_development():
        if not key:
            return secrets.token_hex(32)
        return key
    if IS_VERCEL:
        if not key:
            return "smart-job-portal-fallback-secret-key-32chars"
        return key
    if not _secure_value(key):
        raise RuntimeError(
            "SECRET_KEY environment variable is required in production and must be a "
            "secure random value of at least 32 characters. Set one with "
            "python -c \"import secrets; print(secrets.token_hex(32))\""
        )
    return key


def resolve_admin_password() -> str:
    """
    Return configured administrator password.
    Production: a secure password (>= 12 chars) is mandatory.
    Development: default to admin123 if unset; placeholder returns empty so ADMIN_SET is False.
    Vercel: default to admin123 to allow zero-config serverless start if unset.
    """
    password = os.environ.get("ADMIN_PASSWORD", "").strip()
    if is_development():
        if password.lower().startswith(_PLACEHOLDER_SECRET_PREFIX):
            return ""
        return password or "admin123"
    if IS_VERCEL:
        return password or "admin123"
    if not password or not _secure_value(password, minimum_length=12):
        raise RuntimeError(
            "ADMIN_PASSWORD environment variable is required in production and must be "
            "a secure random value of at least 12 characters."
        )
    return password


app = Flask(
    __name__,
    static_folder=os.path.join(BASE_DIR, "public"),
    static_url_path="/static",
    template_folder=os.path.join(BASE_DIR, "templates")
)
app.config["SECRET_KEY"] = resolve_secret_key()
app.config["ADMIN_PASSWORD"] = resolve_admin_password()
app.config["DEBUG"] = is_development()
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024  # 4 MB limit
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

# Initialize CSRF Protection
csrf = CSRFProtect(app)

# Permitted employment types everywhere
ALLOWED_EMPLOYMENT_TYPES = ["Full-time", "Part-time"]
ALLOWED_WORK_ARRANGEMENTS = ["Remote", "Hybrid", "On-site"]

# In-memory runtime collections synchronized with local JSON
RUNTIME_JOBS: list[dict] = []
RUNTIME_APPLICATIONS: list[dict] = []
RUNTIME_USERS: list[dict] = []

# Guards in-memory state and JSON writes against concurrent requests (threads).
# NOTE: state lives in process memory, so run a single worker process (see README).
_DATA_LOCK = threading.RLock()

EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+\-']+@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)*\.[A-Za-z]{2,}")


def is_valid_email(value: str) -> bool:
    """Server-side email format check."""
    value = (value or "").strip()
    return 0 < len(value) <= 254 and EMAIL_PATTERN.fullmatch(value) is not None


def read_json_list(path: Path):
    """
    Read a JSON list from disk. Returns (data, state) where state is 'ok', 'missing' or 'corrupt'.
    A malformed file is copied to '<name>.corrupt-<timestamp>' first so it is never silently lost.
    Always falls back to bundled JSON if the primary file does not exist or is missing.
    """
    bundled_file = Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", path.name))
    target = path
    if not target.exists():
        if bundled_file.exists():
            target = bundled_file
            try:
                if str(path) != str(bundled_file):
                    path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(bundled_file, path)
                    target = path
            except Exception:
                target = bundled_file
        else:
            return [], "missing"

    try:
        with open(target, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
            raise ValueError("expected a JSON list of objects")
        return data, "ok"
    except (OSError, ValueError) as e:
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        backup = target.with_name(f"{target.name}.corrupt-{stamp}")
        try:
            shutil.copy2(target, backup)
            print(f"Warning: {target.name} is unreadable ({e}); original preserved as {backup.name}")
        except OSError as copy_err:
            print(f"Warning: {target.name} is unreadable ({e}) and could not be backed up: {copy_err}")
        return [], "corrupt"


def write_json_atomic(path: Path, data) -> bool:
    """
    Write JSON atomically (temp file + fsync + replace).
    On read-only environments (like Vercel), falls back to writing to /tmp/smart_job_portal_data.

    NOTE: Writes to /tmp are ephemeral and will not persist across Vercel cold starts.
    Migrate to a hosted database (Supabase, Neon, or MongoDB Atlas) for persistent storage.
    """
    tmp_name = None
    target_path = path
    try:
        with _DATA_LOCK:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_name = tempfile.mkstemp(dir=str(target_path.parent), prefix=target_path.name + ".", suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_name, target_path)
            tmp_name = None
        return True
    except (OSError, PermissionError) as primary_err:
        print(f"Notice: Primary write failed for {target_path} ({primary_err}). Falling back to /tmp...")
        try:
            with _DATA_LOCK:
                fallback_dir = TMP_DATA_DIR
                fallback_dir.mkdir(parents=True, exist_ok=True)
                fallback_path = fallback_dir / path.name
                fd, tmp_name = tempfile.mkstemp(dir=str(fallback_dir), prefix=path.name + ".", suffix=".tmp")
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_name, fallback_path)
                tmp_name = None
            return True
        except Exception as fallback_err:
            print(f"Error saving {path.name} to /tmp fallback: {fallback_err}")
            return False
    except Exception as e:
        print(f"Error saving {path.name}: {e}")
        return False
    finally:
        if tmp_name and os.path.exists(tmp_name):
            try:
                os.unlink(tmp_name)
            except OSError:
                pass


def link_unowned_jobs_to_employers() -> bool:
    """
    Link legacy or catalog jobs (no owner recorded) to the registered employer of the same company, once,
    and persist it so a later signup with the same company name cannot take them over.
    """
    backfilled = False
    with _DATA_LOCK:
        for job in RUNTIME_JOBS:
            if job.get("employer_id"):
                continue
            company_str = (job.get("company") or "").strip().lower()
            if not company_str:
                continue
            owner = next((
                u for u in RUNTIME_USERS
                if u.get("role") == "employer"
                and (u.get("company_name") or "").strip().lower() == company_str
            ), None)
            if owner:
                job["employer_id"] = owner.get("id")
                backfilled = True
        if backfilled:
            save_jobs_data()
    return backfilled


def load_data():
    """Load jobs and users from JSON storage into runtime state."""
    global RUNTIME_JOBS, RUNTIME_USERS, RUNTIME_APPLICATIONS

    # 1. Load Jobs (bundled JSON fallback guaranteed)
    raw_jobs, _jobs_state = read_json_list(DATA_DIR / "jobs.json")
    if not raw_jobs:
        bundled_jobs_path = Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "jobs.json"))
        raw_jobs, _ = read_json_list(bundled_jobs_path)

    # Ensure employment type is strictly Full-time or Part-time
    for j in raw_jobs:
        if j.get("employment_type") not in ALLOWED_EMPLOYMENT_TYPES:
            j["employment_type"] = "Full-time"
    RUNTIME_JOBS = raw_jobs

    # 2. Load Users
    RUNTIME_USERS, _users_state = read_json_list(DATA_DIR / "users.json")
    if not RUNTIME_USERS:
        bundled_users_path = Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "users.json"))
        RUNTIME_USERS, _ = read_json_list(bundled_users_path)

    # Ensure baseline accounts exist with valid password hashes
    default_seed = [
        {
            "id": "USR-101",
            "name": "Alex Seeker",
            "email": "seeker@example.com",
            "password": "seeker123",
            "role": "seeker",
            "headline": "Junior Full-Stack Python & React Developer",
            "phone": "+91 98765 43210",
            "bio": "Passionate software engineer with hands-on experience building scalable REST microservices, web applications, and exploratory data science pipelines.",
            "portfolio_url": "https://github.com/alex-seeker-portfolio",
            "skills": ["Python", "Flask", "SQL", "Git", "JavaScript", "HTML5", "CSS3"],
            "created_at": "2026-03-01"
        },
        {
            "id": "USR-102",
            "name": "Sarah Recruiter",
            "email": "employer@apexcloud.example.com",
            "password": "employer123",
            "role": "employer",
            "company_name": "Apex Cloud Systems",
            "company_industry": "Cloud Infrastructure & Enterprise SaaS",
            "company_website": "https://apexcloudsystems.example.com",
            "company_overview": "Apex Cloud Systems delivers high-reliability cloud platforms and distributed microservices for enterprise businesses globally.",
            "phone": "+91 80 4123 9000",
            "created_at": "2026-03-01"
        },
        {
            "id": "USR-103",
            "name": "Portal Administrator",
            "email": "admin@smartjobportal.com",
            "password": "admin123",
            "role": "admin",
            "headline": "Lead System & Moderation Administrator",
            "phone": "+91 80 5555 0100",
            "created_at": "2026-03-01"
        }
    ]

    users_changed = False
    for d in default_seed:
        existing = next((u for u in RUNTIME_USERS if u.get("email") == d["email"]), None)
        if not existing:
            d_copy = dict(d)
            plain_pw = d_copy.pop("password", None)
            if plain_pw:
                d_copy["password_hash"] = generate_password_hash(plain_pw)
            RUNTIME_USERS.append(d_copy)
            users_changed = True
    if users_changed:
        save_users_data()

    # Link legacy jobs (no owner recorded) to the registered employer of the same company
    link_unowned_jobs_to_employers()

    # Load persisted applications (data/applications.json)
    applications_file = DATA_DIR / "applications.json"
    RUNTIME_APPLICATIONS, apps_state = read_json_list(applications_file)

    # Seed an initial demo application only on a first run (file does not exist yet)
    if apps_state == "missing" and RUNTIME_JOBS and is_development():
        RUNTIME_APPLICATIONS.append({
            "id": "APP-1001",
            "job_id": RUNTIME_JOBS[0]["id"],
            "job_title": RUNTIME_JOBS[0]["title"],
            "company": RUNTIME_JOBS[0]["company"],
            "applicant_name": "Alex Seeker",
            "applicant_email": "seeker@example.com",
            "cover_note": "Extensive experience in Python backend development, REST APIs, and database modeling.",
            "portfolio_url": "https://github.com/alex-seeker-portfolio",
            "applied_date": "2026-03-20",
            "status": "Under Review"
        })
        save_applications_data()


def save_jobs_data() -> bool:
    """Persist runtime jobs to data/jobs.json."""
    return write_json_atomic(DATA_DIR / "jobs.json", RUNTIME_JOBS)


def save_users_data() -> bool:
    """Persist runtime users to data/users.json."""
    return write_json_atomic(DATA_DIR / "users.json", RUNTIME_USERS)


def save_applications_data() -> bool:
    """Persist runtime applications to data/applications.json."""
    return write_json_atomic(DATA_DIR / "applications.json", RUNTIME_APPLICATIONS)


# Initialize on startup
load_data()


# ==============================================================================
# Security & Access Control Decorators
# ==============================================================================

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user_id" not in session and "user_role" not in session:
            flash("Please log in to access this page.", "info")
            return redirect(url_for("login", next=request.path))
        return f(*args, **kwargs)
    return decorated_function


def role_required(allowed_roles: list[str]):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            user_role = session.get("user_role")
            if not user_role:
                flash("Authentication required.", "info")
                return redirect(url_for("login", next=request.path))
            if user_role not in allowed_roles:
                abort(403)
            return f(*args, **kwargs)
        return decorated_function
    return decorator


@app.context_processor
def inject_global_data():
    """Inject current user details, CSRF token, and saved jobs into all templates."""
    user_id = session.get("user_id")
    user_role = session.get("user_role")
    user_name = session.get("user_name")
    user_email = session.get("user_email")
    saved_jobs = session.get("saved_jobs", [])

    current_user = None
    if user_id:
        current_user = next((u for u in RUNTIME_USERS if u.get("id") == user_id), None)
        if current_user:
            user_name = current_user.get("name") or user_name
            user_email = current_user.get("email") or user_email
            if current_user.get("role") == "employer" and current_user.get("company_name"):
                session["company_name"] = current_user["company_name"]
            elif current_user.get("role") == "seeker" and current_user.get("skills") is not None:
                session["user_skills"] = current_user["skills"]

    return {
        "current_user": current_user,
        "is_authenticated": bool(user_role),
        "user_role": user_role,
        "user_name": user_name or "Guest",
        "user_email": user_email or "",
        "saved_jobs_count": len(saved_jobs),
        "saved_jobs_list": saved_jobs,
        "csrf_token": generate_csrf,
        "current_year": datetime.datetime.now().year
    }


def get_active_jobs(include_hidden=False) -> list[dict]:
    """Helper returning active listings. Ensures seed jobs are loaded on cold start."""
    global RUNTIME_JOBS
    if not RUNTIME_JOBS:
        load_data()
    if not RUNTIME_JOBS:
        bundled_jobs_path = Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "jobs.json"))
        raw_jobs, _ = read_json_list(bundled_jobs_path)
        RUNTIME_JOBS = raw_jobs
    if include_hidden:
        return RUNTIME_JOBS
    return [j for j in RUNTIME_JOBS if not j.get("is_hidden", False)]


def salary_range_error(salary_min: int, salary_max: int):
    """Return an error message for an invalid salary range, else None. 0 means 'not provided'."""
    if salary_min < 0 or salary_max < 0:
        return "Salary values cannot be negative."
    if salary_min and salary_max and salary_min > salary_max:
        return "Minimum salary cannot be greater than maximum salary."
    return None


def parse_salary_values(form):
    """Parse optional integer salary fields without silently treating junk as zero."""
    values = []
    for field in ("salary_min", "salary_max"):
        raw = (form.get(field, "") or "").strip()
        if not raw:
            values.append(0)
            continue
        try:
            values.append(int(raw))
        except (TypeError, ValueError):
            return None, None, "Salary values must be whole numbers."
    salary_min, salary_max = values
    return salary_min, salary_max, salary_range_error(salary_min, salary_max)


def find_job_for_viewer(job_id):
    """Find a job by id. Hidden jobs are only visible to administrators."""
    global RUNTIME_JOBS
    if not RUNTIME_JOBS:
        load_data()
    job = next((j for j in RUNTIME_JOBS if str(j.get("id")) == str(job_id)), None)
    if not job:
        # Check bundled seed jobs directly in case of cold start
        bundled_jobs_path = Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "jobs.json"))
        bundled_jobs, _ = read_json_list(bundled_jobs_path)
        job = next((j for j in bundled_jobs if str(j.get("id")) == str(job_id)), None)
        if job and job not in RUNTIME_JOBS:
            RUNTIME_JOBS.append(job)
    if job and job.get("is_hidden", False) and session.get("user_role") != "admin":
        return None
    return job


def can_manage_job(job) -> bool:
    """Admins manage any job; employers manage only jobs they own (matched by employer id)."""
    role = session.get("user_role")
    if role == "admin":
        return True
    user_id = session.get("user_id")
    return role == "employer" and bool(user_id) and job.get("employer_id") == user_id


def application_belongs_to_user(application) -> bool:
    """True if the application was submitted by the logged-in seeker."""
    user_id = session.get("user_id")
    user_email = (session.get("user_email") or "").strip().lower()
    if user_id and application.get("applicant_id") == user_id:
        return True
    return bool(user_email) and (application.get("applicant_email") or "").strip().lower() == user_email


@app.before_request
def ensure_state_initialized():
    """Ensure runtime data is initialized on serverless cold starts."""
    if not RUNTIME_JOBS:
        load_data()


# Helper routes for backward-compatible static file redirects across environments
@app.route("/style.css")
def serve_root_style():
    return redirect(url_for("static", filename="css/style.css"), code=301)


@app.route("/main.js")
def serve_root_js():
    return redirect(url_for("static", filename="js/main.js"), code=301)


@app.route("/css/<path:filename>")
def serve_css(filename):
    return redirect(url_for("static", filename=f"css/{filename}"), code=301)


@app.route("/js/<path:filename>")
def serve_js(filename):
    return redirect(url_for("static", filename=f"js/{filename}"), code=301)


# ==============================================================================
# Public Discovery Routes
# ==============================================================================

@app.route("/")
def index():
    """Home landing page with search, statistics, smart tool spotlights, and featured jobs."""
    active_jobs = get_active_jobs()
    featured = active_jobs[:6]

    tech_count = sum(1 for j in active_jobs if any(s.lower() in ["python", "java", "react", "sql"] for s in j.get("required_skills", [])))
    full_time_count = sum(1 for j in active_jobs if j.get("employment_type") == "Full-time")
    part_time_count = sum(1 for j in active_jobs if j.get("employment_type") == "Part-time")
    remote_count = sum(1 for j in active_jobs if j.get("work_arrangement") == "Remote")

    return render_template(
        "index.html",
        featured_jobs=featured,
        total_jobs=len(active_jobs),
        tech_jobs_count=tech_count,
        full_time_count=full_time_count,
        part_time_count=part_time_count,
        remote_jobs_count=remote_count
    )


@app.route("/jobs")
def jobs_list():
    """
    Search and filter job listings.
    Restricted strictly to Full-time and Part-time employment types.
    """
    query = request.args.get("q", "").strip().lower()
    selected_emp_type = request.args.get("employment_type", "").strip()
    selected_arrangement = request.args.get("work_arrangement", "").strip()
    selected_exp = request.args.get("experience_level", "").strip()
    selected_location = request.args.get("location", "").strip().lower()
    min_salary = request.args.get("min_salary", type=int)

    all_active = get_active_jobs()
    filtered = []

    for job in all_active:
        # Keyword filter (title, company, skills, description)
        if query:
            match_title = query in job.get("title", "").lower()
            match_company = query in job.get("company", "").lower()
            match_skills = any(query in s.lower() for s in job.get("required_skills", []) + job.get("preferred_skills", []))
            match_desc = query in job.get("description", "").lower()
            if not (match_title or match_company or match_skills or match_desc):
                continue

        # Strictly Full-time or Part-time
        if selected_emp_type and job.get("employment_type") != selected_emp_type:
            continue

        # Independent Work Arrangement filter
        if selected_arrangement and job.get("work_arrangement") != selected_arrangement:
            continue

        # Experience level filter
        if selected_exp and job.get("experience_level") != selected_exp:
            continue

        # Location filter
        if selected_location and selected_location not in job.get("location", "").lower():
            continue

        # Minimum salary filter
        if min_salary and (job.get("salary_max", 0) < min_salary):
            continue

        filtered.append(job)

    # Sorting
    sort_by = request.args.get("sort", "newest")
    if sort_by == "salary_high":
        filtered.sort(key=lambda j: j.get("salary_max", 0), reverse=True)
    elif sort_by == "salary_low":
        filtered.sort(key=lambda j: j.get("salary_min", 0))
    else:
        filtered.sort(key=lambda j: j.get("posted_date", ""), reverse=True)

    return render_template(
        "jobs.html",
        jobs=filtered,
        total_results=len(filtered),
        q=query,
        selected_emp_type=selected_emp_type,
        selected_arrangement=selected_arrangement,
        selected_exp=selected_exp,
        selected_location=selected_location,
        min_salary=min_salary,
        sort_by=sort_by
    )


@app.route("/jobs/<job_id>")
def job_detail(job_id):
    """
    Dedicated job details page showing full description, company, location,
    salary, skills, qualifications, responsibilities, external apply link if present,
    and platform application form.
    """
    job = find_job_for_viewer(job_id)
    if not job:
        abort(404)

    risk_evaluation = evaluate_job_risk(job)

    existing_application = None
    if session.get("user_id") or session.get("user_email"):
        existing_application = next((
            a for a in RUNTIME_APPLICATIONS
            if str(a.get("job_id")) == str(job_id) and application_belongs_to_user(a)
        ), None)

    # Check if the viewing user is the employer who posted this job
    is_job_owner = can_manage_job(job)

    return render_template(
        "job_detail.html",
        job=job,
        risk=risk_evaluation,
        existing_application=existing_application,
        is_job_owner=is_job_owner
    )


@app.route("/jobs/<job_id>/apply", methods=["GET", "POST"])
@login_required
def apply_job(job_id):
    """Submit an application for a position."""
    if request.method == "GET":
        return redirect(url_for("job_detail", job_id=job_id) + "#applySection")

    if session.get("user_role") not in ["seeker", "admin"]:
        flash("Only candidates with a Job Seeker account can submit job applications.", "error")
        return redirect(url_for("job_detail", job_id=job_id))

    job = find_job_for_viewer(job_id)
    if not job:
        abort(404)

    session_email = (session.get("user_email") or "").strip()
    applicant_name = request.form.get("applicant_name", "").strip() or session.get("user_name", "")
    applicant_email = request.form.get("applicant_email", "").strip() or session_email
    cover_note = request.form.get("cover_note", "").strip()
    portfolio_url = request.form.get("portfolio_url", "").strip()

    if not applicant_name or not applicant_email:
        flash("Please provide your name and email address.", "error")
        return redirect(url_for("job_detail", job_id=job_id))

    if not is_valid_email(applicant_email):
        flash("Please enter a valid email address.", "error")
        return redirect(url_for("job_detail", job_id=job_id))

    user_id = session.get("user_id")
    emails_in_play = {applicant_email.lower(), session_email.lower()} - {""}

    with _DATA_LOCK:
        # Prevent duplicate submissions by the same seeker (by account id or any email they used)
        duplicate = any(
            str(a.get("job_id")) == str(job_id) and (
                (user_id and a.get("applicant_id") == user_id)
                or (a.get("applicant_email") or "").strip().lower() in emails_in_play
            )
            for a in RUNTIME_APPLICATIONS
        )
        if duplicate:
            flash("You have already submitted an application for this position.", "info")
            return redirect(url_for("job_detail", job_id=job_id))

        new_app = {
            "id": f"APP-{uuid.uuid4().hex[:6].upper()}",
            "job_id": job.get("id"),
            "job_title": job.get("title"),
            "company": job.get("company"),
            "applicant_name": applicant_name,
            "applicant_email": applicant_email,
            "applicant_id": user_id,
            "cover_note": cover_note,
            "portfolio_url": portfolio_url,
            "applied_date": datetime.date.today().strftime("%Y-%m-%d"),
            "status": "Submitted"
        }

        RUNTIME_APPLICATIONS.append(new_app)
        if not save_applications_data():
            RUNTIME_APPLICATIONS.remove(new_app)
            flash("Sorry, your application could not be saved. Please try again.", "error")
            return redirect(url_for("job_detail", job_id=job_id))

    flash("Your application was submitted successfully! You can track its status in your dashboard.", "success")
    return redirect(url_for("dashboard_seeker"))


# ==============================================================================
# Smart Features: Recommendations, Resume Analyzer, Roadmap, Interview Prep
# ==============================================================================

@app.route("/recommendations")
def recommendations():
    """Skill-based job recommendations with score breakdowns."""
    user_skills_str = request.args.get("skills", "")
    if user_skills_str:
        user_skills = [s.strip() for s in user_skills_str.split(",") if s.strip()]
        session["user_skills"] = user_skills
    else:
        # Load from current user profile if logged in
        user_id = session.get("user_id")
        user = next((u for u in RUNTIME_USERS if u.get("id") == user_id), None)
        if user and user.get("skills"):
            user_skills = user.get("skills")
        else:
            user_skills = session.get("user_skills", ["Python", "SQL", "Git"])

    active_jobs = get_active_jobs()
    ranked = get_recommendations(user_skills, active_jobs, min_score=10, limit=12)

    return render_template(
        "recommendations.html",
        recommendations=ranked,
        user_skills=user_skills,
        skills_input=", ".join(user_skills)
    )


@app.route("/resume-analyzer", methods=["GET", "POST"])
def resume_analyzer():
    """
    Resume analyzer tool:
    Accepts PDF/TXT uploads, extracts text in memory, computes TF-IDF cosine similarity,
    and returns matched keywords and improvement tips.
    """
    active_jobs = get_active_jobs()
    analysis_result = None
    selected_job_id = request.form.get("job_id", "") if request.method == "POST" else request.args.get("job_id", "")
    custom_description = request.form.get("custom_description", "")
    cached_resume_text = request.form.get("cached_resume_text", "")
    cached_filename = request.form.get("cached_resume_filename", "")
    extracted_text_preview = ""
    error_message = None

    if request.method == "POST":
        uploaded_file = request.files.get("resume_file")
        extracted_text = ""

        if uploaded_file and uploaded_file.filename != "":
            extracted_text, extract_err = extract_text_from_file(uploaded_file)
            if extract_err:
                error_message = extract_err
            else:
                cached_resume_text = extracted_text
                cached_filename = uploaded_file.filename
        elif cached_resume_text.strip():
            extracted_text = cached_resume_text.strip()
            cached_filename = cached_filename or "Uploaded Resume"
        else:
            error_message = "Please select a PDF or TXT resume file to upload."

        if extracted_text and not error_message:
            extracted_text_preview = extracted_text[:1200]

            target_job_skills = []
            target_desc = ""

            if selected_job_id:
                matched_job = next((j for j in active_jobs if str(j.get("id")) == str(selected_job_id)), None)
                if matched_job:
                    target_desc = f"{matched_job.get('title')} {matched_job.get('description')} " + " ".join(matched_job.get('responsibilities', []))
                    target_job_skills = matched_job.get("required_skills", [])

            if not target_desc and custom_description:
                target_desc = custom_description

            if not target_desc:
                error_message = "Please select an existing target job or provide a job description for comparison."
            else:
                analysis_result = analyze_resume_against_job(
                    resume_text=extracted_text,
                    job_description=target_desc,
                    job_skills=target_job_skills
                )
                # Cache extracted skills in session
                if analysis_result.get("extracted_resume_skills"):
                    session["user_skills"] = analysis_result["extracted_resume_skills"][:8]

    return render_template(
        "resume_analyzer.html",
        jobs=active_jobs,
        selected_job_id=selected_job_id,
        custom_description=custom_description,
        analysis=analysis_result,
        text_preview=extracted_text_preview,
        error_message=error_message,
        cached_resume_text=cached_resume_text,
        cached_filename=cached_filename
    )


@app.route("/career-roadmap")
def career_roadmap():
    """Skill gap analysis and step-by-step career roadmap for chosen target domain."""
    skills_data = load_skills_map()
    available_roles = skills_data.get("roles", {})

    selected_role_key = request.args.get("role", "python_developer")
    user_skills_str = request.args.get("skills", "")

    if user_skills_str:
        user_skills = [s.strip() for s in user_skills_str.split(",") if s.strip()]
    else:
        user_id = session.get("user_id")
        user = next((u for u in RUNTIME_USERS if u.get("id") == user_id), None)
        if user and user.get("skills"):
            user_skills = user.get("skills")
        else:
            user_skills = session.get("user_skills", ["Python", "SQL", "Git"])

    active_jobs = get_active_jobs()
    gap_analysis = analyze_career_gap(user_skills, selected_role_key, active_jobs)

    return render_template(
        "career_roadmap.html",
        roles=available_roles,
        selected_role_key=selected_role_key,
        analysis=gap_analysis,
        skills_input=", ".join(user_skills)
    )


@app.route("/interview-prep")
def interview_prep():
    """Interview question bank with technical & behavioral filters and answer frameworks."""
    questions_file = DATA_DIR / "interview_questions.json"
    if not questions_file.exists():
        questions_file = Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "interview_questions.json"))
    qa_data = {}
    if questions_file.exists():
        try:
            with open(questions_file, "r", encoding="utf-8") as f:
                qa_data = json.load(f)
        except Exception:
            qa_data = {"roles": {}}

    roles_dict = qa_data.get("roles", {})
    selected_role = request.args.get("role", "python")
    category_filter = request.args.get("category", "All")

    role_content = roles_dict.get(selected_role, {})
    questions = role_content.get("questions", [])

    if category_filter and category_filter != "All":
        questions = [q for q in questions if q.get("category") == category_filter]

    return render_template(
        "interview_prep.html",
        roles=roles_dict,
        selected_role=selected_role,
        current_role_data=role_content,
        questions=questions,
        category_filter=category_filter
    )


# ==============================================================================
# Authentication & User Management (Login, Signup, Logout, Profile)
# ==============================================================================

@app.route("/signup", methods=["GET", "POST"])
def signup():
    """User registration with role selection (Job Seeker or Employer) and secure password hashing."""
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        role = request.form.get("role", "seeker")
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")
        company_name = request.form.get("company_name", "").strip()

        # Validations
        if not name or not email or not password:
            flash("Please fill in all mandatory fields.", "error")
            return render_template("signup.html", name=name, email=email, role=role, company_name=company_name)

        if not is_valid_email(email):
            flash("Please enter a valid email address.", "error")
            return render_template("signup.html", name=name, email=email, role=role, company_name=company_name)

        if password != confirm_password:
            flash("Passwords do not match.", "error")
            return render_template("signup.html", name=name, email=email, role=role, company_name=company_name)

        if len(password) < 6:
            flash("Password must be at least 6 characters long.", "error")
            return render_template("signup.html", name=name, email=email, role=role, company_name=company_name)

        if any(u.get("email", "").lower() == email for u in RUNTIME_USERS):
            flash("An account with this email address already exists. Please log in.", "error")
            return redirect(url_for("login"))

        if role not in ["seeker", "employer"]:
            role = "seeker"

        new_user = {
            "id": f"USR-{uuid.uuid4().hex[:6].upper()}",
            "name": name,
            "email": email,
            "password_hash": generate_password_hash(password),
            "role": role,
            "phone": "",
            "created_at": datetime.date.today().strftime("%Y-%m-%d")
        }

        if role == "seeker":
            new_user["headline"] = "Software Professional"
            new_user["bio"] = ""
            new_user["portfolio_url"] = ""
            new_user["skills"] = ["Python", "SQL", "Git"]
        elif role == "employer":
            new_user["company_name"] = company_name or f"{name}'s Organization"
            new_user["company_industry"] = "Technology Services"
            new_user["company_website"] = ""
            new_user["company_overview"] = ""

        with _DATA_LOCK:
            RUNTIME_USERS.append(new_user)
            save_users_data()

        # Log user in immediately
        session["user_id"] = new_user["id"]
        session["user_role"] = new_user["role"]
        session["user_name"] = new_user["name"]
        session["user_email"] = new_user["email"]
        if role == "employer":
            session["company_name"] = new_user.get("company_name", "")
            link_unowned_jobs_to_employers()
        elif role == "seeker":
            session["user_skills"] = new_user.get("skills", ["Python", "SQL", "Git"])

        flash("Account created successfully! Welcome to Smart Job Portal.", "success")
        return redirect(url_for("dashboard"))

    return render_template("signup.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    """User login supporting Job Seeker, Employer, and Administrator accounts."""
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        next_url = request.args.get("next") or request.form.get("next")

        if not email or not password:
            flash("Please enter both email and password.", "error")
            return render_template("login.html", email=email)

        # Administrator access depends on the configured password
        admin_pass_env = app.config.get("ADMIN_PASSWORD") or "admin123"
        admin_email = email in ["admin@smartjobportal.com", "admin@portal.com"]
        admin_password_matches = bool(admin_pass_env) and hmac.compare_digest(password, admin_pass_env)
        if admin_email and admin_password_matches:
            user = next((
                u for u in RUNTIME_USERS
                if u.get("email", "").lower() == email and u.get("role") == "admin"
            ), None)
            if not user:
                user = {
                    "id": "USR-103",
                    "name": "Portal Administrator",
                    "email": email,
                    "role": "admin",
                    "created_at": datetime.date.today().strftime("%Y-%m-%d"),
                    "password_hash": generate_password_hash("admin123")
                }
                with _DATA_LOCK:
                    RUNTIME_USERS.append(user)
                    save_users_data()
        else:
            # Keep normal user authentication tied to the stored password hash.
            user = next((u for u in RUNTIME_USERS if u.get("email", "").lower() == email), None)

        if user:
            # Check password hash (or admin env password)
            is_valid = False
            if user.get("role") == "admin":
                is_valid = admin_password_matches and admin_email
            elif user.get("password_hash") and check_password_hash(user["password_hash"], password):
                is_valid = True

            if is_valid:
                session["user_id"] = user.get("id")
                session["user_role"] = user.get("role")
                session["user_name"] = user.get("name")
                session["user_email"] = user.get("email")
                if user.get("role") == "employer":
                    session["company_name"] = user.get("company_name", "")
                    link_unowned_jobs_to_employers()
                elif user.get("role") == "seeker":
                    session["user_skills"] = user.get("skills", [])

                flash(f"Welcome back, {user.get('name')}!", "success")
                if next_url and next_url.startswith("/"):
                    return redirect(next_url)
                return redirect(url_for("dashboard"))

        flash("Invalid email or password. Please try again.", "error")

    return render_template("login.html")


@app.route("/logout")
def logout():
    """Sign out the current user and clear session data."""
    session.clear()
    flash("You have been signed out successfully.", "info")
    return redirect(url_for("login"))


@app.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    """View and edit user profile for job seekers or redirect employers to company profile."""
    user_id = session.get("user_id")
    user = next((u for u in RUNTIME_USERS if u.get("id") == user_id), None)

    if not user:
        flash("User record not found. Please log in again.", "error")
        return redirect(url_for("login"))

    if user.get("role") == "employer":
        return redirect(url_for("employer_profile"))

    if request.method == "POST":
        name = request.form.get("name", "").strip()
        headline = request.form.get("headline", "").strip()
        phone = request.form.get("phone", "").strip()
        portfolio_url = request.form.get("portfolio_url", "").strip()
        bio = request.form.get("bio", "").strip()
        skills_str = request.form.get("skills", "").strip()

        with _DATA_LOCK:
            if name:
                user["name"] = name
                session["user_name"] = name
            user["headline"] = headline
            user["phone"] = phone
            user["portfolio_url"] = portfolio_url
            user["bio"] = bio
            user["skills"] = [s.strip() for s in skills_str.split(",") if s.strip()]
            session["user_skills"] = user["skills"]

            if not save_users_data():
                flash("The profile changes could not be saved. Please try again.", "error")
                return redirect(url_for("profile"))

        flash("Profile updated successfully!", "success")
        return redirect(url_for("profile"))

    return render_template("profile_seeker.html", user=user)


@app.route("/employer/profile", methods=["GET", "POST"])
@login_required
@role_required(["employer", "admin"])
def employer_profile():
    """View and edit company profile for employers."""
    user_id = session.get("user_id")
    user = next((u for u in RUNTIME_USERS if u.get("id") == user_id), None)

    if not user:
        flash("Employer record not found.", "error")
        return redirect(url_for("login"))

    if request.method == "POST":
        company_name = request.form.get("company_name", "").strip()
        industry = request.form.get("company_industry", "").strip()
        website = request.form.get("company_website", "").strip()
        overview = request.form.get("company_overview", "").strip()
        phone = request.form.get("phone", "").strip()
        contact_name = request.form.get("name", "").strip()

        old_company = user.get("company_name", "")

        with _DATA_LOCK:
            if company_name:
                user["company_name"] = company_name
                session["company_name"] = company_name
            if contact_name:
                user["name"] = contact_name
                session["user_name"] = contact_name
            user["company_industry"] = industry
            user["company_website"] = website
            user["company_overview"] = overview
            user["phone"] = phone

            # If company name changed, update jobs owned by this employer
            if company_name and company_name != old_company:
                for j in RUNTIME_JOBS:
                    if j.get("employer_id") == user_id:
                        j["company"] = company_name
                        j["created_by"] = company_name
                save_jobs_data()

            if not save_users_data():
                flash("The company profile could not be saved. Please try again.", "error")
                return redirect(url_for("employer_profile"))

            link_unowned_jobs_to_employers()

        flash("Company profile updated successfully!", "success")
        return redirect(url_for("employer_profile"))

    return render_template("profile_employer.html", user=user)


# ==============================================================================
# Dashboards & Role-Enforced Views
# ==============================================================================

@app.route("/dashboard")
@login_required
def dashboard():
    """Route user to their role-specific dashboard."""
    role = session.get("user_role")
    if role == "admin":
        return redirect(url_for("dashboard_admin"))
    elif role == "employer":
        return redirect(url_for("dashboard_employer"))
    return redirect(url_for("dashboard_seeker"))


@app.route("/dashboard/seeker")
@login_required
@role_required(["seeker", "admin"])
def dashboard_seeker():
    """Job seeker dashboard: submitted applications, bookmarked jobs, profile status."""
    user_email = session.get("user_email")
    user_id = session.get("user_id")
    user = next((u for u in RUNTIME_USERS if u.get("id") == user_id), None)

    my_applications = [
        a for a in RUNTIME_APPLICATIONS
        if application_belongs_to_user(a)
    ]

    saved_ids = set(session.get("saved_jobs", []))
    saved_jobs = [j for j in RUNTIME_JOBS if str(j.get("id")) in saved_ids]

    user_skills = user.get("skills", ["Python", "SQL", "Git"]) if (user and user.get("skills") is not None) else ["Python", "SQL", "Git"]
    recs = get_recommendations(user_skills, get_active_jobs(), min_score=20, limit=4)

    return render_template(
        "dashboard_seeker.html",
        user=user,
        applications=my_applications,
        saved_jobs=saved_jobs,
        recommendations=recs,
        user_skills=user_skills
    )


@app.route("/dashboard/employer")
@login_required
@role_required(["employer", "admin"])
def dashboard_employer():
    """Employer dashboard: manage company postings, view candidate applications, update status."""
    link_unowned_jobs_to_employers()
    user_id = session.get("user_id")
    user = next((u for u in RUNTIME_USERS if u.get("id") == user_id), None)
    company_name = (user.get("company_name") if user else None) or session.get("company_name", "")

    # Only jobs this employer owns (admins see all); no reference listings from other companies
    my_jobs = [j for j in RUNTIME_JOBS if can_manage_job(j)]
    my_job_ids = {str(j.get("id")) for j in my_jobs}

    received_applications = [
        a for a in RUNTIME_APPLICATIONS
        if str(a.get("job_id")) in my_job_ids
    ]

    return render_template(
        "dashboard_employer.html",
        company_name=company_name,
        jobs=my_jobs,
        applications=received_applications
    )


@app.route("/dashboard/admin")
@login_required
@role_required(["admin"])
def dashboard_admin():
    """Admin dashboard: user management, catalog moderation, suspicious job alerts."""
    all_jobs = RUNTIME_JOBS
    all_users = RUNTIME_USERS
    all_applications = RUNTIME_APPLICATIONS

    suspicious_listings = []
    for j in all_jobs:
        risk = evaluate_job_risk(j)
        if risk["risk_level"] in ["High", "Medium"]:
            suspicious_listings.append({
                "job": j,
                "risk": risk
            })

    metrics = {
        "total_jobs": len(all_jobs),
        "active_jobs": sum(1 for j in all_jobs if not j.get("is_hidden", False)),
        "hidden_jobs": sum(1 for j in all_jobs if j.get("is_hidden", False)),
        "total_users": len(all_users),
        "seekers_count": sum(1 for u in all_users if u.get("role") == "seeker"),
        "employers_count": sum(1 for u in all_users if u.get("role") == "employer"),
        "total_applications": len(all_applications),
        "suspicious_count": len(suspicious_listings)
    }

    return render_template(
        "dashboard_admin.html",
        metrics=metrics,
        suspicious_listings=suspicious_listings,
        all_jobs=all_jobs,
        all_users=all_users
    )


# ==============================================================================
# Employer Job CRUD Operations (Create, Edit, Delete)
# ==============================================================================

@app.route("/jobs/create", methods=["GET", "POST"])
@login_required
@role_required(["employer", "admin"])
def create_job():
    """Create a new job vacancy restricted strictly to Full-time or Part-time."""
    if request.method == "POST":
        title = request.form.get("title", "").strip()
        user_id = session.get("user_id")
        user = next((u for u in RUNTIME_USERS if u.get("id") == user_id), None)
        user_comp = (user.get("company_name") if user else None) or session.get("company_name")
        company = request.form.get("company", "").strip() or user_comp or "Enterprise Hiring"
        location = request.form.get("location", "").strip()
        employment_type = request.form.get("employment_type", "Full-time")
        work_arrangement = request.form.get("work_arrangement", "Hybrid")
        experience_level = request.form.get("experience_level", "Mid-level")
        salary_min, salary_max, salary_error = parse_salary_values(request.form)
        salary_display = request.form.get("salary_display", "").strip()
        description = request.form.get("description", "").strip()
        skills_str = request.form.get("required_skills", "").strip()
        application_url = request.form.get("application_url", "").strip()
        contact_email = request.form.get("contact_email", "").strip() or session.get("user_email", "recruiting@company.example.com")

        # Enforce strict employment type
        if employment_type not in ALLOWED_EMPLOYMENT_TYPES:
            employment_type = "Full-time"

        responsibilities_raw = request.form.get("responsibilities", "").strip()
        responsibilities = [r.strip("- ") for r in responsibilities_raw.split("\n") if r.strip()]

        if not title or not description:
            flash("Job title and description are mandatory fields.", "error")
            return render_template("create_job.html")

        if salary_error:
            flash(salary_error, "error")
            return render_template("create_job.html")

        salary_display = salary_display or f"₹{salary_min:,} - ₹{salary_max:,} / yr"

        if not is_valid_email(contact_email):
            flash("Please enter a valid contact email address.", "error")
            return render_template("create_job.html")

        required_skills = [s.strip() for s in skills_str.split(",") if s.strip()]

        new_job = {
            "id": f"JOB-{uuid.uuid4().hex[:5].upper()}",
            "title": title,
            "company": company,
            "location": location or "Remote / Flexible",
            "employment_type": employment_type,
            "work_arrangement": work_arrangement,
            "experience_level": experience_level,
            "salary_min": salary_min,
            "salary_max": salary_max,
            "salary_currency": "INR",
            "salary_display": salary_display,
            "description": description,
            "responsibilities": responsibilities or ["Contribute to sprint goals and team deliverables."],
            "required_skills": required_skills or ["Software Engineering"],
            "preferred_skills": [],
            "education": "Relevant Bachelor's Degree or Equivalent Experience",
            "application_instructions": "Apply via official link or submit an application through the portal.",
            "application_url": application_url if application_url else None,
            "posted_date": datetime.date.today().strftime("%Y-%m-%d"),
            "contact_email": contact_email,
            "employer_id": session.get("user_id"),
            "created_by": company
        }

        RUNTIME_JOBS.insert(0, new_job)
        save_jobs_data()

        flash(f"Job listing '{title}' has been successfully published!", "success")
        return redirect(url_for("job_detail", job_id=new_job["id"]))

    return render_template("create_job.html")


@app.route("/jobs/<job_id>/edit", methods=["GET", "POST"])
@login_required
@role_required(["employer", "admin"])
def edit_job(job_id):
    """Edit an existing job listing."""
    job = find_job_for_viewer(job_id)
    if not job:
        abort(404)

    # Permission check: must be job owner or admin
    if not can_manage_job(job):
        abort(403)

    if request.method == "POST":
        title = request.form.get("title", "").strip()
        company = request.form.get("company", "").strip()
        location = request.form.get("location", "").strip()
        employment_type = request.form.get("employment_type", "Full-time")
        work_arrangement = request.form.get("work_arrangement", "Hybrid")
        experience_level = request.form.get("experience_level", "Mid-level")
        salary_min, salary_max, salary_error = parse_salary_values(request.form)
        salary_display = request.form.get("salary_display", "").strip()
        description = request.form.get("description", "").strip()
        skills_str = request.form.get("required_skills", "").strip()
        application_url = request.form.get("application_url", "").strip()
        contact_email = request.form.get("contact_email", "").strip()

        if employment_type not in ALLOWED_EMPLOYMENT_TYPES:
            employment_type = "Full-time"

        responsibilities_raw = request.form.get("responsibilities", "").strip()
        responsibilities = [r.strip("- ") for r in responsibilities_raw.split("\n") if r.strip()]

        if not title or not description:
            flash("Title and description are mandatory.", "error")
            return render_template("edit_job.html", job=job)

        if salary_error:
            flash(salary_error, "error")
            return render_template("edit_job.html", job=job)

        if contact_email and not is_valid_email(contact_email):
            flash("Please enter a valid contact email address.", "error")
            return render_template("edit_job.html", job=job)

        job["title"] = title
        job["company"] = company or job.get("company")
        job["location"] = location or job.get("location")
        job["employment_type"] = employment_type
        job["work_arrangement"] = work_arrangement
        job["experience_level"] = experience_level
        job["salary_min"] = salary_min
        job["salary_max"] = salary_max
        job["salary_display"] = salary_display or f"₹{salary_min:,} - ₹{salary_max:,} / yr"
        job["description"] = description
        job["responsibilities"] = responsibilities or job.get("responsibilities", [])
        job["required_skills"] = [s.strip() for s in skills_str.split(",") if s.strip()]
        job["application_url"] = application_url if application_url else None
        job["contact_email"] = contact_email or job.get("contact_email")

        save_jobs_data()
        flash(f"Job '{title}' was updated successfully.", "success")
        return redirect(url_for("job_detail", job_id=job["id"]))

    return render_template("edit_job.html", job=job)


@app.route("/jobs/<job_id>/delete", methods=["POST"])
@login_required
@role_required(["employer", "admin"])
def delete_job(job_id):
    """Delete a job listing."""
    global RUNTIME_JOBS
    job = find_job_for_viewer(job_id)
    if not job:
        abort(404)

    if not can_manage_job(job):
        abort(403)

    with _DATA_LOCK:
        RUNTIME_JOBS = [j for j in RUNTIME_JOBS if str(j.get("id")) != str(job_id)]
        save_jobs_data()

    flash(f"Listing '{job.get('title')}' has been deleted.", "info")
    if session.get("user_role") == "admin":
        return redirect(url_for("dashboard_admin"))
    return redirect(url_for("dashboard_employer"))


@app.route("/employer/application/<app_id>/status", methods=["POST"])
@login_required
@role_required(["employer", "admin"])
def update_application_status(app_id):
    """Employer updates candidate review status."""
    new_status = request.form.get("status")
    allowed_statuses = {"Submitted", "Under Review", "Shortlisted", "Rejected"}

    app_item = next((a for a in RUNTIME_APPLICATIONS if a.get("id") == app_id), None)
    if not app_item:
        abort(404)

    # Only the owner of the job the application belongs to (or an admin) may change its status
    job = next((j for j in RUNTIME_JOBS if str(j.get("id")) == str(app_item.get("job_id"))), None)
    if session.get("user_role") != "admin" and not (job and can_manage_job(job)):
        abort(403)

    if new_status not in allowed_statuses:
        flash("Invalid application status.", "error")
        return redirect(url_for("dashboard_employer"))

    with _DATA_LOCK:
        previous_status = app_item.get("status")
        app_item["status"] = new_status
        if not save_applications_data():
            app_item["status"] = previous_status
            flash("The status could not be saved. Please try again.", "error")
            return redirect(url_for("dashboard_employer"))

    flash(f"Candidate application {app_id} status updated to '{new_status}'.", "success")
    return redirect(url_for("dashboard_employer"))


# ==============================================================================
# Admin Actions
# ==============================================================================

@app.route("/admin/job/<job_id>/toggle-visibility", methods=["POST"])
@login_required
@role_required(["admin"])
def admin_toggle_visibility(job_id):
    """Toggle public visibility for a listing."""
    job = next((j for j in RUNTIME_JOBS if str(j.get("id")) == str(job_id)), None)
    if not job:
        abort(404)

    job["is_hidden"] = not job.get("is_hidden", False)
    save_jobs_data()
    status_label = "hidden from public view" if job["is_hidden"] else "restored to active listings"
    flash(f"Listing '{job.get('title')}' was {status_label}.", "info")
    return redirect(url_for("dashboard_admin"))


@app.route("/admin/user/<user_id>/delete", methods=["POST"])
@login_required
@role_required(["admin"])
def admin_delete_user(user_id):
    """Admin removes a user account."""
    global RUNTIME_USERS
    target = next((u for u in RUNTIME_USERS if u.get("id") == user_id), None)
    if not target:
        abort(404)

    if target.get("role") == "admin":
        flash("Cannot delete primary administrator account.", "error")
        return redirect(url_for("dashboard_admin"))

    RUNTIME_USERS = [u for u in RUNTIME_USERS if u.get("id") != user_id]
    save_users_data()
    flash(f"User account for '{target.get('name')}' ({target.get('email')}) has been deleted.", "info")
    return redirect(url_for("dashboard_admin"))


# ==============================================================================
# Utility & API Routes
# ==============================================================================

@app.route("/api/toggle-save", methods=["POST"])
def toggle_save():
    """Toggle bookmark in session storage."""
    data = request.get_json(silent=True) or {}
    job_id = str(data.get("job_id", "")).strip()

    if not job_id:
        return jsonify({"success": False, "error": "Missing job_id"}), 400

    saved = set(session.get("saved_jobs", []))

    if not find_job_for_viewer(job_id):
        # Unknown (or hidden) job: never report a successful bookmark.
        # A stale bookmark that is already saved may still be cleared.
        if job_id in saved:
            saved.remove(job_id)
            session["saved_jobs"] = list(saved)
            return jsonify({"success": True, "is_saved": False, "count": len(saved)})
        return jsonify({"success": False, "error": "Job not found"}), 404

    if job_id in saved:
        saved.remove(job_id)
        is_saved = False
    else:
        saved.add(job_id)
        is_saved = True

    session["saved_jobs"] = list(saved)
    return jsonify({"success": True, "is_saved": is_saved, "count": len(saved)})


# ==============================================================================
# Error Handlers
# ==============================================================================

@app.errorhandler(403)
def forbidden_error(e):
    return render_template("errors/403.html"), 403


@app.errorhandler(404)
def not_found_error(e):
    return render_template("errors/404.html"), 404


@app.errorhandler(500)
def internal_server_error(e):
    return (
        "<!doctype html><html><head><title>500 Internal Server Error</title>"
        "<link rel='stylesheet' href='/static/css/style.css'></head><body>"
        "<div class='container' style='max-width: 650px; text-align: center; padding: 5rem 1.5rem;'>"
        "<div style='font-size: 4rem; font-weight: 800; color: #1e293b; line-height: 1; margin-bottom: 1rem;'>500</div>"
        "<h1 style='font-size: 2rem; margin-bottom: 0.75rem;'>Internal Server Error</h1>"
        "<p style='color: #64748b; margin-bottom: 2rem;'>An unexpected error occurred. Please try again later.</p>"
        "<a href='/' class='btn btn-primary'>Return to Home</a></div></body></html>"
    ), 500


# WSGI application alias for serverless deployments (Vercel, Gunicorn)
application = app


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="127.0.0.1", port=port, debug=is_development())
