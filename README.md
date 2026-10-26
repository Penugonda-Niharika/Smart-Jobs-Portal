# Smart Job Portal

A Flask job portal with candidate and employer accounts, administrator moderation, JSON file persistence, and local career tools. The repository includes 32 sample job listings, resume parsing, TF-IDF text similarity, required-skill matching, recommendations, career roadmaps, interview preparation, and heuristic listing-risk checks.

## Features

- Search and filter listings by keyword, location, experience, salary, employment type, and work arrangement. The supported employment types are **Full-time** and **Part-time**. Work arrangements are Remote, Hybrid, and On-site.
- Candidate accounts can apply for jobs, track application status, update their profile, and save listings in browser local storage synchronized with the server session.
- Employer accounts can create, edit, and delete their own jobs and review applications to those jobs.
- Administrators can moderate listings, review platform metrics, and remove user accounts.
- Resume uploads accept text-based PDF and TXT files up to 4 MiB. Files are processed in memory and are not saved.
- The resume analyzer reports required-skill match percentage separately from TF-IDF text similarity. Both are heuristic signals, not hiring predictions.
- Skill recommendations, career roadmaps, interview question sets, and suspicious-listing checks are local features.

## Requirements

- Python 3.10–3.12
- Packages in `requirements.txt` (Flask, Flask-WTF, scikit-learn, pypdf, python-dotenv, and Gunicorn)

## Local setup

Create and activate a virtual environment, then install the project requirements:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

For Command Prompt, activate with `.venv\Scripts\activate.bat`. On macOS or Linux, use `python3 -m venv .venv`, `source .venv/bin/activate`, then `python -m pip install -r requirements.txt`.

Copy `.env.example` to `.env` and replace the placeholder values. For local development, explicitly opt in to debug mode in the launching process:

```powershell
$env:APP_ENV = "development"
python app.py
```

`APP_ENV=development` must be set in the process environment; a `.env` value, `FLASK_ENV`, and `FLASK_DEBUG` do not enable debug mode. Development may generate a random session key if `SECRET_KEY` is blank; sessions then expire on each process restart. Sample seeker and employer accounts are created only in development when missing. A passwordless administrator profile is seeded only when a strong `ADMIN_PASSWORD` is configured; example or weak values leave administrator login disabled. To sign in as the administrator, set a strong `ADMIN_PASSWORD` in `.env` or the process environment and use `admin@smartjobportal.com`.

Open <http://127.0.0.1:5000>.

## Configuration

| Variable | Purpose |
| --- | --- |
| `SECRET_KEY` | Flask session signing key. Production requires at least 32 characters and rejects blank, placeholder, and obviously low-entropy values. Generate it with `python -c "import secrets; print(secrets.token_hex(32))"`. |
| `ADMIN_PASSWORD` | Required in production; at least 16 characters and not a placeholder or obvious weak value. It is checked at startup and is the only accepted password for administrator accounts. |
| `APP_ENV` | Set to `development` in the process environment to enable the local development server debugger and development seed users. Unset or any other value means production mode. |
| `PORT` | Optional local port; defaults to `5000`. |
| `SMART_JOB_PORTAL_DATA_DIR` | Optional data directory override, useful for isolated tests or a mounted persistent volume. |

Do not put production secrets in source control or a distributable archive. `.env.example` contains placeholders only. A `.env` file cannot enable debug mode.

## Persistence and deployment limits

Users, jobs, and applications are stored as JSON in `data/users.json`, `data/jobs.json`, and `data/applications.json`. Writes use temporary files and atomic replacement. Existing data is loaded at startup; malformed JSON is copied to a timestamped `.corrupt-*` backup before the app continues with an empty collection for that file. Keep backups and persistent storage for real deployments.

Fresh distributable archives omit `data/users.json` and `data/applications.json` because they contain runtime account/application records. The app starts without those files and creates them when new records are saved. The included `data/jobs.json` is the sample listing catalog.

The process also keeps these records in memory. The thread lock protects writes within one process, but it does not coordinate multiple Gunicorn worker processes or multiple application instances. Run a **single worker** with a persistent writable data directory. This file-based design is not suitable for multi-worker or horizontally scaled production use without replacing the storage layer with a transactional shared database. Ephemeral hosting filesystems can lose data on restart or redeploy. No email delivery, external resume storage, or OCR is implemented.

> **Vercel Serverless Deployment Note:**
> On Vercel, the filesystem is read-only except for `/tmp`. Writes (new jobs, signups, applications) fall back to `/tmp` so the serverless function operates without crashing. However, data in `/tmp` is ephemeral and does not persist across cold starts or between multiple isolated lambda instances. For full production persistence on Vercel, migrate the JSON storage layer to a hosted database such as **Supabase**, **Neon (Serverless Postgres)**, or **MongoDB Atlas**.

The included `Procfile` starts Gunicorn. Before deploying, configure `SECRET_KEY` and `ADMIN_PASSWORD` as platform secrets, provide persistent writable storage for `data/`, use one worker, and ensure the hosting platform does not inject a development process mode. Flask debug remains off by default. This project does not include database migrations, backups, monitoring, or production-grade multi-instance coordination.

## Tests

Run the suites from the project directory:

```powershell
python -m unittest -v test_portal test_regressions
```

Both suites import `portal_test_env.py`, which copies the shipped data into a temporary test directory. Tests do not write to the project’s JSON files. Flask-WTF CSRF protection is exercised with real CSRF tokens in the regression suite. The suites cover access control, ownership, hidden jobs, email and salary validation, duplicate applications, bookmarks, persistence across process restarts, corrupt JSON recovery, uploads and size limits, configuration, and concurrent applications.

## Project structure

```text
app.py                       Flask application, routes, access checks, and JSON storage
requirements.txt             Runtime and test dependencies
.env.example                 Placeholder-only configuration example
Procfile                     Gunicorn launch command
data/jobs.json               Sample listings
data/skills_map.json         Career roadmap content
data/interview_questions.json Interview preparation content
services/                    Recommendation, resume, career, and risk logic
templates/                   Flask/Jinja pages
static/                      Stylesheet and browser interactions
test_portal.py               Main route and feature tests
test_regressions.py          Security, persistence, and regression tests
```
