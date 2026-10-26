"""
Regression tests for: email validation, hidden jobs, salary validation, bookmark validation,
SECRET_KEY / debug handling, resume scoring, application persistence and access control.

All tests run against a temporary copy of data/ (see portal_test_env.py).
"""

import portal_test_env as env  # must be first: redirects the app to temporary data
import io
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path

import app as app_module
from services.resume_analyzer import analyze_resume_against_job, extract_text_from_file
from werkzeug.datastructures import FileStorage

PROJECT_DIR = env.PROJECT_DIR
APPLICATIONS_FILE = lambda: env.DATA_DIR / "applications.json"  # noqa: E731


def read_applications_file():
    return json.loads(APPLICATIONS_FILE().read_text(encoding="utf-8"))


def make_pdf(text: str) -> bytes:
    """Build a minimal valid single-page PDF containing `text` (no extra libraries needed)."""
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode("latin-1")
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n".encode() + b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


class PortalCase(unittest.TestCase):
    """Base class: fresh data + fresh in-memory state before every test."""

    def setUp(self):
        env.fresh_work_dir()
        app_module.app.config.update(TESTING=True, WTF_CSRF_ENABLED=False)
        app_module.load_data()

    # ---- helpers
    def login(self, email, password):
        client = app_module.app.test_client()
        resp = client.post("/login", data={"email": email, "password": password})
        self.assertEqual(resp.status_code, 302, f"login failed for {email}")
        return client

    def seeker(self):
        return self.login("seeker@example.com", "seeker123")

    def apex(self):
        return self.login("employer@apexcloud.example.com", "employer123")

    def admin(self):
        return self.login("admin@smartjobportal.com", os.environ["ADMIN_PASSWORD"])

    def signup(self, email, role="seeker", company="", name="Test User", password="password123"):
        client = app_module.app.test_client()
        client.post("/signup", data={
            "name": name, "email": email, "role": role, "company_name": company,
            "password": password, "confirm_password": password,
        })
        return client

    _seeker_counter = 0

    def new_seeker(self, label="fresh"):
        """Register a brand-new seeker (the demo account already owns the seeded demo application)."""
        PortalCase._seeker_counter += 1
        email = f"{label}{PortalCase._seeker_counter}@example.com"
        client = self.signup(email, name=f"Fresh {label.title()}")
        return client, email

    def apply_as(self, client, email, job_id, **extra):
        return self.apply(client, job_id, applicant_name="Fresh Seeker", applicant_email=email, **extra)

    def apex_job(self):
        job = next((j for j in app_module.RUNTIME_JOBS if j.get("employer_id") == "USR-102"), None)
        self.assertIsNotNone(job, "legacy Apex job should be linked to its employer")
        return job

    def other_company_job(self):
        return next(j for j in app_module.RUNTIME_JOBS if j.get("employer_id") != "USR-102")

    def job_form(self, **overrides):
        form = {
            "title": "Regression Role", "company": "Apex Cloud Systems", "location": "Hyderabad, India",
            "employment_type": "Full-time", "work_arrangement": "Hybrid", "experience_level": "Mid-level",
            "salary_min": "600000", "salary_max": "900000", "description": "A description of the role.",
            "required_skills": "Python, Flask", "responsibilities": "Build things",
            "application_url": "", "contact_email": "jobs@apex.example.com",
        }
        form.update(overrides)
        return form

    def apply(self, client, job_id, **extra):
        data = {"applicant_name": "Alex Seeker", "applicant_email": "seeker@example.com", "cover_note": "Hi"}
        data.update(extra)
        return client.post(f"/jobs/{job_id}/apply", data=data)

    def find_job(self, title):
        return next((j for j in app_module.RUNTIME_JOBS if j["title"] == title), None)


# ==============================================================================
# Resume scoring
# ==============================================================================
class ResumeScoreTests(PortalCase):
    SIX = ["Python", "Machine Learning", "scikit-learn", "PyTorch", "NLP", "Pandas"]
    SIX_RESUME = "Python developer. Machine Learning, scikit-learn, Pandas, NLP projects with PyTorch."

    def test_all_six_skills_scores_100_not_13_5(self):
        r = analyze_resume_against_job(self.SIX_RESUME, "Machine Learning Engineer role", self.SIX)
        self.assertEqual(r["similarity_score"], 100.0)
        self.assertEqual(r["skill_match_percentage"], 100.0)
        self.assertEqual(len(r["matched_skills"]), 6)
        self.assertEqual(r["missing_skills"], [])
        self.assertEqual(r["score_basis"], "skill_match")

    def test_partial_match_is_matched_over_required(self):
        r = analyze_resume_against_job("Python and Flask and SQL work", "Backend job", ["Python", "Flask", "SQL", "Docker"])
        self.assertEqual(r["similarity_score"], 75.0)
        self.assertEqual(r["missing_skills"], ["Docker"])

    def test_no_matching_skills_is_zero(self):
        r = analyze_resume_against_job("Chef with ten years of Italian cooking", "Backend job", ["Python", "Flask"])
        self.assertEqual(r["similarity_score"], 0.0)
        self.assertEqual(r["matched_skills"], [])
        self.assertEqual(r["missing_skills"], ["Python", "Flask"])

    def test_headline_always_consistent_with_displayed_skills(self):
        cases = [
            ("Python Flask", ["Python", "Flask", "SQL"]),
            ("Python", ["Python", "Flask", "SQL", "Docker", "Git"]),
            ("nothing relevant here", ["Python"]),
            ("Python Flask SQL Docker Git", ["Python", "Flask", "SQL", "Docker", "Git"]),
        ]
        for resume, skills in cases:
            r = analyze_resume_against_job(resume, "A job", skills)
            expected = round(len(r["matched_skills"]) / len(skills) * 100, 1)
            self.assertEqual(r["similarity_score"], expected, (resume, skills))
            self.assertEqual(len(r["matched_skills"]) + len(r["missing_skills"]), len(skills))

    def test_empty_inputs_are_handled(self):
        for resume, job in [("", "Some job"), ("   ", "Some job"), ("Python dev", ""), ("", "")]:
            r = analyze_resume_against_job(resume, job, ["Python"])
            self.assertEqual(r["similarity_score"], 0)
            self.assertEqual(r["score_basis"], "none")
            self.assertTrue(r["suggestions"])

    def test_no_explicit_skills_uses_skills_found_in_description(self):
        r = analyze_resume_against_job("I use Python and Docker daily", "Needs Python, Docker and Kubernetes experience")
        self.assertEqual(r["score_basis"], "skill_match")
        self.assertEqual(sorted(r["matched_skills"]), ["Docker", "Python"])
        self.assertEqual(r["missing_skills"], ["Kubernetes"])
        self.assertEqual(r["similarity_score"], 66.7)

    def test_no_identifiable_skills_falls_back_to_text_similarity(self):
        r = analyze_resume_against_job("Experienced storyteller and communicator", "Seeking a friendly storyteller")
        self.assertEqual(r["score_basis"], "text_similarity")
        self.assertEqual(r["similarity_score"], r["tfidf_similarity"])

    def test_skill_outside_builtin_taxonomy_still_matches(self):
        r = analyze_resume_against_job("Built pipelines with Kafka and Python", "Streaming job", ["Kafka", "Python"])
        self.assertEqual(r["similarity_score"], 100.0)

    def test_similar_name_does_not_false_match(self):
        r = analyze_resume_against_job("I write JavaScript", "Job", ["Java"])
        self.assertEqual(r["matched_skills"], [])

    # ---- upload handling through the real route (PDF/TXT still supported)
    def post_resume(self, client, filename, payload, **extra):
        data = {"resume_file": (io.BytesIO(payload), filename)}
        data.update(extra)
        return client.post("/resume-analyzer", data=data, content_type="multipart/form-data")

    def test_route_txt_upload_shows_skill_match_headline(self):
        job = next(j for j in app_module.RUNTIME_JOBS if j["title"] == "Machine Learning Engineer")
        resp = self.post_resume(self.seeker(), "cv.txt", self.SIX_RESUME.encode(), job_id=job["id"])
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"100.0%", resp.data)

    def test_route_pdf_upload_works(self):
        pdf = make_pdf("Python Flask SQL Docker developer")
        text, err = extract_text_from_file(FileStorage(stream=io.BytesIO(pdf), filename="cv.pdf"))
        self.assertIsNone(err)
        self.assertIn("Python", text)
        resp = self.post_resume(self.seeker(), "cv.pdf", pdf, custom_description="Python, Flask and SQL engineer")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"100.0%", resp.data)

    def test_route_invalid_inputs_show_errors_not_crashes(self):
        client = self.seeker()
        job_id = app_module.RUNTIME_JOBS[0]["id"]
        cases = [
            (self.post_resume(client, "cv.docx", b"x", job_id=job_id), b"Unsupported file extension"),
            (self.post_resume(client, "empty.txt", b"", job_id=job_id), b"empty"),
            (self.post_resume(client, "bad.pdf", b"%PDF-1.4 garbage", job_id=job_id), b"PDF"),
            (client.post("/resume-analyzer", data={"job_id": job_id}), b"select a PDF or TXT"),
            (self.post_resume(client, "cv.txt", b"Python"), b"target job"),
        ]
        for resp, expected in cases:
            self.assertEqual(resp.status_code, 200)
            self.assertIn(expected, resp.data)

    def test_oversize_upload_rejected(self):
        resp = self.post_resume(self.seeker(), "big.txt", b"a" * (5 * 1024 * 1024))
        self.assertEqual(resp.status_code, 413)


# ==============================================================================
# Email validation
# ==============================================================================
class EmailValidationTests(PortalCase):
    BAD = ["notanemail", "a@b", "a@@b.com", "a b@c.com", "@x.com", "x@.com", "x@com.", "x@y..com", "plain.text"]

    def test_invalid_emails_rejected_at_signup(self):
        before = len(app_module.RUNTIME_USERS)
        for bad in self.BAD:
            resp = app_module.app.test_client().post("/signup", data={
                "name": "T", "email": bad, "role": "seeker",
                "password": "password123", "confirm_password": "password123"})
            self.assertEqual(resp.status_code, 200, bad)
            self.assertIn(b"valid email", resp.data, bad)
        self.assertEqual(len(app_module.RUNTIME_USERS), before)

    def test_valid_email_accepted_and_stored_lowercase(self):
        self.signup("New.Person+tag@Example.co.in")
        user = next((u for u in app_module.RUNTIME_USERS if u["email"] == "new.person+tag@example.co.in"), None)
        self.assertIsNotNone(user)

    def test_duplicate_email_still_case_insensitive(self):
        self.signup("dupe@example.com")
        count = len(app_module.RUNTIME_USERS)
        self.signup("DUPE@Example.COM")
        self.assertEqual(len(app_module.RUNTIME_USERS), count)

    def test_invalid_applicant_email_rejected_on_apply(self):
        job = app_module.RUNTIME_JOBS[2]
        before = len(app_module.RUNTIME_APPLICATIONS)
        self.apply(self.seeker(), job["id"], applicant_email="notanemail")
        self.assertEqual(len(app_module.RUNTIME_APPLICATIONS), before)

    def test_invalid_job_contact_email_rejected(self):
        before = len(app_module.RUNTIME_JOBS)
        resp = self.apex().post("/jobs/create", data=self.job_form(title="Bad Contact", contact_email="not-an-email"))
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"valid contact email", resp.data)
        self.assertEqual(len(app_module.RUNTIME_JOBS), before)


# ==============================================================================
# Hidden jobs
# ==============================================================================
class HiddenJobTests(PortalCase):
    def setUp(self):
        super().setUp()
        self.job = self.apex_job()
        self.admin_client = self.admin()
        resp = self.admin_client.post(f"/admin/job/{self.job['id']}/toggle-visibility")
        self.assertEqual(resp.status_code, 302)
        self.assertTrue(self.job.get("is_hidden"))

    def test_hidden_job_detail_404_for_anonymous_seeker_and_owner(self):
        url = f"/jobs/{self.job['id']}"
        self.assertEqual(app_module.app.test_client().get(url).status_code, 404)
        self.assertEqual(self.seeker().get(url).status_code, 404)
        self.assertEqual(self.apex().get(url).status_code, 404)

    def test_hidden_job_not_in_public_surfaces(self):
        needle = self.job["id"].encode()
        anon = app_module.app.test_client()
        for path in ["/jobs", "/", "/recommendations", "/resume-analyzer"]:
            self.assertNotIn(b"/jobs/" + needle, anon.get(path).data, path)

    def test_cannot_apply_to_hidden_job(self):
        before = len(app_module.RUNTIME_APPLICATIONS)
        resp = self.apply(self.seeker(), self.job["id"])
        self.assertEqual(resp.status_code, 404)
        self.assertEqual(len(app_module.RUNTIME_APPLICATIONS), before)

    def test_cannot_bookmark_hidden_job(self):
        resp = self.seeker().post("/api/toggle-save", json={"job_id": self.job["id"]})
        self.assertEqual(resp.status_code, 404)
        self.assertFalse(resp.get_json()["success"])

    def test_owner_cannot_edit_or_delete_hidden_job_via_direct_url(self):
        employer = self.apex()
        self.assertEqual(employer.get(f"/jobs/{self.job['id']}/edit").status_code, 404)
        self.assertEqual(employer.post(f"/jobs/{self.job['id']}/delete").status_code, 404)
        self.assertIn(self.job, app_module.RUNTIME_JOBS)

    def test_admin_can_still_view_edit_and_restore(self):
        self.assertEqual(self.admin_client.get(f"/jobs/{self.job['id']}").status_code, 200)
        self.assertEqual(self.admin_client.get(f"/jobs/{self.job['id']}/edit").status_code, 200)
        self.admin_client.post(f"/admin/job/{self.job['id']}/toggle-visibility")
        self.assertFalse(self.job.get("is_hidden"))
        self.assertEqual(app_module.app.test_client().get(f"/jobs/{self.job['id']}").status_code, 200)

    def test_hidden_flag_persists_to_disk(self):
        on_disk = json.loads((env.DATA_DIR / "jobs.json").read_text(encoding="utf-8"))
        self.assertTrue(next(j for j in on_disk if j["id"] == self.job["id"]).get("is_hidden"))


# ==============================================================================
# Salary validation
# ==============================================================================
class SalaryValidationTests(PortalCase):
    def test_create_and_edit_forms_explain_salary_range_before_submission(self):
        client = self.apex()
        create_page = client.get("/jobs/create").get_data(as_text=True)
        self.assertIn('data-salary-validation', create_page)
        client.post("/jobs/create", data=self.job_form(title="Salary Form"))
        job = self.find_job("Salary Form")
        edit_page = client.get(f"/jobs/{job['id']}/edit").get_data(as_text=True)
        self.assertIn('data-salary-validation', edit_page)
        script = (PROJECT_DIR / "static" / "js" / "main.js").read_text(encoding="utf-8")
        self.assertIn("setCustomValidity", script)
        self.assertIn("Maximum salary must be greater than or equal to minimum salary.", script)

    def test_create_rejects_min_greater_than_max(self):
        resp = self.apex().post("/jobs/create", data=self.job_form(title="Bad Salary", salary_min="900000", salary_max="100000"))
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"Minimum salary cannot be greater", resp.data)
        self.assertIsNone(self.find_job("Bad Salary"))

    def test_create_accepts_valid_equal_missing_and_single_values(self):
        client = self.apex()
        cases = {
            "Sal Valid": ("600000", "900000"), "Sal Equal": ("500000", "500000"),
            "Sal Missing": ("", ""), "Sal MinOnly": ("500000", ""), "Sal MaxOnly": ("", "500000"),
        }
        for title, (lo, hi) in cases.items():
            resp = client.post("/jobs/create", data=self.job_form(title=title, salary_min=lo, salary_max=hi))
            self.assertEqual(resp.status_code, 302, title)
            self.assertIsNotNone(self.find_job(title), title)

    def test_create_rejects_negative_salary(self):
        self.apex().post("/jobs/create", data=self.job_form(title="Negative", salary_min="-5", salary_max="100"))
        self.assertIsNone(self.find_job("Negative"))

    def test_create_rejects_non_integer_salary(self):
        resp = self.apex().post("/jobs/create", data=self.job_form(title="Non Integer", salary_min="six hundred"))
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"whole numbers", resp.data)
        self.assertIsNone(self.find_job("Non Integer"))

    def test_edit_rejects_min_greater_than_max_and_keeps_job(self):
        client = self.apex()
        client.post("/jobs/create", data=self.job_form(title="Editable"))
        job = self.find_job("Editable")
        resp = client.post(f"/jobs/{job['id']}/edit", data=self.job_form(title="Edited", salary_min="800", salary_max="100"))
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"Minimum salary cannot be greater", resp.data)
        self.assertEqual(job["title"], "Editable")
        self.assertEqual(job["salary_min"], 600000)

    def test_edit_accepts_valid_range(self):
        client = self.apex()
        client.post("/jobs/create", data=self.job_form(title="Editable2"))
        job = self.find_job("Editable2")
        resp = client.post(f"/jobs/{job['id']}/edit", data=self.job_form(title="Edited2", salary_min="1", salary_max="2"))
        self.assertEqual(resp.status_code, 302)
        self.assertEqual(job["title"], "Edited2")


# ==============================================================================
# Bookmarks
# ==============================================================================
class BookmarkTests(PortalCase):
    def test_nonexistent_job_rejected(self):
        client = self.seeker()
        resp = client.post("/api/toggle-save", json={"job_id": "JOB-DOES-NOT-EXIST"})
        self.assertEqual(resp.status_code, 404)
        self.assertFalse(resp.get_json()["success"])
        with client.session_transaction() as sess:
            self.assertEqual(sess.get("saved_jobs", []), [])

    def test_missing_job_id_is_400(self):
        resp = self.seeker().post("/api/toggle-save", json={})
        self.assertEqual(resp.status_code, 400)
        self.assertFalse(resp.get_json()["success"])

    def test_valid_job_toggles_on_and_off(self):
        client = self.seeker()
        job_id = app_module.RUNTIME_JOBS[0]["id"]
        on = client.post("/api/toggle-save", json={"job_id": job_id}).get_json()
        off = client.post("/api/toggle-save", json={"job_id": job_id}).get_json()
        self.assertEqual((on["success"], on["is_saved"], on["count"]), (True, True, 1))
        self.assertEqual((off["success"], off["is_saved"], off["count"]), (True, False, 0))

    def test_stale_bookmark_can_be_cleared_but_never_added(self):
        client = self.seeker()
        with client.session_transaction() as sess:
            sess["saved_jobs"] = ["JOB-GONE"]
        resp = client.post("/api/toggle-save", json={"job_id": "JOB-GONE"})
        self.assertEqual(resp.status_code, 200)
        self.assertFalse(resp.get_json()["is_saved"])
        self.assertEqual(client.post("/api/toggle-save", json={"job_id": "JOB-GONE"}).status_code, 404)


# ==============================================================================
# SECRET_KEY and debug mode (run in fresh processes: configuration is read at import)
# ==============================================================================
class ConfigurationTests(unittest.TestCase):
    def run_app_import(self, extra_env, code="import app; print('DEBUG=%s' % app.app.debug, 'KEY_LEN=%d' % len(app.app.config['SECRET_KEY']), 'DEV=%s' % app.is_development())"):
        data_dir = env.copy_snapshot_to(Path(tempfile.mkdtemp(prefix="sjp-cfg-", dir=env.TEMP_ROOT)))
        child_env = dict(os.environ)
        for k in ("APP_ENV", "FLASK_ENV", "FLASK_DEBUG", "SECRET_KEY", "ADMIN_PASSWORD"):
            child_env.pop(k, None)
        child_env["APP_ENV"] = "production"
        child_env["SMART_JOB_PORTAL_DATA_DIR"] = str(data_dir)
        child_env.update(extra_env)
        return subprocess.run([sys.executable, "-c", code], cwd=str(PROJECT_DIR), env=child_env,
                              capture_output=True, text=True, timeout=120)

    def test_production_without_secret_key_fails_safely(self):
        for value in (None, "", "   "):
            overrides = {"ADMIN_PASSWORD": "test-admin-password-for-prod"}
            if value is not None:
                overrides["SECRET_KEY"] = value
            proc = self.run_app_import(overrides)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("SECRET_KEY", proc.stderr)

    def test_unset_flask_env_is_treated_as_production(self):
        # In-process: the project's own .env (FLASK_ENV=development) would mask an unset variable in a subprocess.
        from unittest import mock
        clean = {k: v for k, v in os.environ.items() if k not in ("APP_ENV", "FLASK_ENV", "FLASK_DEBUG", "SECRET_KEY", "ADMIN_PASSWORD")}
        with mock.patch.dict(os.environ, clean, clear=True), mock.patch.object(app_module, "PROCESS_APP_ENV", "production"):
                self.assertFalse(app_module.is_development())
                with self.assertRaises(RuntimeError):
                    app_module.resolve_secret_key()
        with mock.patch.dict(os.environ, dict(clean, FLASK_DEBUG="1"), clear=True), mock.patch.object(app_module, "PROCESS_APP_ENV", "production"):
                self.assertFalse(app_module.is_development())
        with mock.patch.dict(os.environ, dict(clean, SECRET_KEY=__import__("secrets").token_hex(32)), clear=True), mock.patch.object(app_module, "PROCESS_APP_ENV", "production"):
                self.assertFalse(app_module.is_development())
                self.assertEqual(app_module.resolve_secret_key(), os.environ["SECRET_KEY"])

    def test_production_rejects_placeholder_secret(self):
        proc = self.run_app_import({"SECRET_KEY": "replace-with-a-secure-random-secret-key-in-production", "ADMIN_PASSWORD": "test-admin-password-for-prod"})
        self.assertNotEqual(proc.returncode, 0)

    def test_production_rejects_low_entropy_secret(self):
        proc = self.run_app_import({"SECRET_KEY": "x" * 64, "ADMIN_PASSWORD": "test-admin-password-for-prod"})
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("SECRET_KEY", proc.stderr)

    def test_production_with_real_secret_starts_with_debug_off(self):
        import secrets
        proc = self.run_app_import({"SECRET_KEY": secrets.token_hex(32), "ADMIN_PASSWORD": secrets.token_urlsafe(24), "FLASK_DEBUG": "1", "FLASK_ENV": "development"},
                                   code="import app; print('DEBUG=%s' % app.app.debug, 'KEY_LEN=%d' % len(app.app.config['SECRET_KEY']), 'DEV=%s' % app.is_development(), 'APPS=%d' % len(app.RUNTIME_APPLICATIONS))")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("DEBUG=False", proc.stdout)
        self.assertIn("DEV=False", proc.stdout)
        self.assertIn("KEY_LEN=64", proc.stdout)
        self.assertIn("APPS=0", proc.stdout)

    def test_production_missing_or_weak_admin_password_fails(self):
        secret = __import__("secrets").token_hex(32)
        for password in (None, "admin123", "short", "replace-with-a-strong-admin-password"):
            env_overrides = {"SECRET_KEY": secret}
            if password is not None:
                env_overrides["ADMIN_PASSWORD"] = password
            proc = self.run_app_import(env_overrides)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("ADMIN_PASSWORD", proc.stderr)

    def test_development_without_secret_uses_random_key_not_hardcoded(self):
        a = self.run_app_import({"APP_ENV": "development", "SECRET_KEY": "", "ADMIN_PASSWORD": "test-admin-password"},
                                code="import app; print(app.app.config['SECRET_KEY'])")
        b = self.run_app_import({"APP_ENV": "development", "SECRET_KEY": "", "ADMIN_PASSWORD": "test-admin-password"},
                                code="import app; print(app.app.config['SECRET_KEY'])")
        self.assertEqual(a.returncode, 0, a.stderr)
        key_a, key_b = a.stdout.strip().splitlines()[-1], b.stdout.strip().splitlines()[-1]
        self.assertEqual(len(key_a), 64)
        self.assertNotEqual(key_a, key_b)
        self.assertNotIn("smart-job-portal-production-secret", key_a)

    def test_development_example_admin_placeholder_is_not_usable(self):
        proc = self.run_app_import({"APP_ENV": "development", "SECRET_KEY": "", "ADMIN_PASSWORD": "replace-with-a-strong-admin-password"},
                                   code="import app; print('ADMIN_SET=%s' % bool(app.app.config['ADMIN_PASSWORD']))")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("ADMIN_SET=False", proc.stdout)

    def test_no_hardcoded_secret_fallback_in_source(self):
        source = (PROJECT_DIR / "app.py").read_text(encoding="utf-8")
        self.assertNotIn("smart-job-portal-production-secret-key", source)

    def test_debug_only_in_development(self):
        proc = self.run_app_import({"APP_ENV": "development", "SECRET_KEY": "k" * 20, "ADMIN_PASSWORD": "test-admin-password"})
        self.assertIn("DEV=True", proc.stdout)
        self.assertIn("APP_ENV", (PROJECT_DIR / "app.py").read_text(encoding="utf-8"))
        source = (PROJECT_DIR / "app.py").read_text(encoding="utf-8")
        self.assertIn("debug=is_development()", source)
        self.assertNotIn("debug=True", source)
        prod = self.run_app_import({"SECRET_KEY": __import__("secrets").token_hex(32), "ADMIN_PASSWORD": __import__("secrets").token_urlsafe(24), "FLASK_ENV": "development", "FLASK_DEBUG": "1"})
        self.assertIn("DEV=False", prod.stdout)


# ==============================================================================
# Application persistence
# ==============================================================================
class ApplicationPersistenceTests(PortalCase):
    def test_applications_file_created_and_seed_loaded(self):
        self.assertTrue(APPLICATIONS_FILE().exists())
        ids = [a["id"] for a in read_applications_file()]
        self.assertIn("APP-1001", ids)

    def test_apply_persists_with_expected_fields(self):
        job = self.apex_job()
        client, email = self.new_seeker()
        resp = self.apply_as(client, email, job["id"], cover_note="My note", portfolio_url="https://x.example.com")
        self.assertEqual(resp.status_code, 302)
        saved = [a for a in read_applications_file() if a["job_id"] == job["id"] and a["applicant_email"] == email]
        self.assertEqual(len(saved), 1)
        rec = saved[0]
        for key in ("id", "job_id", "job_title", "company", "applicant_name", "applicant_email",
                    "cover_note", "portfolio_url", "applied_date", "status"):
            self.assertIn(key, rec)
        self.assertEqual(rec["status"], "Submitted")
        self.assertEqual(rec["cover_note"], "My note")
        self.assertTrue(rec["id"].startswith("APP-"))
        self.assertTrue(rec["applicant_id"].startswith("USR-"))

    def test_status_update_persists(self):
        job = self.apex_job()
        client, email = self.new_seeker()
        self.apply_as(client, email, job["id"])
        app_id = next(a["id"] for a in app_module.RUNTIME_APPLICATIONS if a["job_id"] == job["id"] and a["applicant_email"] == email)
        resp = self.apex().post(f"/employer/application/{app_id}/status", data={"status": "Shortlisted"})
        self.assertEqual(resp.status_code, 302)
        self.assertEqual(next(a for a in read_applications_file() if a["id"] == app_id)["status"], "Shortlisted")

    def test_duplicate_blocked_even_with_changed_email_or_case(self):
        job = self.apex_job()
        client, email = self.new_seeker()
        self.apply_as(client, email, job["id"])
        count = len([a for a in app_module.RUNTIME_APPLICATIONS if a["job_id"] == job["id"]])
        self.apply_as(client, email.upper(), job["id"])
        self.apply_as(client, "another.address@example.com", job["id"])
        self.assertEqual(len([a for a in app_module.RUNTIME_APPLICATIONS if a["job_id"] == job["id"]]), count)
        self.assertEqual(len([a for a in read_applications_file() if a["job_id"] == job["id"]]), count)

    def test_different_seekers_can_apply_to_same_job(self):
        job = self.apex_job()
        first, first_email = self.new_seeker()
        second, second_email = self.new_seeker()
        self.apply_as(first, first_email, job["id"])
        self.apply_as(second, second_email, job["id"])
        mine = [a for a in read_applications_file() if a["job_id"] == job["id"] and a["applicant_email"] in (first_email, second_email)]
        self.assertEqual(len(mine), 2)

    def test_restart_keeps_applications_and_status_and_blocks_duplicates(self):
        data_dir = env.copy_snapshot_to(Path(tempfile.mkdtemp(prefix="sjp-restart-", dir=env.TEMP_ROOT)))
        child_env = dict(os.environ, SMART_JOB_PORTAL_DATA_DIR=str(data_dir), APP_ENV="development",
                         SECRET_KEY="restart-test-secret", ADMIN_PASSWORD="restart-test-admin-password")

        def run(code):
            proc = subprocess.run([sys.executable, "-c", code], cwd=str(PROJECT_DIR), env=child_env,
                                  capture_output=True, text=True, timeout=180)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            return json.loads(proc.stdout.strip().splitlines()[-1])

        first = run('''
import json, app as A
A.app.config.update(TESTING=True, WTF_CSRF_ENABLED=False)
s = A.app.test_client()
s.post("/signup", data={"name": "Restart Person", "email": "restart.person@example.com", "role": "seeker", "password": "password123", "confirm_password": "password123"})
job = next(j for j in A.RUNTIME_JOBS if j.get("employer_id") == "USR-102")
s.post("/jobs/" + job["id"] + "/apply", data={"applicant_name": "Restart Person", "applicant_email": "restart.person@example.com", "cover_note": "persist me"})
rec = next(a for a in A.RUNTIME_APPLICATIONS if a["job_id"] == job["id"] and a["applicant_email"] == "restart.person@example.com")
e = A.app.test_client(); e.post("/login", data={"email": "employer@apexcloud.example.com", "password": "employer123"})
e.post("/employer/application/" + rec["id"] + "/status", data={"status": "Shortlisted"})
print(json.dumps({"id": rec["id"], "job": job["id"]}))
''')
        second = run('''
import json, app as A
A.app.config.update(TESTING=True, WTF_CSRF_ENABLED=False)
APP_ID, JOB_ID = "@@APP@@", "@@JOB@@"
rec = next((a for a in A.RUNTIME_APPLICATIONS if a["id"] == APP_ID), None)
s = A.app.test_client(); s.post("/login", data={"email": "restart.person@example.com", "password": "password123"})
before = len([a for a in A.RUNTIME_APPLICATIONS if a["job_id"] == JOB_ID])
s.post("/jobs/" + JOB_ID + "/apply", data={"applicant_name": "Restart Person", "applicant_email": "restart.person@example.com"})
after = len([a for a in A.RUNTIME_APPLICATIONS if a["job_id"] == JOB_ID])
dash = s.get("/dashboard/seeker").get_data(as_text=True)
e = A.app.test_client(); e.post("/login", data={"email": "employer@apexcloud.example.com", "password": "employer123"})
edash = e.get("/dashboard/employer").get_data(as_text=True)
print(json.dumps({"found": rec is not None, "status": rec and rec["status"], "note": rec and rec["cover_note"],
                  "dup_blocked": before == after, "seeker_sees": "Shortlisted" in dash,
                  "employer_sees": "restart.person@example.com" in edash}))
'''.replace("@@APP@@", first["id"]).replace("@@JOB@@", first["job"]))
        self.assertTrue(second["found"], "application lost after restart")
        self.assertEqual(second["status"], "Shortlisted")
        self.assertEqual(second["note"], "persist me")
        self.assertTrue(second["dup_blocked"])
        self.assertTrue(second["seeker_sees"])
        self.assertTrue(second["employer_sees"])

    def test_existing_empty_list_is_not_reseeded(self):
        APPLICATIONS_FILE().write_text("[]", encoding="utf-8")
        app_module.load_data()
        self.assertEqual(app_module.RUNTIME_APPLICATIONS, [])
        self.assertEqual(read_applications_file(), [])

    def test_malformed_file_is_preserved_not_overwritten_silently(self):
        garbage = '{"this is": not valid json'
        APPLICATIONS_FILE().write_text(garbage, encoding="utf-8")
        app_module.load_data()  # must not crash
        backups = list(env.DATA_DIR.glob("applications.json.corrupt-*"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_text(encoding="utf-8"), garbage)
        # the app keeps working and can write a fresh, valid file afterwards
        job = self.apex_job()
        client, email = self.new_seeker()
        self.assertEqual(self.apply_as(client, email, job["id"]).status_code, 302)
        self.assertEqual(len(read_applications_file()), 1)

    def test_wrong_json_shape_is_treated_as_corrupt(self):
        APPLICATIONS_FILE().write_text('{"id": "APP-1"}', encoding="utf-8")
        app_module.load_data()
        self.assertEqual(len(list(env.DATA_DIR.glob("applications.json.corrupt-*"))), 1)

    def test_malformed_users_and_jobs_files_are_backed_up_too(self):
        (env.DATA_DIR / "users.json").write_text("not json", encoding="utf-8")
        (env.DATA_DIR / "jobs.json").write_text("", encoding="utf-8")
        app_module.load_data()
        self.assertEqual(len(list(env.DATA_DIR.glob("users.json.corrupt-*"))), 1)
        self.assertEqual(len(list(env.DATA_DIR.glob("jobs.json.corrupt-*"))), 1)

    def test_atomic_writes_leave_no_temp_files(self):
        client, email = self.new_seeker()
        self.apply_as(client, email, self.apex_job()["id"])
        self.assertEqual(list(env.DATA_DIR.glob("*.tmp")), [])

    def test_concurrent_applications_all_persist(self):
        job = self.apex_job()
        clients = []
        for i in range(12):
            email = f"concurrent{i}@example.com"
            clients.append((self.signup(email, name=f"Concurrent {i}"), email))
        errors = []

        def go(client, email):
            try:
                self.apply(client, job["id"], applicant_name="C", applicant_email=email)
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

        threads = [threading.Thread(target=go, args=pair) for pair in clients]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])
        saved = [a for a in read_applications_file() if a["job_id"] == job["id"] and a["applicant_email"].startswith("concurrent")]
        self.assertEqual(len(saved), 12)
        self.assertEqual(len({a["id"] for a in saved}), 12)

    def test_save_failure_rolls_back_apply_and_status(self):
        job = self.apex_job()
        original = app_module.write_json_atomic
        app_module.write_json_atomic = lambda path, data: False
        try:
            before = len(app_module.RUNTIME_APPLICATIONS)
            client, email = self.new_seeker()
            resp = self.apply_as(client, email, job["id"])
            self.assertEqual(resp.status_code, 302)
            self.assertEqual(len(app_module.RUNTIME_APPLICATIONS), before)
            seed = app_module.RUNTIME_APPLICATIONS[0]
            seed_job = next(j for j in app_module.RUNTIME_JOBS if j["id"] == seed["job_id"])
            old = seed["status"]
            owner = self.login("employer@apexcloud.example.com", "employer123") if seed_job.get("employer_id") == "USR-102" else self.admin()
            owner.post(f"/employer/application/{seed['id']}/status", data={"status": "Rejected"})
            self.assertEqual(seed["status"], old)
        finally:
            app_module.write_json_atomic = original


# ==============================================================================
# Access control for applications, jobs and dashboards
# ==============================================================================
class AccessControlTests(PortalCase):
    def setUp(self):
        super().setUp()
        self.job = self.apex_job()
        self.seeker_client, self.seeker_email = self.new_seeker("applicant")
        self.apply_as(self.seeker_client, self.seeker_email, self.job["id"], cover_note="visible to owner only")
        self.app_id = next(a["id"] for a in app_module.RUNTIME_APPLICATIONS
                           if a["job_id"] == self.job["id"] and a["applicant_email"] == self.seeker_email)
        self.other_employer = self.signup("other.employer@example.com", role="employer", company="Other Co", name="Other Employer")

    def status(self, client, app_id=None, status="Rejected"):
        return client.post(f"/employer/application/{app_id or self.app_id}/status", data={"status": status})

    def current_status(self):
        return next(a for a in app_module.RUNTIME_APPLICATIONS if a["id"] == self.app_id)["status"]

    def test_owner_employer_can_update_status(self):
        self.assertEqual(self.status(self.apex(), status="Under Review").status_code, 302)
        self.assertEqual(self.current_status(), "Under Review")

    def test_other_employer_cannot_update_status(self):
        self.assertEqual(self.status(self.other_employer).status_code, 403)
        self.assertEqual(self.current_status(), "Submitted")
        self.assertEqual(next(a for a in read_applications_file() if a["id"] == self.app_id)["status"], "Submitted")

    def test_employer_with_same_company_name_cannot_take_over(self):
        impostor = self.signup("impostor@example.com", role="employer", company="Apex Cloud Systems", name="Impostor")
        self.assertEqual(self.status(impostor).status_code, 403)
        self.assertEqual(impostor.post(f"/jobs/{self.job['id']}/delete").status_code, 403)
        self.assertEqual(impostor.get(f"/jobs/{self.job['id']}/edit").status_code, 403)
        body = impostor.get("/dashboard/employer").get_data(as_text=True)
        self.assertNotIn(self.seeker_email, body)
        self.assertNotIn(self.job["title"], body)
        self.assertIn(self.job, app_module.RUNTIME_JOBS)

    def test_seeker_anonymous_cannot_update_status(self):
        self.assertEqual(self.status(self.seeker_client).status_code, 403)
        anon = self.status(app_module.app.test_client())
        self.assertEqual(anon.status_code, 302)
        self.assertIn("/login", anon.headers["Location"])
        self.assertEqual(self.current_status(), "Submitted")

    def test_admin_can_update_status(self):
        self.assertEqual(self.status(self.admin(), status="Shortlisted").status_code, 302)
        self.assertEqual(self.current_status(), "Shortlisted")

    def test_unknown_application_404_and_invalid_status_ignored(self):
        employer = self.apex()
        self.assertEqual(self.status(employer, app_id="APP-NOPE").status_code, 404)
        self.assertEqual(self.status(employer, status="Hacked").status_code, 302)
        self.assertEqual(self.current_status(), "Submitted")

    def test_employer_dashboard_shows_only_own_applicants(self):
        owner_page = self.apex().get("/dashboard/employer").get_data(as_text=True)
        self.assertIn(self.seeker_email, owner_page)
        other_page = self.other_employer.get("/dashboard/employer").get_data(as_text=True)
        self.assertNotIn(self.seeker_email, other_page)
        self.assertNotIn("Apex Cloud Systems", other_page.split("Candidate Applications")[-1].split("Company Vacancies")[0])
        self.assertIn("Company Vacancies (0)", other_page)

    def test_employer_cannot_manage_other_companys_job(self):
        foreign = self.other_company_job()
        employer = self.apex()
        self.assertEqual(employer.post(f"/jobs/{foreign['id']}/delete").status_code, 403)
        self.assertEqual(employer.get(f"/jobs/{foreign['id']}/edit").status_code, 403)
        self.assertIn(foreign, app_module.RUNTIME_JOBS)

    def test_seeker_sees_only_own_applications_and_status(self):
        second, second_email = self.new_seeker("second")
        self.apply_as(second, second_email, self.job["id"], cover_note="second note")
        first_job_page = self.seeker_client.get(f"/jobs/{self.job['id']}").get_data(as_text=True)
        second_job_page = second.get(f"/jobs/{self.job['id']}").get_data(as_text=True)
        # each seeker sees only their own submitted application on the job page
        self.assertIn(self.seeker_email, first_job_page)
        self.assertNotIn(second_email, first_job_page)
        self.assertIn(second_email, second_job_page)
        self.assertNotIn(self.seeker_email, second_job_page)
        # and the dashboards list only their own applications
        self.assertEqual(len([a for a in app_module.RUNTIME_APPLICATIONS if a["applicant_email"] == second_email]), 1)
        self.assertIn(self.job["title"], self.seeker_client.get("/dashboard/seeker").get_data(as_text=True))
        # an unrelated seeker with no application sees none of them
        outsider, outsider_email = self.new_seeker("outsider")
        outsider_page = outsider.get("/dashboard/seeker").get_data(as_text=True)
        self.assertNotIn(self.seeker_email, outsider_page)
        self.assertNotIn(second_email, outsider_page)

    def test_non_seeker_roles_cannot_apply(self):
        before = len(app_module.RUNTIME_APPLICATIONS)
        self.apply(self.apex(), self.job["id"])
        self.apply(app_module.app.test_client(), self.job["id"])
        self.assertEqual(len(app_module.RUNTIME_APPLICATIONS), before)

    def test_role_dashboards_and_pages_are_protected(self):
        anon = app_module.app.test_client()
        for path in ["/dashboard", "/dashboard/seeker", "/dashboard/employer", "/dashboard/admin",
                     "/profile", "/employer/profile", "/jobs/create"]:
            self.assertEqual(anon.get(path).status_code, 302, path)
        seeker = self.seeker_client
        for path in ["/dashboard/employer", "/dashboard/admin", "/jobs/create", "/employer/profile"]:
            self.assertIn(seeker.get(path).status_code, (302, 403), path)
        employer = self.apex()
        for path in ["/dashboard/admin", "/dashboard/seeker"]:
            self.assertEqual(employer.get(path).status_code, 403, path)

    def test_legacy_jobs_linked_to_employer_and_persisted(self):
        on_disk = json.loads((env.DATA_DIR / "jobs.json").read_text(encoding="utf-8"))
        apex_jobs = [j for j in on_disk if j.get("company") == "Apex Cloud Systems"]
        self.assertTrue(apex_jobs)
        self.assertTrue(all(j.get("employer_id") == "USR-102" for j in apex_jobs))


# ==============================================================================
# Security and general regression
# ==============================================================================
class SecurityAndRegressionTests(PortalCase):
    def test_updated_frontend_assets_are_cache_versioned(self):
        response = app_module.app.test_client().get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"style.css?v=20261023", response.data)
        self.assertIn(b"main.js?v=20261023", response.data)

    def test_resume_analyzer_method_name_is_hidden_from_rendered_ui(self):
        client = app_module.app.test_client()
        public_pages = ("/", "/login", "/signup", "/jobs")
        for path in public_pages:
            response = client.get(path)
            self.assertEqual(response.status_code, 200, path)
            self.assertNotIn("TF-IDF", response.get_data(as_text=True), path)

        seeker = self.seeker()
        for path in ("/dashboard/seeker", "/resume-analyzer"):
            response = seeker.get(path)
            self.assertEqual(response.status_code, 200, path)
            self.assertNotIn("TF-IDF", response.get_data(as_text=True), path)

    def test_admin_login_requires_configured_password_not_legacy_hash(self):
        seeded_admin = next((u for u in app_module.RUNTIME_USERS if u.get("email") == "admin@smartjobportal.com"), None)
        self.assertIsNotNone(seeded_admin)
        self.assertEqual(seeded_admin["role"], "admin")
        client = app_module.app.test_client()
        login_page = client.get("/login").get_data(as_text=True)
        self.assertNotIn("ADMIN_PASSWORD", login_page)
        self.assertNotIn("admin123", login_page)
        for page in ("/login", "/signup"):
            rendered = client.get(page).get_data(as_text=True)
            self.assertNotIn("Test Accounts for Evaluator", rendered)
            self.assertNotIn("seeker123", rendered)
            self.assertNotIn("employer123", rendered)
        bad = client.post("/login", data={"email": "admin@smartjobportal.com", "password": "admin123"})
        self.assertIn(b"Invalid email or password", bad.data)
        good = client.post("/login", data={"email": "admin@smartjobportal.com", "password": os.environ["ADMIN_PASSWORD"]})
        self.assertEqual(good.status_code, 302)
        with client.session_transaction() as sess:
            self.assertEqual(sess["user_role"], "admin")

        # A legacy reserved-email row with a non-admin role must not block the
        # configured administrator credential or inherit its conflicting role.
        original_role = seeded_admin["role"]
        seeded_admin["role"] = "seeker"
        try:
            legacy_client = app_module.app.test_client()
            response = legacy_client.post("/login", data={
                "email": "admin@smartjobportal.com", "password": os.environ["ADMIN_PASSWORD"]
            })
            self.assertEqual(response.status_code, 302)
            with legacy_client.session_transaction() as sess:
                self.assertEqual(sess["user_role"], "admin")
        finally:
            seeded_admin["role"] = original_role

    def test_dotenv_is_loaded_from_app_directory_when_launched_elsewhere(self):
        with tempfile.TemporaryDirectory(prefix="sjp-dotenv-", dir=env.TEMP_ROOT) as tmp:
            project = Path(tmp) / "project"
            project.mkdir()
            shutil.copy2(PROJECT_DIR / "app.py", project / "app.py")
            shutil.copytree(PROJECT_DIR / "services", project / "services")
            shutil.copytree(PROJECT_DIR / "data", project / "data")
            secret_key = secrets.token_hex(32)
            admin_password = secrets.token_urlsafe(24)
            (project / ".env").write_text(
                f"SECRET_KEY={secret_key}\nADMIN_PASSWORD={admin_password}\n",
                encoding="utf-8",
            )
            child_env = dict(os.environ)
            for key in ("APP_ENV", "FLASK_ENV", "FLASK_DEBUG", "SECRET_KEY", "ADMIN_PASSWORD", "SMART_JOB_PORTAL_DATA_DIR"):
                child_env.pop(key, None)
            code = (
                "import sys; sys.path.insert(0, " + repr(str(project)) + "); import app; "
                "print(app.app.config['SECRET_KEY']); print(app.app.config['ADMIN_PASSWORD'])"
            )
            proc = subprocess.run([sys.executable, "-c", code], cwd=tmp, env=child_env,
                                  capture_output=True, text=True, timeout=120)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn(secret_key, proc.stdout)
            self.assertIn(admin_password, proc.stdout)

    def test_resume_score_label_and_value_are_constrained_and_centered(self):
        template = (PROJECT_DIR / "templates" / "resume_analyzer.html").read_text(encoding="utf-8")
        css = (PROJECT_DIR / "static" / "css" / "style.css").read_text(encoding="utf-8")
        self.assertIn('class="score-label">SKILL MATCH</span>', template)
        self.assertRegex(css, r"\.score-num\s*\{[^}]*max-width:\s*100%;[^}]*text-align:\s*center;")
        self.assertRegex(css, r"\.score-label\s*\{[^}]*max-width:\s*100%;[^}]*text-align:\s*center;")

    def test_every_delete_form_has_a_confirmation_prompt(self):
        templates = ["dashboard_admin.html", "dashboard_employer.html", "job_detail.html"]
        forms = []
        for template_name in templates:
            source = (PROJECT_DIR / "templates" / template_name).read_text(encoding="utf-8")
            forms.extend(tag for tag in re.findall(r"<form[^>]*>", source) if "delete" in tag.lower())
        self.assertGreaterEqual(len(forms), 4)
        self.assertTrue(all("onsubmit=" in form and "confirm(" in form for form in forms))

    def test_signup_login_logout_and_password_rules(self):
        c = app_module.app.test_client()
        short = c.post("/signup", data={"name": "A", "email": "short@example.com", "role": "seeker", "password": "12345", "confirm_password": "12345"})
        self.assertIn(b"at least 6", short.data)
        mismatch = c.post("/signup", data={"name": "A", "email": "mm@example.com", "role": "seeker", "password": "password1", "confirm_password": "password2"})
        self.assertIn(b"do not match", mismatch.data)
        admin_attempt = self.signup("wannabe@example.com")
        c2 = app_module.app.test_client()
        c2.post("/signup", data={"name": "Eve", "email": "eve@example.com", "role": "admin", "password": "password123", "confirm_password": "password123"})
        self.assertEqual(next(u for u in app_module.RUNTIME_USERS if u["email"] == "eve@example.com")["role"], "seeker")
        client = self.seeker()
        self.assertEqual(client.get("/dashboard/seeker").status_code, 200)
        client.get("/logout")
        self.assertEqual(client.get("/dashboard/seeker").status_code, 302)
        self.assertIn(b"Invalid email or password",
                      app_module.app.test_client().post("/login", data={"email": "seeker@example.com", "password": "nope"}).data)

    def test_login_has_no_open_redirect(self):
        resp = app_module.app.test_client().post("/login?next=https://evil.example", data={"email": "seeker@example.com", "password": "seeker123"})
        self.assertNotIn("evil.example", resp.headers["Location"])

    def test_csrf_enforced_when_enabled(self):
        app_module.app.config["WTF_CSRF_ENABLED"] = True
        try:
            client = app_module.app.test_client()
            self.assertEqual(client.post("/login", data={"email": "seeker@example.com", "password": "seeker123"}).status_code, 400)
            page = client.get("/login").get_data(as_text=True)
            token = re.search(r'name="csrf_token" value="([^"]+)"', page).group(1)
            ok = client.post("/login", data={"email": "seeker@example.com", "password": "seeker123", "csrf_token": token})
            self.assertEqual(ok.status_code, 302)
            # API endpoint also needs the header token
            self.assertEqual(client.post("/api/toggle-save", json={"job_id": app_module.RUNTIME_JOBS[0]["id"]}).status_code, 400)
        finally:
            app_module.app.config["WTF_CSRF_ENABLED"] = False

    def test_user_input_is_escaped(self):
        client = self.apex()
        client.post("/jobs/create", data=self.job_form(title="<script>alert(1)</script>", description="<img src=x onerror=alert(2)>"))
        job = self.find_job("<script>alert(1)</script>")
        self.assertIsNotNone(job)
        listing = app_module.app.test_client().get("/jobs?q=script").get_data(as_text=True)
        detail = app_module.app.test_client().get(f"/jobs/{job['id']}").get_data(as_text=True)
        for page in (listing, detail):
            self.assertNotIn("<script>alert(1)</script>", page)
            self.assertNotIn("<img src=x onerror=alert(2)>", page)

    def test_job_crud_search_and_filters(self):
        client = self.apex()
        self.assertEqual(client.post("/jobs/create", data=self.job_form(title="Zebra Searchable Role", location="Chennai, India")).status_code, 302)
        job = self.find_job("Zebra Searchable Role")
        public = app_module.app.test_client()
        self.assertIn(b"Zebra Searchable Role", public.get("/jobs?q=zebra").data)
        self.assertIn(b"Zebra Searchable Role", public.get("/jobs?location=chennai").data)
        self.assertNotIn(b"Zebra Searchable Role", public.get("/jobs?q=zebra&employment_type=Part-time").data)
        self.assertEqual(client.post(f"/jobs/{job['id']}/edit", data=self.job_form(title="Zebra Renamed")).status_code, 302)
        self.assertEqual(job["title"], "Zebra Renamed")
        self.assertEqual(client.post(f"/jobs/{job['id']}/delete").status_code, 302)
        self.assertIsNone(self.find_job("Zebra Renamed"))
        self.assertEqual(public.get(f"/jobs/{job['id']}").status_code, 404)
        for query in ["?work_arrangement=Remote", "?employment_type=Full-time", "?experience_level=Entry-level", "?sort=salary_high", "?min_salary=500000"]:
            self.assertEqual(public.get("/jobs" + query).status_code, 200, query)

    def test_invalid_ids_return_404(self):
        employer, admin, seeker = self.apex(), self.admin(), self.seeker()
        self.assertEqual(app_module.app.test_client().get("/jobs/NOPE").status_code, 404)
        self.assertEqual(employer.get("/jobs/NOPE/edit").status_code, 404)
        self.assertEqual(employer.post("/jobs/NOPE/delete").status_code, 404)
        self.assertEqual(seeker.post("/jobs/NOPE/apply", data={"applicant_name": "A", "applicant_email": "a@b.co"}).status_code, 404)
        self.assertEqual(admin.post("/admin/job/NOPE/toggle-visibility").status_code, 404)
        self.assertEqual(admin.post("/admin/user/NOPE/delete").status_code, 404)

    def test_admin_moderation_and_user_deletion_rules(self):
        admin = self.admin()
        job = app_module.RUNTIME_JOBS[4]
        admin.post(f"/admin/job/{job['id']}/toggle-visibility")
        self.assertTrue(job["is_hidden"])
        admin.post(f"/admin/job/{job['id']}/toggle-visibility")
        self.assertFalse(job["is_hidden"])
        admin.post("/admin/user/USR-103/delete")
        self.assertTrue(any(u["id"] == "USR-103" for u in app_module.RUNTIME_USERS))
        self.assertEqual(self.seeker().post("/admin/user/USR-102/delete").status_code, 403)
        self.assertEqual(self.apex().post(f"/admin/job/{job['id']}/toggle-visibility").status_code, 403)
        self.assertEqual(admin.get("/dashboard/admin").status_code, 200)

    def test_ai_feature_pages_still_work(self):
        client = self.seeker()
        for path in ["/recommendations", "/recommendations?skills=Python,SQL", "/career-roadmap",
                     "/career-roadmap?role=python_developer", "/interview-prep", "/interview-prep?category=technical",
                     "/resume-analyzer", "/dashboard/seeker", "/profile"]:
            self.assertEqual(client.get(path).status_code, 200, path)

    def test_profile_update_persists_to_users_file(self):
        client = self.seeker()
        client.post("/profile", data={"name": "Alex Seeker", "headline": "Dev", "phone": "1", "bio": "b",
                                      "portfolio_url": "https://example.com", "skills": "Python, React"})
        saved = json.loads((env.DATA_DIR / "users.json").read_text(encoding="utf-8"))
        self.assertIn("React", next(u for u in saved if u["email"] == "seeker@example.com")["skills"])


class ZZDataIsolationTest(unittest.TestCase):
    """Runs last (alphabetical class order is not guaranteed, so it checks state at call time)."""

    def test_real_project_data_was_never_touched(self):
        self.assertTrue(env.real_data_unchanged(), "tests modified the real data/ directory")
        self.assertNotEqual(env.DATA_DIR.resolve(), env.REAL_DATA_DIR.resolve())
        self.assertFalse((env.REAL_DATA_DIR / "applications.json").exists())
        self.assertEqual(list(env.REAL_DATA_DIR.glob("*.corrupt-*")), [])


if __name__ == "__main__":
    unittest.main()
