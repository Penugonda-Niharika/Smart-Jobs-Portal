"""
Automated Comprehensive Test Suite for Smart Job Portal.
Executes real input/output verification across authentication, role permissions,
job CRUD, filtering, AI/ML features, and responsive data integrity.
"""

import portal_test_env  # noqa: F401  (must come first: points the app at a temporary copy of data/)
import unittest
import json
import io
import app as app_module
from app import app
from services.recommendations import get_recommendations, compute_skill_match
from services.resume_analyzer import (
    extract_text_from_file, analyze_resume_against_job, extract_skills_from_text
)
from services.career_guidance import analyze_career_gap
from services.job_risk_checker import evaluate_job_risk
from werkzeug.datastructures import FileStorage
from werkzeug.security import check_password_hash


class SmartJobPortalComprehensiveTests(unittest.TestCase):
    def setUp(self):
        app.config["TESTING"] = True
        app.config["WTF_CSRF_ENABLED"] = False  # Disable CSRF during test calls
        self.client = app.test_client()

    # --------------------------------------------------------------------------
    # 1. Authentication & Security
    # --------------------------------------------------------------------------
    def test_01_user_signup_seeker(self):
        """Test candidate user registration with password hashing."""
        response = self.client.post("/signup", data={
            "name": "Jane Candidate",
            "email": "jane.candidate@test.com",
            "role": "seeker",
            "password": "securepassword123",
            "confirm_password": "securepassword123"
        }, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        user = next((u for u in app_module.RUNTIME_USERS if u["email"] == "jane.candidate@test.com"), None)
        self.assertIsNotNone(user)
        self.assertTrue(check_password_hash(user["password_hash"], "securepassword123"))
        self.assertEqual(user["role"], "seeker")

    def test_02_user_signup_employer(self):
        """Test employer registration with company details."""
        response = self.client.post("/signup", data={
            "name": "Robert Recruiter",
            "email": "robert@innovatetech.test",
            "role": "employer",
            "company_name": "InnovateTech Corp",
            "password": "employerpass123",
            "confirm_password": "employerpass123"
        }, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        user = next((u for u in app_module.RUNTIME_USERS if u["email"] == "robert@innovatetech.test"), None)
        self.assertIsNotNone(user)
        self.assertEqual(user["company_name"], "InnovateTech Corp")
        self.assertEqual(user["role"], "employer")

    def test_03_login_valid_and_invalid(self):
        """Test authentication for valid and invalid credentials."""
        # Invalid password
        bad_resp = self.client.post("/login", data={
            "email": "seeker@example.com",
            "password": "wrongpassword"
        }, follow_redirects=True)
        self.assertIn(b"Invalid email or password", bad_resp.data)

        # Valid password
        good_resp = self.client.post("/login", data={
            "email": "seeker@example.com",
            "password": "seeker123"
        }, follow_redirects=True)
        self.assertEqual(good_resp.status_code, 200)
        self.assertIn(b"Candidate Dashboard", good_resp.data)

    def test_04_role_access_control(self):
        """Verify that job seekers cannot access employer or admin dashboards."""
        with self.client.session_transaction() as sess:
            sess["user_id"] = "USR-101"
            sess["user_role"] = "seeker"
            sess["user_email"] = "seeker@example.com"

        # Seeker accessing employer dashboard should be forbidden (403)
        emp_resp = self.client.get("/dashboard/employer")
        self.assertEqual(emp_resp.status_code, 403)

        # Seeker accessing admin dashboard should be forbidden (403)
        admin_resp = self.client.get("/dashboard/admin")
        self.assertEqual(admin_resp.status_code, 403)

    # --------------------------------------------------------------------------
    # 2. Strict Employment Types & Job Exploration
    # --------------------------------------------------------------------------
    def test_05_strict_employment_types_data(self):
        """Verify that every single job in the catalog is strictly Full-time or Part-time."""
        allowed = {"Full-time", "Part-time"}
        for j in app_module.RUNTIME_JOBS:
            self.assertIn(
                j.get("employment_type"), allowed,
                f"Job {j.get('id')} has invalid type: {j.get('employment_type')}"
            )

    def test_06_jobs_filtering(self):
        """Test search and independent filtering by type and arrangement."""
        # Filter full-time
        resp_ft = self.client.get("/jobs?employment_type=Full-time")
        self.assertEqual(resp_ft.status_code, 200)
        self.assertIn(b"Full-time", resp_ft.data)

        # Filter part-time
        resp_pt = self.client.get("/jobs?employment_type=Part-time")
        self.assertEqual(resp_pt.status_code, 200)
        self.assertIn(b"Part-time", resp_pt.data)

        # Filter Remote
        resp_remote = self.client.get("/jobs?work_arrangement=Remote")
        self.assertEqual(resp_remote.status_code, 200)

    def test_07_job_detail_and_links(self):
        """Test that job details page renders full attributes and external link if present."""
        first_job = app_module.RUNTIME_JOBS[0]
        resp = self.client.get(f"/jobs/{first_job['id']}")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(first_job["title"].encode("utf-8"), resp.data)
        self.assertIn(b"Key Responsibilities", resp.data)
        self.assertIn(b"Required Skills", resp.data)

        if first_job.get("application_url"):
            self.assertIn(b"Apply on Company / LinkedIn Site", resp.data)

    # --------------------------------------------------------------------------
    # 3. Employer CRUD Operations
    # --------------------------------------------------------------------------
    def test_08_employer_job_crud(self):
        """Test job creation, editing, and deletion by an employer."""
        with self.client.session_transaction() as sess:
            sess["user_id"] = "USR-102"
            sess["user_role"] = "employer"
            sess["company_name"] = "Apex Cloud Systems"

        # 1. Create Job
        create_resp = self.client.post("/jobs/create", data={
            "title": "Lead Python Distributed Architect",
            "company": "Apex Cloud Systems",
            "location": "Bengaluru, India",
            "employment_type": "Full-time",
            "work_arrangement": "Hybrid",
            "experience_level": "Senior-level",
            "salary_min": 2500000,
            "salary_max": 3500000,
            "salary_display": "₹25L - ₹35L / year",
            "description": "Architect high performance microservices.",
            "required_skills": "Python, Redis, Kafka, Docker",
            "responsibilities": "Design microservices\nLead engineering",
            "application_url": "https://careers.apexcloudsystems.example.com/lead-architect",
            "contact_email": "careers@apexcloudsystems.example.com"
        }, follow_redirects=True)
        self.assertEqual(create_resp.status_code, 200)

        created_job = next((j for j in app_module.RUNTIME_JOBS if j["title"] == "Lead Python Distributed Architect"), None)
        self.assertIsNotNone(created_job)
        self.assertEqual(created_job["employment_type"], "Full-time")

        # 2. Edit Job
        edit_resp = self.client.post(f"/jobs/{created_job['id']}/edit", data={
            "title": "Principal Python Systems Architect",
            "company": "Apex Cloud Systems",
            "location": "Bengaluru, India",
            "employment_type": "Full-time",
            "work_arrangement": "Remote",
            "experience_level": "Senior-level",
            "salary_min": 3000000,
            "salary_max": 4000000,
            "salary_display": "₹30L - ₹40L / year",
            "description": "Updated architecture description.",
            "required_skills": "Python, Redis, Kafka, Kubernetes",
            "responsibilities": "Lead systems design",
            "application_url": "https://careers.apexcloudsystems.example.com/lead-architect",
            "contact_email": "careers@apexcloudsystems.example.com"
        }, follow_redirects=True)
        self.assertEqual(edit_resp.status_code, 200)
        self.assertEqual(created_job["title"], "Principal Python Systems Architect")

        # 3. Delete Job
        delete_resp = self.client.post(f"/jobs/{created_job['id']}/delete", follow_redirects=True)
        self.assertEqual(delete_resp.status_code, 200)
        # Read the live list: the delete route rebinds app.RUNTIME_JOBS, so an earlier reference would be stale.
        self.assertIsNone(next((j for j in app_module.RUNTIME_JOBS if j["id"] == created_job["id"]), None))
        on_disk = json.loads((portal_test_env.DATA_DIR / "jobs.json").read_text(encoding="utf-8"))
        self.assertFalse(any(j["id"] == created_job["id"] for j in on_disk))

    # --------------------------------------------------------------------------
    # 4. Job Applications & Status Updates
    # --------------------------------------------------------------------------
    def test_09_applications_flow(self):
        """Test candidate application submission, duplicate blocking, and status updates."""
        # The status-updating employer (USR-102) may only manage applications on jobs they own.
        target_job = next(j for j in app_module.RUNTIME_JOBS if j.get("employer_id") == "USR-102")

        with self.client.session_transaction() as sess:
            sess["user_id"] = "USR-TEST-APPLICANT"
            sess["user_role"] = "seeker"
            sess["user_name"] = "Test Candidate"
            sess["user_email"] = "candidate.applicant@test.com"

        # Apply
        app_resp = self.client.post(f"/jobs/{target_job['id']}/apply", data={
            "applicant_name": "Test Candidate",
            "applicant_email": "candidate.applicant@test.com",
            "cover_note": "I have solid Python and Machine Learning experience.",
            "portfolio_url": "https://github.com/testcandidate"
        }, follow_redirects=True)
        self.assertEqual(app_resp.status_code, 200)

        app_record = next((a for a in app_module.RUNTIME_APPLICATIONS if a["applicant_email"] == "candidate.applicant@test.com"), None)
        self.assertIsNotNone(app_record)
        self.assertEqual(app_record["status"], "Submitted")

        # Duplicate submission check
        dup_resp = self.client.post(f"/jobs/{target_job['id']}/apply", data={
            "applicant_name": "Test Candidate",
            "applicant_email": "candidate.applicant@test.com"
        }, follow_redirects=True)
        self.assertIn(b"already submitted an application", dup_resp.data)

        # Employer updates status to Shortlisted
        with self.client.session_transaction() as sess:
            sess["user_id"] = "USR-102"
            sess["user_role"] = "employer"
            sess["company_name"] = target_job["company"]

        status_resp = self.client.post(f"/employer/application/{app_record['id']}/status", data={
            "status": "Shortlisted"
        }, follow_redirects=True)
        self.assertEqual(status_resp.status_code, 200)
        self.assertEqual(app_record["status"], "Shortlisted")

    # --------------------------------------------------------------------------
    # 5. AI / ML Features Verification
    # --------------------------------------------------------------------------
    def test_10_resume_text_extraction(self):
        """Verify text extraction on valid and invalid files."""
        # Valid TXT resume
        txt_bytes = b"Jane Doe - Software Engineer with Python, Flask, SQL, Git, and Docker experience."
        storage_txt = FileStorage(stream=io.BytesIO(txt_bytes), filename="resume.txt")
        text, err = extract_text_from_file(storage_txt)
        self.assertIsNone(err)
        self.assertIn("Python", text)

        # Unsupported file extension
        unsupported = FileStorage(stream=io.BytesIO(b"binary"), filename="resume.docx")
        text_bad, err_bad = extract_text_from_file(unsupported)
        self.assertIn("Unsupported file extension", err_bad)

        # Empty file
        empty = FileStorage(stream=io.BytesIO(b""), filename="empty.txt")
        text_empty, err_empty = extract_text_from_file(empty)
        self.assertIn("empty", err_empty)

    def test_11_tfidf_cosine_similarity(self):
        """Test scikit-learn TF-IDF cosine similarity between resume and job description."""
        resume = "Senior Python engineer building Flask web applications, Docker containers, and SQL queries."
        job_desc = "Seeking Python Developer with Flask, REST APIs, SQL, and Docker containerization experience."
        skills = ["Python", "Flask", "SQL", "Docker", "Kubernetes"]

        result = analyze_resume_against_job(resume, job_desc, skills)
        # Headline score is the skill-match percentage: 4 of the 5 required skills are present.
        self.assertEqual(result["similarity_score"], 80.0)
        self.assertEqual(result["skill_match_percentage"], 80.0)
        self.assertEqual(result["score_basis"], "skill_match")
        self.assertGreater(result["tfidf_similarity"], 0)  # text similarity still computed separately
        self.assertIn("Python", result["matched_skills"])
        self.assertIn("Kubernetes", result["missing_skills"])
        self.assertTrue(len(result["suggestions"]) > 0)

    def test_12_skill_recommendations(self):
        """Test recommendation engine ranking logic."""
        user_skills = ["Python", "Machine Learning", "scikit-learn"]
        recs = get_recommendations(user_skills, app_module.RUNTIME_JOBS, min_score=10, limit=5)
        self.assertTrue(len(recs) > 0)
        # Highest match should have high score
        self.assertGreaterEqual(recs[0]["score"], recs[-1]["score"])

    def test_13_career_roadmap(self):
        """Test career gap analysis and structured phased learning roadmaps."""
        gap = analyze_career_gap(["Python"], "python_developer", app_module.RUNTIME_JOBS)
        self.assertEqual(gap["role_title"], "Python Developer")
        self.assertIn("Python", gap["possessed_skills"])
        self.assertTrue(len(gap["missing_skills"]) > 0)
        self.assertEqual(len(gap["roadmap"]), 4)

    def test_14_suspicious_job_checker(self):
        """Test heuristic scam detection engine on normal vs suspicious jobs."""
        normal_job = app_module.RUNTIME_JOBS[0]
        normal_eval = evaluate_job_risk(normal_job)
        self.assertEqual(normal_eval["risk_level"], "Low")

        # Test job with upfront payment and personal email
        suspicious_job = next((j for j in app_module.RUNTIME_JOBS if "Deposit Required" in j["title"]), None)
        self.assertIsNotNone(suspicious_job)
        scam_eval = evaluate_job_risk(suspicious_job)
        self.assertEqual(scam_eval["risk_level"], "High")
        self.assertTrue(len(scam_eval["warnings"]) >= 2)

    def test_15_profile_persistence_after_logout_login(self):
        """Verify profile updates save properly and persist after logout/login for seekers and employers."""
        # 1. Job Seeker Profile Update & Persistence
        self.client.post("/signup", data={
            "name": "Alex Original",
            "email": "alex.persistent@test.com",
            "role": "seeker",
            "password": "seekerpass123",
            "confirm_password": "seekerpass123"
        }, follow_redirects=True)

        update_resp = self.client.post("/profile", data={
            "name": "Alex Updated",
            "headline": "Lead Full-Stack Python Architect",
            "phone": "+91 99999 11111",
            "portfolio_url": "https://github.com/alex-updated",
            "bio": "Experienced architect specializing in microservices and distributed storage.",
            "skills": "Python, Flask, Docker, Kubernetes, PostgreSQL"
        }, follow_redirects=True)
        self.assertEqual(update_resp.status_code, 200)

        # Logout seeker
        self.client.get("/logout", follow_redirects=True)

        # Login seeker again
        login_resp = self.client.post("/login", data={
            "email": "alex.persistent@test.com",
            "password": "seekerpass123"
        }, follow_redirects=True)
        self.assertEqual(login_resp.status_code, 200)

        # Verify profile has preserved all changes
        prof_page = self.client.get("/profile").get_data(as_text=True)
        self.assertIn("Alex Updated", prof_page)
        self.assertIn("Lead Full-Stack Python Architect", prof_page)
        self.assertIn("+91 99999 11111", prof_page)
        self.assertIn("https://github.com/alex-updated", prof_page)
        self.assertIn("PostgreSQL", prof_page)

        # 2. Employer Profile Update & Persistence
        self.client.get("/logout", follow_redirects=True)
        self.client.post("/signup", data={
            "name": "Elena Recruiter",
            "email": "elena@cloudscale.test",
            "role": "employer",
            "company_name": "CloudScale Innovations",
            "password": "employerpass123",
            "confirm_password": "employerpass123"
        }, follow_redirects=True)

        emp_update = self.client.post("/employer/profile", data={
            "name": "Elena Chief Talent Officer",
            "company_name": "CloudScale Global Systems",
            "company_industry": "Enterprise Cloud and AI Infra",
            "company_website": "https://cloudscaleglobal.example.com",
            "company_overview": "Building next-generation distributed inference nodes.",
            "phone": "+91 80 8888 2222"
        }, follow_redirects=True)
        self.assertEqual(emp_update.status_code, 200)

        # Logout employer
        self.client.get("/logout", follow_redirects=True)

        # Login employer again
        emp_login = self.client.post("/login", data={
            "email": "elena@cloudscale.test",
            "password": "employerpass123"
        }, follow_redirects=True)
        self.assertEqual(emp_login.status_code, 200)

        # Verify employer profile has preserved all changes
        emp_prof_page = self.client.get("/employer/profile").get_data(as_text=True)
        self.assertIn("Elena Chief Talent Officer", emp_prof_page)
        self.assertIn("CloudScale Global Systems", emp_prof_page)
        self.assertIn("Enterprise Cloud and AI Infra", emp_prof_page)
        self.assertIn("https://cloudscaleglobal.example.com", emp_prof_page)

    def test_16_resume_analyzer_recalculates_on_changing_job_role(self):
        """Verify Resume Analyzer recalculates score and skills when changing selected role with uploaded resume."""
        job_ml = next(j for j in app_module.RUNTIME_JOBS if "Machine Learning" in j["title"])
        job_fe = next(j for j in app_module.RUNTIME_JOBS if "Frontend" in j["title"])

        resume_payload = b"Experienced engineer skilled in Python, Scikit-learn, PyTorch, Docker, NLP, and Machine Learning."

        # 1. First calculation against Machine Learning Engineer
        resp1 = self.client.post("/resume-analyzer", data={
            "resume_file": (io.BytesIO(resume_payload), "engineer_resume.txt"),
            "job_id": job_ml["id"]
        }, follow_redirects=True)
        self.assertEqual(resp1.status_code, 200)
        body1 = resp1.get_data(as_text=True)
        self.assertIn("Resume Match Results", body1)
        self.assertIn("engineer_resume.txt", body1)
        # ML job requires Python, Scikit-learn, PyTorch, Docker, etc., so skill match is high
        self.assertIn("Machine Learning Engineer", body1)

        # 2. Recalculate against Frontend position using the cached uploaded resume
        resp2 = self.client.post("/resume-analyzer", data={
            "job_id": job_fe["id"],
            "cached_resume_text": "Experienced engineer skilled in Python, Scikit-learn, PyTorch, Docker, NLP, and Machine Learning.",
            "cached_resume_filename": "engineer_resume.txt"
        }, follow_redirects=True)
        self.assertEqual(resp2.status_code, 200)
        body2 = resp2.get_data(as_text=True)
        self.assertIn("Resume Match Results", body2)
        self.assertIn(job_fe["title"], body2)

    def test_17_employer_publishes_jobs_and_reviews_seeker_applications(self):
        """End-to-end test: employer posts multiple jobs, seeker applies, employer reviews applications."""
        # 1. Employer signs up
        emp_email = "lead.recruiter@hypergrowth.test"
        emp_pass = "HyperSecret2026!"
        self.client.post("/signup", data={
            "name": "Marcus Talent Lead",
            "email": emp_email,
            "role": "employer",
            "company_name": "HyperGrowth Cloud Corp",
            "password": emp_pass,
            "confirm_password": emp_pass
        }, follow_redirects=True)

        # 2. Employer publishes 2 jobs
        job1_resp = self.client.post("/jobs/create", data={
            "title": "Senior Cloud SRE Specialist",
            "company": "HyperGrowth Cloud Corp",
            "location": "Bengaluru, India",
            "employment_type": "Full-time",
            "work_arrangement": "Remote",
            "experience_level": "Senior-level",
            "salary_min": "1800000",
            "salary_max": "2800000",
            "description": "Lead multi-cloud reliability, zero-downtime deployments, and infrastructure automation.",
            "required_skills": "Python, Docker, Kubernetes, Terraform, Linux",
            "contact_email": "sre-hiring@hypergrowth.test"
        }, follow_redirects=True)
        self.assertEqual(job1_resp.status_code, 200)

        job2_resp = self.client.post("/jobs/create", data={
            "title": "Data Platform Automation Engineer",
            "company": "HyperGrowth Cloud Corp",
            "location": "Hyderabad, India",
            "employment_type": "Full-time",
            "work_arrangement": "Hybrid",
            "experience_level": "Mid-level",
            "salary_min": "1200000",
            "salary_max": "1900000",
            "description": "Architect high-throughput event ingestion pipelines and Kafka clusters.",
            "required_skills": "Python, Kafka, SQL, Redis, Docker",
            "contact_email": "data-jobs@hypergrowth.test"
        }, follow_redirects=True)
        self.assertEqual(job2_resp.status_code, 200)

        job3_resp = self.client.post("/jobs/create", data={
            "title": "Full-Stack AI Application Developer",
            "company": "HyperGrowth Cloud Corp",
            "location": "Bengaluru, India",
            "employment_type": "Full-time",
            "work_arrangement": "Remote",
            "experience_level": "Mid-level",
            "salary_min": "1400000",
            "salary_max": "2200000",
            "description": "Develop full-stack web applications integrating LLMs and real-time APIs.",
            "required_skills": "Python, JavaScript, React, FastAPI, Docker",
            "contact_email": "ai-careers@hypergrowth.test"
        }, follow_redirects=True)
        self.assertEqual(job3_resp.status_code, 200)

        emp_jobs = [j for j in app_module.RUNTIME_JOBS if j.get("company") == "HyperGrowth Cloud Corp"]
        self.assertEqual(len(emp_jobs), 3)

        # 3. Job Seeker signs up and applies to all 3 jobs
        self.client.get("/logout", follow_redirects=True)
        seeker_email = "priya.candidate@test.com"
        self.client.post("/signup", data={
            "name": "Priya Candidate",
            "email": seeker_email,
            "role": "seeker",
            "password": "candidatepass123",
            "confirm_password": "candidatepass123"
        }, follow_redirects=True)

        for job in emp_jobs:
            app_sub = self.client.post(f"/jobs/{job['id']}/apply", data={
                "applicant_name": "Priya Candidate",
                "applicant_email": seeker_email,
                "portfolio_url": "https://github.com/priya-candidate",
                "cover_note": "Experienced with automated infra, pipelines, and distributed platforms."
            }, follow_redirects=True)
            self.assertEqual(app_sub.status_code, 200)

        # Verify seeker sees all 3 applications
        seeker_dash = self.client.get("/dashboard/seeker").get_data(as_text=True)
        self.assertIn("Senior Cloud SRE Specialist", seeker_dash)
        self.assertIn("Data Platform Automation Engineer", seeker_dash)
        self.assertIn("Full-Stack AI Application Developer", seeker_dash)

        # 4. Employer logs in and views candidate applications
        self.client.get("/logout", follow_redirects=True)
        self.client.post("/login", data={"email": emp_email, "password": emp_pass}, follow_redirects=True)
        emp_dash = self.client.get("/dashboard/employer").get_data(as_text=True)
        self.assertIn("Priya Candidate", emp_dash)
        self.assertIn("Senior Cloud SRE Specialist", emp_dash)
        self.assertIn("Data Platform Automation Engineer", emp_dash)
        self.assertIn("Full-Stack AI Application Developer", emp_dash)

        # 5. Employer updates status on applications
        seeker_apps = [a for a in app_module.RUNTIME_APPLICATIONS if a["applicant_email"] == seeker_email]
        self.assertEqual(len(seeker_apps), 3)
        stat_update = self.client.post(f"/employer/application/{seeker_apps[0]['id']}/status", data={
            "status": "Shortlisted"
        }, follow_redirects=True)
        self.assertEqual(stat_update.status_code, 200)
        self.assertEqual(seeker_apps[0]["status"], "Shortlisted")


if __name__ == "__main__":
    unittest.main()

