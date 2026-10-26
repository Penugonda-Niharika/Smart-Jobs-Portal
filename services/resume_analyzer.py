"""
Resume analyzer and job matcher service.
Extracts text from PDF/TXT resumes, computes TF-IDF cosine similarity against
job descriptions, identifies keyword skill gaps, and suggests improvements.
"""

import re
import io
from typing import Tuple, List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pypdf

# Common technical and professional skill taxonomy for keyword matching
SKILLS_TAXONOMY = [
    # Languages
    "python", "java", "javascript", "typescript", "c++", "c#", "go", "ruby", "php", "swift", "dart", "sql", "bash",
    # Frameworks & Libraries
    "flask", "django", "fastapi", "react", "next.js", "angular", "vue", "express", "node.js", "spring boot", "pandas",
    "numpy", "scikit-learn", "pytorch", "tensorflow", "keras", "selenium", "playwright", "cypress", "tailwind",
    # Cloud & DevOps
    "docker", "kubernetes", "aws", "azure", "gcp", "terraform", "ansible", "ci/cd", "git", "linux", "prometheus", "grafana",
    # Data & Analytics
    "machine learning", "deep learning", "nlp", "computer vision", "llms", "data analysis", "data visualization",
    "power bi", "tableau", "excel", "etl", "data warehousing", "postgresql", "mysql", "mongodb", "redis",
    # Business & Design
    "agile", "scrum", "jira", "figma", "ui/ux", "product management", "seo", "google analytics", "copywriting",
    "content strategy", "technical writing", "customer service", "event management", "accounting", "financial analysis"
]

MAX_FILE_SIZE_BYTES = 4 * 1024 * 1024  # 4 MB
ALLOWED_EXTENSIONS = {"pdf", "txt"}


def is_allowed_file(filename: str) -> bool:
    """Check if uploaded file has an allowed extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_text_from_file(file_storage) -> Tuple[str, str | None]:
    """
    Extracts text from an uploaded Werkzeug FileStorage object.
    Returns (extracted_text, error_message).
    Handles empty files, malformed PDFs, and scanned PDFs gracefully.
    """
    filename = file_storage.filename or "unknown"
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if ext not in ALLOWED_EXTENSIONS:
        return "", f"Unsupported file extension '.{ext}'. Please upload a .pdf or .txt resume."

    # Read bytes into memory to check size
    content_bytes = file_storage.read()
    if len(content_bytes) > MAX_FILE_SIZE_BYTES:
        return "", f"File size exceeds the 4 MB limit ({len(content_bytes) / (1024*1024):.1f} MB)."
    if len(content_bytes) == 0:
        return "", "The uploaded file is empty (0 bytes)."

    if ext == "txt":
        try:
            text = content_bytes.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text = content_bytes.decode("latin-1")
            except Exception as e:
                return "", f"Unable to read TXT file encoding: {str(e)}"
        
        cleaned = text.strip()
        if not cleaned:
            return "", "The uploaded text file does not contain readable characters."
        return cleaned, None

    elif ext == "pdf":
        try:
            reader = pypdf.PdfReader(io.BytesIO(content_bytes))
            if reader.is_encrypted:
                try:
                    # Attempt empty password decryption
                    reader.decrypt("")
                except Exception:
                    return "", "The PDF is password protected. Please upload an unlocked PDF."

            pages_text = []
            for i, page in enumerate(reader.pages):
                page_content = page.extract_text() or ""
                pages_text.append(page_content)

            full_text = "\n".join(pages_text).strip()

            if not full_text:
                return "", (
                    "Could not extract readable text from this PDF. "
                    "If this is an image-only or scanned PDF, please upload a text-selectable PDF or a .txt resume "
                    "(optical character recognition / OCR is not supported)."
                )

            return full_text, None

        except Exception as e:
            return "", f"Malformed or unreadable PDF file: {str(e)}"

    return "", "Unknown file processing error."


def extract_skills_from_text(text: str) -> List[str]:
    """
    Extracts recognized technical and professional skills from raw text using regex word boundaries.
    """
    lower_text = text.lower()
    found_skills = []
    
    for skill in SKILLS_TAXONOMY:
        # Match as discrete word or phrase
        pattern = r"(?<!\w)" + re.escape(skill) + r"(?!\w)"
        if re.search(pattern, lower_text):
            found_skills.append(skill.title() if len(skill) > 3 else skill.upper())
            
    return sorted(list(set(found_skills)))


def _skill_in_text(skill: str, text_lower: str) -> bool:
    """Whole-word/phrase check so job skills outside the built-in taxonomy (e.g. 'Kafka') can still match."""
    pattern = r"(?<![\w+#.])" + re.escape(skill.strip().lower()) + r"(?![\w+#])"
    return re.search(pattern, text_lower) is not None


def analyze_resume_against_job(resume_text: str, job_description: str, job_skills: List[str] = None) -> Dict[str, Any]:
    """
    Compares a resume with a target job.

    Headline score (``similarity_score``) = skill-match percentage:
        matched required skills / total required skills * 100.
    Required skills come from ``job_skills`` or, if none are given, are extracted from the job description.
    If no required skills can be identified at all, the headline falls back to TF-IDF text similarity
    (``score_basis`` tells which one is used). TF-IDF similarity is always reported as ``tfidf_similarity``.
    """
    job_skills = [s for s in (job_skills or []) if s and s.strip()]

    # Clean and check inputs
    resume_clean = (resume_text or "").strip()
    job_clean = (job_description or "").strip()

    if not resume_clean or not job_clean:
        return {
            "similarity_score": 0,
            "skill_match_percentage": 0,
            "tfidf_similarity": 0,
            "score_basis": "none",
            "matched_skills": [],
            "missing_skills": job_skills,
            "extracted_resume_skills": [],
            "suggestions": ["Please ensure both your resume and the target job description contain readable text."],
            "disclaimer": "This score is a heuristic estimate and not a guarantee of hiring."
        }

    # 1. TF-IDF & Cosine Similarity (secondary, text-level signal)
    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=5000
        )
        tfidf_matrix = vectorizer.fit_transform([resume_clean, job_clean])
        cos_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        tfidf_similarity = round(float(cos_sim) * 100, 1)
    except Exception:
        tfidf_similarity = 0.0

    # 2. Skill Extraction & matching
    extracted_resume_skills = extract_skills_from_text(resume_clean)
    resume_skills_lower = {s.lower() for s in extracted_resume_skills}
    resume_lower = resume_clean.lower()

    # Required skills: explicit job skills, otherwise those found in the job description
    required_skills = job_skills if job_skills else extract_skills_from_text(job_clean)

    matched_skills = []
    missing_skills = []
    for skill in required_skills:
        if skill.lower() in resume_skills_lower or _skill_in_text(skill, resume_lower):
            matched_skills.append(skill)
        else:
            missing_skills.append(skill)

    # 3. Headline score
    if required_skills:
        skill_match_percentage = round(len(matched_skills) / len(required_skills) * 100, 1)
        similarity_score = skill_match_percentage
        score_basis = "skill_match"
    else:
        skill_match_percentage = 0
        similarity_score = tfidf_similarity
        score_basis = "text_similarity"

    # 4. Actionable Suggestions Generation
    suggestions = []

    if missing_skills:
        missing_sample = ", ".join(missing_skills[:4])
        suggestions.append(f"Consider integrating experience or projects involving missing target skills: {missing_sample}.")

    if len(resume_clean.split()) < 150:
        suggestions.append("Your resume content seems brief (< 150 words). Adding specific project details, tools, and quantified outcomes will improve matching.")

    if similarity_score < 30:
        suggestions.append("Few of the target requirements appear in your resume. Tailor your summary and bullet points to mirror the key skills and keywords used in the job description.")
    elif similarity_score < 60:
        suggestions.append("Good baseline alignment. To boost your match, emphasize your hands-on accomplishments with the target tools and frameworks.")
    else:
        suggestions.append("Strong alignment! Make sure your achievements feature concrete metrics (e.g., 'improved query performance by 35%').")

    suggestions.append("Ensure your contact information, GitHub/LinkedIn links, and education credentials are easily readable at the top.")

    if score_basis == "skill_match":
        disclaimer = ("Notice: The headline score is the share of the job's required skills found in your resume "
                      f"({len(matched_skills)} of {len(required_skills)}). TF-IDF text similarity is reported separately. "
                      "It does not evaluate candidates' human potential, soft skills, or actual hiring probability.")
    else:
        disclaimer = ("Notice: No specific required skills could be identified, so this score measures lexical similarity "
                      "via TF-IDF cosine analysis. It does not evaluate candidates' human potential, soft skills, or actual hiring probability.")

    return {
        "similarity_score": similarity_score,
        "skill_match_percentage": skill_match_percentage,
        "tfidf_similarity": tfidf_similarity,
        "score_basis": score_basis,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "extracted_resume_skills": extracted_resume_skills,
        "suggestions": suggestions,
        "word_count": len(resume_clean.split()),
        "disclaimer": disclaimer
    }
