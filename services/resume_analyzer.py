"""
Resume analyzer and job matcher service.
Extracts text from PDF/TXT resumes, computes TF-IDF cosine similarity against
job descriptions, identifies keyword skill gaps, and suggests improvements.
"""

import re
import io
import math
from collections import Counter
from typing import Tuple, List, Dict, Any
import pypdf

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

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


STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
    "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't",
    "down", "during", "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have",
    "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him",
    "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't",
    "it", "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor",
    "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out",
    "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some",
    "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there",
    "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through",
    "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've",
    "were", "weren't", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who",
    "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll",
    "you're", "you've", "your", "yours", "yourself", "yourselves"
}


def _tokenize_text(text: str) -> List[str]:
    """Tokenize text into lowercase unigrams and bigrams, removing punctuation and stop words."""
    words = re.findall(r"\b[a-zA-Z]{2,}\b", text.lower())
    unigrams = [w for w in words if w not in STOP_WORDS]
    bigrams = [
        f"{words[i]} {words[i+1]}"
        for i in range(len(words) - 1)
        if words[i] not in STOP_WORDS or words[i+1] not in STOP_WORDS
    ]
    return unigrams + bigrams


def compute_pure_python_tfidf_similarity(text1: str, text2: str) -> float:
    """
    Pure-Python TF-IDF and Cosine Similarity calculation.
    Enables serverless deployment on Vercel without requiring the heavy (>200MB)
    scikit-learn, scipy, and numpy packages.
    """
    tokens1 = _tokenize_text(text1)
    tokens2 = _tokenize_text(text2)
    if not tokens1 or not tokens2:
        return 0.0

    counts1 = Counter(tokens1)
    counts2 = Counter(tokens2)
    vocab = set(counts1.keys()) | set(counts2.keys())
    if not vocab:
        return 0.0

    N = 2
    tfidf1 = {}
    tfidf2 = {}
    for term in vocab:
        df = (1 if term in counts1 else 0) + (1 if term in counts2 else 0)
        idf = math.log((1 + N) / (1 + df)) + 1.0
        tfidf1[term] = (counts1.get(term, 0) / len(tokens1)) * idf
        tfidf2[term] = (counts2.get(term, 0) / len(tokens2)) * idf

    dot_product = sum(tfidf1[term] * tfidf2[term] for term in vocab)
    norm1 = math.sqrt(sum(val * val for val in tfidf1.values()))
    norm2 = math.sqrt(sum(val * val for val in tfidf2.values()))

    if norm1 == 0 or norm2 == 0:
        return 0.0

    sim = dot_product / (norm1 * norm2)
    return round(float(sim) * 100, 1)


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
    tfidf_similarity = 0.0
    if SKLEARN_AVAILABLE:
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
            tfidf_similarity = compute_pure_python_tfidf_similarity(resume_clean, job_clean)
    else:
        tfidf_similarity = compute_pure_python_tfidf_similarity(resume_clean, job_clean)

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
