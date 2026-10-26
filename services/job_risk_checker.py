"""
Suspicious job listing detection service.
Applies transparent heuristic rules to scan job descriptions, compensation,
and contact methods for common fraud or scam signals.
"""

import re
from typing import Dict, Any, List

UPFRONT_PAYMENT_PATTERNS = [
    r"upfront\s+fee", r"registration\s+fee", r"onboarding\s+fee",
    r"equipment\s+deposit", r"wire\s+transfer", r"send\s+money",
    r"buy\s+gift\s+card", r"western\s+union", r"crypto", r"bitcoin",
    r"cheque\s+deposit", r"deposit\s+it\s+in\s+your\s+personal\s+account"
]

SUSPICIOUS_CONTACT_PATTERNS = [
    r"telegram", r"whatsapp\s+only", r"@gmail\.com", r"@yahoo\.com",
    r"@hotmail\.com", r"@outlook\.com", r"@mail\.ru"
]

SENSITIVE_INFO_PATTERNS = [
    r"bank\s+account\s+login", r"bank\s+details", r"credit\s+card",
    r"routing\s+number", r"password", r"otp", r"id\s+card\s+copy"
]

URGENT_PRESSURE_PATTERNS = [
    r"immediate\s+hire", r"urgent\s+response\s+required", r"earn\s+\$\d{2,3}/hr",
    r"no\s+experience\s+required\s+at\s+all", r"free\s+cash"
]


def evaluate_job_risk(job: dict) -> Dict[str, Any]:
    """
    Evaluates a job listing for risk signals.
    Returns:
    - risk_level: 'Low', 'Medium', 'High'
    - risk_score: 0 to 100
    - warnings: list of specific identified reasons
    - disclaimer: explanation that this is a heuristic indicator
    """
    warnings: List[Dict[str, str]] = []
    score = 0

    title = job.get("title", "")
    description = job.get("description", "")
    responsibilities = " ".join(job.get("responsibilities", []))
    application_instructions = job.get("application_instructions", "")
    contact_email = job.get("contact_email", "")
    company = job.get("company", "")
    salary_max = job.get("salary_max", 0)
    exp_level = job.get("experience_level", "")

    full_text = f"{title} {description} {responsibilities} {application_instructions}".lower()

    # 1. Check for upfront payment / deposit requests (Critical signal)
    for pattern in UPFRONT_PAYMENT_PATTERNS:
        if re.search(pattern, full_text):
            score += 40
            warnings.append({
                "signal": "Upfront Payment or Financial Transaction Mentioned",
                "detail": f"Detected mention matching '{pattern}'. Legitimate employers never ask candidates to pay onboarding fees, buy software kits, or process wire transfers."
            })
            break

    # 2. Check for suspicious contact channels (free personal email or messaging apps)
    suspicious_contacts_found = []
    if contact_email:
        for pattern in SUSPICIOUS_CONTACT_PATTERNS:
            if re.search(pattern, contact_email.lower()):
                suspicious_contacts_found.append(contact_email)
                break

    for pattern in [r"telegram", r"whatsapp"]:
        if re.search(pattern, full_text):
            suspicious_contacts_found.append("Messaging App (Telegram/WhatsApp)")
            break

    if suspicious_contacts_found:
        score += 25
        warnings.append({
            "signal": "Non-Corporate Contact Channel",
            "detail": "Listing directs communication to personal email domains (e.g., Gmail/Yahoo) or chat handles rather than verified enterprise domains."
        })

    # 3. Check for sensitive personal information solicitation
    for pattern in SENSITIVE_INFO_PATTERNS:
        if re.search(pattern, full_text):
            score += 30
            warnings.append({
                "signal": "Sensitive Banking / Identity Details Requested",
                "detail": "Listing asks for confidential bank credentials, direct deposit passwords, or sensitive identity copies upfront."
            })
            break

    # 4. Check for high urgency or unrealistic low-skill wage claims
    for pattern in URGENT_PRESSURE_PATTERNS:
        if re.search(pattern, full_text):
            score += 15
            warnings.append({
                "signal": "Extreme Urgency or Get-Rich-Quick Wording",
                "detail": "Listing emphasizes immediate hiring without credentials or advertises disproportionately high wages for trivial tasks."
            })
            break

    # 5. Salary outlier check for entry-level data entry
    if "data entry" in title.lower() and exp_level == "Entry-level" and salary_max > 2500000:
        score += 20
        warnings.append({
            "signal": "Disproportionate Compensation for Role Scope",
            "detail": f"Advertised compensation ({job.get('salary_display')}) is unusually inflated for entry-level routine tasks, a frequent hallmark of recruitment phishing."
        })

    # 6. Vague employer or brief description check
    if len(description.split()) < 20:
        score += 10
        warnings.append({
            "signal": "Very Brief or Vague Job Description",
            "detail": "The listing lacks clear role objectives, company background, and specific operational deliverables."
        })

    # Cap score at 100
    risk_score = min(score, 100)

    if risk_score >= 50:
        risk_level = "High"
    elif risk_score >= 20:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    return {
        "risk_level": risk_level,
        "risk_score": risk_score,
        "warnings": warnings,
        "is_safe": risk_level == "Low",
        "disclaimer": (
            "Notice: This automated checker scans for known suspicious patterns and heuristic risk flags. "
            "It does not provide definitive legal proof of whether a listing is genuine or fraudulent. "
            "Always verify employers independently and never send money for a job."
        )
    }
