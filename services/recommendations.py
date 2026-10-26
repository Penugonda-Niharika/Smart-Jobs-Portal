"""
Skill-based job recommendation service.
Compares a user's acquired skill set with job requirements to compute
match percentages, identify skill gaps, and provide heuristic ranking.
"""

def normalize_skill(skill: str) -> str:
    """Normalize skill string for case-insensitive matching."""
    return skill.strip().lower()


def compute_skill_match(user_skills: list[str], job_required_skills: list[str], job_preferred_skills: list[str] = None) -> dict:
    """
    Computes matching and missing skills along with a heuristic match score.
    Weights required skills at 80% and preferred skills at 20%.
    Gracefully handles empty lists and avoids division by zero.
    """
    if not job_required_skills and not job_preferred_skills:
        return {
            "score": 0,
            "matched_skills": [],
            "missing_skills": [],
            "match_ratio": 0.0,
            "explanation": "This job does not specify required skills."
        }

    job_preferred_skills = job_preferred_skills or []
    user_skill_set = {normalize_skill(s) for s in user_skills if s and s.strip()}
    
    if not user_skill_set:
        return {
            "score": 0,
            "matched_skills": [],
            "missing_skills": job_required_skills,
            "match_ratio": 0.0,
            "explanation": "Add skills to your profile or upload a resume to receive match scores."
        }

    # Match required skills
    matched_required = []
    missing_required = []
    for skill in job_required_skills:
        if normalize_skill(skill) in user_skill_set:
            matched_required.append(skill)
        else:
            missing_required.append(skill)

    # Match preferred skills
    matched_preferred = []
    missing_preferred = []
    for skill in job_preferred_skills:
        if normalize_skill(skill) in user_skill_set:
            matched_preferred.append(skill)
        else:
            missing_preferred.append(skill)

    req_count = len(job_required_skills)
    pref_count = len(job_preferred_skills)

    # Calculate weighted percentage
    req_score = (len(matched_required) / req_count * 100) if req_count > 0 else 0
    pref_score = (len(matched_preferred) / pref_count * 100) if pref_count > 0 else 0

    if req_count > 0 and pref_count > 0:
        composite_score = round(req_score * 0.8 + pref_score * 0.2)
    elif req_count > 0:
        composite_score = round(req_score)
    else:
        composite_score = round(pref_score)

    all_matched = matched_required + matched_preferred
    all_missing = missing_required

    # Generate transparent explanation
    if composite_score >= 80:
        explanation = f"Strong match: You possess {len(matched_required)} of {req_count} essential skills for this role."
    elif composite_score >= 50:
        explanation = f"Moderate match: You have key foundational skills, but would benefit from learning {', '.join(missing_required[:2]) if missing_required else 'additional tools'}."
    elif composite_score > 0:
        explanation = f"Partial match: You match on {', '.join(all_matched)}, with several core requirements remaining."
    else:
        explanation = "Low direct skill overlap. Consider reviewing required prerequisites in our Career Roadmap."

    return {
        "score": composite_score,
        "matched_skills": all_matched,
        "missing_skills": all_missing,
        "match_ratio": round(composite_score / 100.0, 2),
        "explanation": explanation
    }


def get_recommendations(user_skills: list[str], jobs: list[dict], min_score: int = 15, limit: int = 10) -> list[dict]:
    """
    Ranks available job listings for a user based on their skills.
    Returns ranked listings with score breakdowns.
    """
    recommendations = []
    for job in jobs:
        # Don't recommend demo suspicious test jobs in standard recommendations
        if "DEMO TEST" in job.get("title", ""):
            continue

        req_skills = job.get("required_skills", [])
        pref_skills = job.get("preferred_skills", [])
        
        match_info = compute_skill_match(user_skills, req_skills, pref_skills)
        
        if match_info["score"] >= min_score or len(user_skills) == 0:
            rec_item = {
                "job": job,
                "score": match_info["score"],
                "matched_skills": match_info["matched_skills"],
                "missing_skills": match_info["missing_skills"],
                "explanation": match_info["explanation"]
            }
            recommendations.append(rec_item)

    # Sort descending by match score
    recommendations.sort(key=lambda item: item["score"], reverse=True)
    return recommendations[:limit]
