"""
Career guidance and skill-gap analysis service.
Loads predefined role skill frameworks and produces gap analyses and ordered roadmaps.
"""

import json
from pathlib import Path
from typing import Dict, Any, List

SKILLS_MAP_PATH = Path(__file__).resolve().parent.parent / "data" / "skills_map.json"


def load_skills_map() -> Dict[str, Any]:
    """Loads skill maps from data/skills_map.json with fallback."""
    if not SKILLS_MAP_PATH.exists():
        return {"roles": {}}
    try:
        with open(SKILLS_MAP_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"roles": {}}


def analyze_career_gap(user_skills: List[str], role_key: str, available_jobs: List[dict] = None) -> Dict[str, Any]:
    """
    Compares user skills against target role curriculum and returns:
    - Possessed skills
    - Missing skills
    - Readiness percentage
    - Ordered learning roadmap
    - Matching sample jobs
    """
    data = load_skills_map()
    roles = data.get("roles", {})
    
    if role_key not in roles:
        # Default to first role if key not found
        role_key = next(iter(roles)) if roles else "python_developer"

    role_info = roles.get(role_key, {})
    core_skills = role_info.get("core_skills", [])
    advanced_skills = role_info.get("advanced_skills", [])
    all_target_skills = core_skills + advanced_skills

    user_skills_clean = {s.strip().lower() for s in user_skills if s and s.strip()}

    possessed_core = [s for s in core_skills if s.lower() in user_skills_clean]
    missing_core = [s for s in core_skills if s.lower() not in user_skills_clean]

    possessed_advanced = [s for s in advanced_skills if s.lower() in user_skills_clean]
    missing_advanced = [s for s in advanced_skills if s.lower() not in user_skills_clean]

    total_target = len(all_target_skills)
    total_possessed = len(possessed_core) + len(possessed_advanced)
    readiness_percentage = round((total_possessed / total_target * 100)) if total_target > 0 else 0

    # Determine suggested learning priority based on gaps
    learning_priorities = []
    for skill in missing_core:
        learning_priorities.append({
            "skill": skill,
            "category": "Core Requirement",
            "priority": "High",
            "recommendation": f"Focus on {skill} first as it is required for most {role_info.get('title', 'target')} listings."
        })
    for skill in missing_advanced:
        learning_priorities.append({
            "skill": skill,
            "category": "Advanced Advantage",
            "priority": "Medium",
            "recommendation": f"Learn {skill} to distinguish yourself for competitive mid-level roles."
        })

    # Find relevant jobs in current catalog
    relevant_jobs = []
    if available_jobs:
        target_title = role_info.get("title", "").lower()
        for j in available_jobs:
            if "DEMO TEST" in j.get("title", ""):
                continue
            title_match = target_title.split()[0] in j.get("title", "").lower()
            skill_match = any(s.lower() in [rs.lower() for rs in j.get("required_skills", [])] for s in core_skills[:3])
            if title_match or skill_match:
                relevant_jobs.append(j)

    return {
        "role_key": role_key,
        "role_title": role_info.get("title", "Target Role"),
        "role_category": role_info.get("category", "General"),
        "possessed_skills": possessed_core + possessed_advanced,
        "missing_skills": missing_core + missing_advanced,
        "core_possessed": possessed_core,
        "core_missing": missing_core,
        "readiness_percentage": readiness_percentage,
        "learning_priorities": learning_priorities,
        "roadmap": role_info.get("roadmap", []),
        "relevant_jobs": relevant_jobs[:4],
        "disclaimer": "This career roadmap and skill-gap assessment is generated using local rule-based curriculum mappings."
    }
