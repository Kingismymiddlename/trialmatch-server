import os
import re
import json
import httpx
import uvicorn
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


load_dotenv()

app = FastAPI(title="Clinical Trials Matcher")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_BASE = "https://api.groq.com/openai/v1/chat/completions"

# Correct replacement for deprecated llama-3.3-70b-versatile
GROQ_MODEL = "openai/gpt-oss-120b"

# NCI Clinical Trials API - cancer trials
NCI_BASE = "https://clinicaltrialsapi.cancer.gov/api/v2/trials"

# ClinicalTrials.gov API v2 fallback
CLINICALTRIALS_BASE = "https://clinicaltrials.gov/api/v2/studies"


@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "Clinical Trials Matcher",
        "health": "/health",
        "search_trials": "/search-trials?condition=breast%20cancer",
        "match": "/match",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "provider": "groq",
        "ai": GROQ_MODEL,
        "groq_key_configured": bool(GROQ_API_KEY),
    }


def safe_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, str):
        return value
    return str(value)


def safe_join(values: Any) -> str:
    if not values:
        return ""
    if isinstance(values, list):
        return ", ".join([safe_str(v) for v in values if v])
    return safe_str(values)


def model_to_dict(model: BaseModel) -> Dict[str, Any]:
    """
    Compatible with both Pydantic v1 and v2.
    """
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def normalize_nci_phase(phase_value: Any) -> str:
    if not phase_value:
        return "N/A"

    if isinstance(phase_value, dict):
        return safe_str(phase_value.get("phase"), "N/A")

    if isinstance(phase_value, list):
        return ", ".join([safe_str(p) for p in phase_value if p]) or "N/A"

    return safe_str(phase_value, "N/A")


def normalize_nci_sponsor(lead_org: Any) -> str:
    if not lead_org:
        return ""

    if isinstance(lead_org, dict):
        return safe_str(
            lead_org.get("name")
            or lead_org.get("org_name")
            or lead_org.get("lead_org")
            or ""
        )

    return safe_str(lead_org)


def parse_model_json_object(text: str) -> Dict[str, Any]:
    """
    Parse JSON object from model response.
    Uses direct parsing first, with fallback extraction if extra text appears.
    """
    cleaned_text = text.strip()
    cleaned_text = re.sub(r"^```json\s*", "", cleaned_text)
    cleaned_text = re.sub(r"^```\s*", "", cleaned_text)
    cleaned_text = re.sub(r"\s*```$", "", cleaned_text).strip()

    try:
        parsed = json.loads(cleaned_text)
        if isinstance(parsed, dict):
            return parsed
        raise ValueError("Model response was not a JSON object.")
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", cleaned_text)
        if not match:
            raise ValueError("Could not parse response as JSON object.")

        parsed = json.loads(match.group())

        if not isinstance(parsed, dict):
            raise ValueError("Parsed response was not a JSON object.")

        return parsed


@app.get("/search-trials")
async def search_trials(
    condition: str,
    location: str = "",
    max_results: int = Query(default=20, ge=1, le=50),
):
    trials: List[Dict[str, Any]] = []

    # Try NCI API first.
    try:
        params: Dict[str, Any] = {
            "diseases.name": condition,
            "current_trial_status": "Active",
            "size": min(max_results, 20),
            "include": [
                "nct_id",
                "brief_title",
                "brief_summary",
                "eligibility",
                "principal_investigator",
                "sites",
                "phase",
                "diseases",
                "arms",
                "lead_org",
            ],
        }

        if location:
            params["sites.org_country"] = location

        headers = {"accept": "application/json"}

        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(NCI_BASE, params=params, headers=headers)

            if resp.status_code == 200:
                data = resp.json()

                for trial in data.get("data", []):
                    eligibility = trial.get("eligibility") or {}
                    structured = eligibility.get("structured") or {}
                    unstructured = eligibility.get("unstructured") or []

                    eligibility_text = " ".join(
                        [
                            safe_str(item.get("description"))
                            for item in unstructured
                            if isinstance(item, dict)
                        ]
                    )[:600]

                    sites = trial.get("sites") or []
                    locations = []

                    for site in sites[:3]:
                        if not isinstance(site, dict):
                            continue

                        city = safe_str(site.get("org_city"))
                        country = safe_str(site.get("org_country"))

                        if city or country:
                            locations.append(f"{city}, {country}".strip(", "))

                    arms = trial.get("arms") or []
                    interventions = []

                    for arm in arms[:3]:
                        if not isinstance(arm, dict):
                            continue

                        arm_interventions = arm.get("interventions") or []

                        if arm_interventions and isinstance(arm_interventions[0], dict):
                            intervention_name = arm_interventions[0].get(
                                "intervention_name"
                            )
                            if intervention_name:
                                interventions.append(safe_str(intervention_name))

                    diseases = [
                        safe_str(disease.get("name"))
                        for disease in (trial.get("diseases") or [])[:3]
                        if isinstance(disease, dict) and disease.get("name")
                    ]

                    trials.append(
                        {
                            "nct_id": safe_str(trial.get("nct_id")),
                            "title": safe_str(trial.get("brief_title"), "No title"),
                            "summary": safe_str(trial.get("brief_summary"))[:400],
                            "eligibility": eligibility_text,
                            "min_age": safe_str(structured.get("min_age_in_years")),
                            "max_age": safe_str(structured.get("max_age_in_years")),
                            "sex": safe_str(structured.get("gender"), "ALL"),
                            "phase": normalize_nci_phase(trial.get("phase")),
                            "sponsor": normalize_nci_sponsor(trial.get("lead_org")),
                            "conditions": diseases,
                            "interventions": interventions,
                            "locations": locations,
                        }
                    )

    except Exception:
        # NCI lookup is best-effort. Fall back to ClinicalTrials.gov.
        pass

    # Fallback: ClinicalTrials.gov API v2.
    if not trials:
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (compatible; clinical-trials-matcher/1.0)",
                "Accept": "application/json",
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": "https://clinicaltrials.gov/",
            }

            params: Dict[str, Any] = {
                "query.cond": condition,
                "filter.overallStatus": "RECRUITING",
                "pageSize": min(max_results, 20),
                "format": "json",
            }

            if location:
                params["query.locn"] = location

            async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
                resp = await client.get(
                    CLINICALTRIALS_BASE,
                    params=params,
                    headers=headers,
                )

                if resp.status_code == 200:
                    data = resp.json()

                    for study in data.get("studies", []):
                        try:
                            protocol = study.get("protocolSection") or {}

                            identification_module = (
                                protocol.get("identificationModule") or {}
                            )
                            description_module = protocol.get("descriptionModule") or {}
                            eligibility_module = protocol.get("eligibilityModule") or {}
                            design_module = protocol.get("designModule") or {}
                            sponsor_module = (
                                protocol.get("sponsorCollaboratorsModule") or {}
                            )
                            conditions_module = protocol.get("conditionsModule") or {}
                            interventions_module = (
                                protocol.get("armsInterventionsModule") or {}
                            )
                            contacts_module = (
                                protocol.get("contactsLocationsModule") or {}
                            )

                            locations = []

                            for loc in (contacts_module.get("locations") or [])[:3]:
                                city = safe_str(loc.get("city"))
                                country = safe_str(loc.get("country"))

                                if city or country:
                                    locations.append(
                                        f"{city}, {country}".strip(", ")
                                    )

                            interventions = [
                                safe_str(intervention.get("name"))
                                for intervention in (
                                    interventions_module.get("interventions") or []
                                )[:3]
                                if intervention.get("name")
                            ]

                            phases = design_module.get("phases") or []

                            trials.append(
                                {
                                    "nct_id": safe_str(
                                        identification_module.get("nctId")
                                    ),
                                    "title": safe_str(
                                        identification_module.get("briefTitle"),
                                        "No title",
                                    ),
                                    "summary": safe_str(
                                        description_module.get("briefSummary")
                                    )[:400],
                                    "eligibility": safe_str(
                                        eligibility_module.get("eligibilityCriteria")
                                    )[:600],
                                    "min_age": safe_str(
                                        eligibility_module.get("minimumAge")
                                    ),
                                    "max_age": safe_str(
                                        eligibility_module.get("maximumAge")
                                    ),
                                    "sex": safe_str(
                                        eligibility_module.get("sex"),
                                        "ALL",
                                    ),
                                    "phase": safe_join(phases) if phases else "N/A",
                                    "sponsor": safe_str(
                                        sponsor_module.get("leadSponsor", {}).get(
                                            "name"
                                        )
                                    ),
                                    "conditions": (
                                        conditions_module.get("conditions") or []
                                    )[:3],
                                    "interventions": interventions,
                                    "locations": locations,
                                }
                            )

                        except Exception:
                            continue

        except Exception:
            pass

    if not trials:
        return {
            "trials": [],
            "error": "Could not fetch trials from any source. Please try again.",
        }

    return {"trials": trials}


class ClinicalTrial(BaseModel):
    nct_id: str = ""
    title: str = "No title"
    summary: str = ""
    eligibility: str = ""
    min_age: str = ""
    max_age: str = ""
    sex: str = "ALL"
    phase: str = "N/A"
    sponsor: str = ""
    conditions: List[str] = Field(default_factory=list)
    interventions: List[str] = Field(default_factory=list)
    locations: List[str] = Field(default_factory=list)


class MatchRequest(BaseModel):
    patient: Dict[str, Any]
    trials: List[ClinicalTrial]


@app.post("/match")
async def match(req: MatchRequest):
    if not GROQ_API_KEY:
        return {"error": "GROQ_API_KEY not configured on server."}

    if not req.trials:
        return {"error": "No trials provided for matching."}

    patient_str = json.dumps(req.patient, indent=2)

    trials_str = "\n\n---\n\n".join(
        [
            (
                f"Trial {index + 1}: {trial.title}\n"
                f"NCT ID: {trial.nct_id}\n"
                f"Phase: {trial.phase}\n"
                f"Sponsor: {trial.sponsor}\n"
                f"Conditions: {', '.join(trial.conditions)}\n"
                f"Interventions: {', '.join(trial.interventions)}\n"
                f"Locations: {', '.join(trial.locations)}\n"
                f"Age Range: {trial.min_age or 'N/A'} - {trial.max_age or 'N/A'}\n"
                f"Sex: {trial.sex}\n"
                f"Eligibility: {trial.eligibility}\n"
                f"Summary: {trial.summary}"
            )
            for index, trial in enumerate(req.trials[:15])
        ]
    )

    prompt = f"""
Match this patient to the most suitable clinical trials.

Patient Profile:
{patient_str}

Clinical Trials:
{trials_str}

Return ONLY a valid JSON object with this exact structure:
{{
  "matches": [
    {{
      "nct_id": "string",
      "score": 0,
      "match": "Strong Match OR Possible Match OR Weak Match",
      "reasons": ["reason 1", "reason 2"],
      "concerns": ["concern 1"],
      "recommendation": "1-2 sentence recommendation"
    }}
  ]
}}

Rules:
- The "score" must be an integer from 0 to 100.
- The "match" value must be exactly one of: "Strong Match", "Possible Match", "Weak Match".
- The "reasons" array must contain 2-3 strings.
- The "concerns" array must contain 0-2 strings.
- Only include trials with score above 20.
- Include maximum 8 trials.
- Sort matches by score descending.
- Do not include markdown.
- Do not include backticks.
- Do not include text before or after the JSON object.
""".strip()

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                GROQ_BASE,
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": GROQ_MODEL,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are a clinical trial matching JSON API. "
                                "Output only valid JSON objects. Do not output markdown."
                            ),
                        },
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ],
                    "temperature": 0.1,
                    "max_tokens": 2000,
                    "response_format": {"type": "json_object"},
                },
            )

            resp.raise_for_status()
            data = resp.json()

            if "error" in data:
                return {
                    "error": data["error"].get("message", "Groq API error")
                }

            text = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )

            if not text:
                return {"error": "Empty response from model."}

            parsed = parse_model_json_object(text)
            matches = parsed.get("matches", [])

            if not isinstance(matches, list):
                return {
                    "error": "Model response did not contain a valid matches array."
                }

            trial_map = {
                trial.nct_id: model_to_dict(trial)
                for trial in req.trials
            }

            cleaned_matches = []

            for item in matches:
                if not isinstance(item, dict):
                    continue

                nct_id = safe_str(item.get("nct_id"))
                score = item.get("score", 0)

                try:
                    score = int(score)
                except Exception:
                    score = 0

                if score <= 20:
                    continue

                match_label = safe_str(item.get("match"))

                if match_label not in {
                    "Strong Match",
                    "Possible Match",
                    "Weak Match",
                }:
                    match_label = (
                        "Strong Match"
                        if score >= 75
                        else "Possible Match"
                        if score >= 45
                        else "Weak Match"
                    )

                reasons = item.get("reasons", [])
                concerns = item.get("concerns", [])

                if not isinstance(reasons, list):
                    reasons = [safe_str(reasons)] if reasons else []

                if not isinstance(concerns, list):
                    concerns = [safe_str(concerns)] if concerns else []

                cleaned_matches.append(
                    {
                        "nct_id": nct_id,
                        "score": max(0, min(score, 100)),
                        "match": match_label,
                        "reasons": [safe_str(reason) for reason in reasons[:3]],
                        "concerns": [safe_str(concern) for concern in concerns[:2]],
                        "recommendation": safe_str(item.get("recommendation")),
                        "trial": trial_map.get(nct_id, {}),
                    }
                )

            cleaned_matches.sort(key=lambda item: item["score"], reverse=True)

            return {"matches": cleaned_matches[:8]}

    except httpx.HTTPStatusError as e:
        return {
            "error": f"Groq API HTTP error: {e.response.status_code}",
            "details": e.response.text,
        }
    except json.JSONDecodeError:
        return {"error": "Model returned invalid JSON."}
    except Exception as e:
        return {"error": f"Trial matching failed: {str(e)}"}


# Mount static frontend only if the folder exists.
static_dir = Path("static")

if static_dir.exists() and static_dir.is_dir():
    app.mount("/app", StaticFiles(directory="static", html=True), name="static")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
