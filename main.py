import os
import re
import json
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

TRIALS_BASE = "https://clinicaltrials.gov/api/v2/studies"
SERVER_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/search-trials")
async def search_trials(condition: str, location: str = "", max_results: int = 20):
    params = {
        "query.cond": condition,
        "filter.overallStatus": "RECRUITING",
        "pageSize": max_results,
        "format": "json",
        "fields": "NCTId,BriefTitle,BriefSummary,EligibilityCriteria,MinimumAge,MaximumAge,Sex,OverallStatus,LocationCity,LocationCountry,Phase,Condition,InterventionName,LeadSponsorName"
    }
    if location:
        params["query.locn"] = location

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(TRIALS_BASE, params=params)
        data = resp.json()

    studies = data.get("studies", [])
    trials = []
    for s in studies:
        pm = s.get("protocolSection", {})
        id_mod = pm.get("identificationModule", {})
        desc_mod = pm.get("descriptionModule", {})
        elig_mod = pm.get("eligibilityModule", {})
        status_mod = pm.get("statusModule", {})
        design_mod = pm.get("designModule", {})
        sponsor_mod = pm.get("sponsorCollaboratorsModule", {})
        conditions_mod = pm.get("conditionsModule", {})
        interventions_mod = pm.get("armsInterventionsModule", {})
        contacts_mod = pm.get("contactsLocationsModule", {})

        locations = []
        for loc in contacts_mod.get("locations", [])[:3]:
            city = loc.get("city", "")
            country = loc.get("country", "")
            if city or country:
                locations.append(f"{city}, {country}".strip(", "))

        interventions = [i.get("interventionName", "") for i in interventions_mod.get("interventions", [])[:3]]

        trials.append({
            "nct_id": id_mod.get("nctId", ""),
            "title": id_mod.get("briefTitle", "No title"),
            "summary": desc_mod.get("briefSummary", "")[:400],
            "eligibility": elig_mod.get("eligibilityCriteria", "")[:600],
            "min_age": elig_mod.get("minimumAge", ""),
            "max_age": elig_mod.get("maximumAge", ""),
            "sex": elig_mod.get("sex", "ALL"),
            "status": status_mod.get("overallStatus", ""),
            "phase": ", ".join(design_mod.get("phases", [])),
            "sponsor": sponsor_mod.get("leadSponsor", {}).get("name", ""),
            "conditions": conditions_mod.get("conditions", [])[:3],
            "interventions": interventions,
            "locations": locations,
        })

    return {"trials": trials}


class MatchRequest(BaseModel):
    patient: dict
    trials: list


@app.post("/match")
async def match(req: MatchRequest, x_user_api_key: Optional[str] = Header(default=None)):
    api_key = x_user_api_key or SERVER_API_KEY
    if not api_key:
        return {"error": "No API key provided. Please enter your Anthropic API key."}

    patient_str = json.dumps(req.patient, indent=2)
    trials_str = "\n\n---\n\n".join([
        f"Trial {i+1}: {t['title']}\nNCT ID: {t['nct_id']}\nPhase: {t['phase']}\nSponsor: {t['sponsor']}\nConditions: {', '.join(t['conditions'])}\nInterventions: {', '.join(t['interventions'])}\nLocations: {', '.join(t['locations'])}\nAge Range: {t['min_age']} - {t['max_age']}\nSex: {t['sex']}\nEligibility Criteria: {t['eligibility']}\nSummary: {t['summary']}"
        for i, t in enumerate(req.trials[:15])
    ])

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01"
            },
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 2000,
                "system": """You are an expert clinical trial matcher. Given a patient profile and a list of clinical trials, rank the trials by eligibility match.

Return ONLY a JSON array (no markdown) like this:
[
  {
    "nct_id": "NCT123456",
    "score": 95,
    "match": "Strong Match",
    "reasons": ["Age fits 18-65 range", "Diagnosis matches inclusion criteria", "No conflicting medications"],
    "concerns": ["Must discontinue current medication 2 weeks prior"],
    "recommendation": "1-2 sentence summary of why this trial is a good fit"
  }
]

Score from 0-100. Match labels: Strong Match (80-100), Possible Match (50-79), Weak Match (20-49), Not Eligible (0-19).
Only return trials with score above 20. Sort by score descending. Return max 8 trials.""",
                "messages": [{"role": "user", "content": f"Patient Profile:\n{patient_str}\n\nClinical Trials to evaluate:\n{trials_str}"}]
            }
        )
        data = resp.json()
        if "error" in data:
            return {"error": data["error"]["message"]}

        text = "".join(b["text"] for b in data.get("content", []) if b["type"] == "text")
        match_result = re.search(r'\[[\s\S]*\]', text)
        if not match_result:
            return {"error": "Could not parse matching results"}

        matches = json.loads(match_result.group())
        trial_map = {t["nct_id"]: t for t in req.trials}
        for m in matches:
            m["trial"] = trial_map.get(m["nct_id"], {})

        return {"matches": matches}


app.mount("/", StaticFiles(directory="static", html=True), name="static")
