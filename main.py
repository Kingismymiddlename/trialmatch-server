import os
import re
import json
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

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
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_BASE = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"


@app.get("/health")
def health():
    return {"status": "ok", "ai": "groq/llama-3.3-70b"}


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
async def match(req: MatchRequest):
    if not GROQ_API_KEY:
        return {"error": "GROQ_API_KEY not configured on server."}

    patient_str = json.dumps(req.patient, indent=2)
    trials_str = "\n\n---\n\n".join([
        f"Trial {i+1}: {t['title']}\nNCT ID: {t['nct_id']}\nPhase: {t['phase']}\nSponsor: {t['sponsor']}\nConditions: {', '.join(t['conditions'])}\nInterventions: {', '.join(t['interventions'])}\nLocations: {', '.join(t['locations'])}\nAge Range: {t['min_age']} - {t['max_age']}\nSex: {t['sex']}\nEligibility: {t['eligibility']}\nSummary: {t['summary']}"
        for i, t in enumerate(req.trials[:15])
    ])

    prompt = f"""Match this patient to the most suitable clinical trials.

Patient Profile:
{patient_str}

Clinical Trials:
{trials_str}

Respond with ONLY a raw JSON array starting with [ and ending with ]. No markdown, no code fences, no explanation.

Each item must have:
- nct_id: string
- score: integer 0-100
- match: exactly one of: Strong Match, Possible Match, Weak Match, Not Eligible
- reasons: array of 2-3 strings explaining why it fits
- concerns: array of 1-2 strings (or empty array)
- recommendation: 1-2 sentence string

Score labels: Strong Match (80-100), Possible Match (50-79), Weak Match (20-49), Not Eligible (0-19).
Only include trials with score above 20. Sort by score descending. Max 8 trials."""

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            GROQ_BASE,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": GROQ_MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a JSON API. You only output raw JSON arrays with no markdown, no code fences, no explanation. Every response starts with [ and ends with ]."
                    },
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1,
                "max_tokens": 2000
            }
        )
        data = resp.json()
        if "error" in data:
            return {"error": data["error"].get("message", "Groq API error")}

        text = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

        # Strip markdown fences
        text = re.sub(r'^```json\s*', '', text)
        text = re.sub(r'^```\s*', '', text)
        text = re.sub(r'\s*```$', '', text).strip()

        # Try direct parse
        try:
            matches = json.loads(text)
        except Exception:
            match_result = re.search(r'\[[\s\S]*\]', text)
            if not match_result:
                return {"error": f"Could not parse response: {text[:300]}"}
            try:
                matches = json.loads(match_result.group())
            except Exception:
                return {"error": f"Could not parse response: {text[:300]}"}

        trial_map = {t["nct_id"]: t for t in req.trials}
        for m in matches:
            m["trial"] = trial_map.get(m["nct_id"], {})

        return {"matches": matches}


app.mount("/", StaticFiles(directory="static", html=True), name="static")
