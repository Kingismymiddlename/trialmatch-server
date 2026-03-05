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

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_BASE = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"

# NCI Clinical Trials API - more server-friendly than ClinicalTrials.gov
NCI_BASE = "https://clinicaltrialsapi.cancer.gov/api/v2/trials"


@app.get("/health")
def health():
    return {"status": "ok", "ai": "groq/llama-3.3-70b"}


@app.get("/search-trials")
async def search_trials(condition: str, location: str = "", max_results: int = 20):
    # Try NCI API first (cancer trials, very reliable)
    trials = []
    try:
        params = {
            "diseases.name": condition,
            "current_trial_status": "Active",
            "size": min(max_results, 20),
            "include": ["nct_id", "brief_title", "brief_summary", "eligibility",
                        "principal_investigator", "sites", "phase", "diseases",
                        "arms", "lead_org"]
        }
        if location:
            params["sites.org_country"] = location

        headers = {"accept": "application/json"}

        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(NCI_BASE, params=params, headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                for t in data.get("data", []):
                    elig = t.get("eligibility", {})
                    structured = elig.get("structured", {})
                    unstructured = elig.get("unstructured", [])
                    elig_text = " ".join([u.get("description", "") for u in unstructured])[:600]

                    sites = t.get("sites", [])[:3]
                    locations = [f"{s.get('org_city','')}, {s.get('org_country','')}".strip(", ") for s in sites if s.get('org_city') or s.get('org_country')]

                    arms = t.get("arms", [])[:3]
                    interventions = [a.get("interventions", [{}])[0].get("intervention_name", "") for a in arms if a.get("interventions")]

                    diseases = [d.get("name", "") for d in t.get("diseases", [])[:3]]

                    trials.append({
                        "nct_id": t.get("nct_id", ""),
                        "title": t.get("brief_title", "No title"),
                        "summary": (t.get("brief_summary", "") or "")[:400],
                        "eligibility": elig_text,
                        "min_age": str(structured.get("min_age_in_years", "")),
                        "max_age": str(structured.get("max_age_in_years", "")),
                        "sex": structured.get("gender", "ALL"),
                        "phase": t.get("phase", {}).get("phase", "N/A") if isinstance(t.get("phase"), dict) else str(t.get("phase", "N/A")),
                        "sponsor": t.get("lead_org", ""),
                        "conditions": diseases,
                        "interventions": [i for i in interventions if i],
                        "locations": locations,
                    })
    except Exception as e:
        pass

    # Fallback: ClinicalTrials.gov with aggressive headers
    if not trials:
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (compatible; research-tool/1.0; +https://github.com)",
                "Accept": "application/json",
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": "https://clinicaltrials.gov/",
            }
            params = {
                "query.cond": condition,
                "filter.overallStatus": "RECRUITING",
                "pageSize": min(max_results, 20),
                "format": "json"
            }
            if location:
                params["query.locn"] = location

            async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
                resp = await client.get(
                    "https://clinicaltrials.gov/api/v2/studies",
                    params=params,
                    headers=headers
                )
                if resp.status_code == 200:
                    data = resp.json()
                    for s in data.get("studies", []):
                        try:
                            pm = s.get("protocolSection", {})
                            id_mod = pm.get("identificationModule", {})
                            desc_mod = pm.get("descriptionModule", {})
                            elig_mod = pm.get("eligibilityModule", {})
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

                            interventions = [iv.get("name", "") for iv in interventions_mod.get("interventions", [])[:3] if iv.get("name")]
                            phases = design_mod.get("phases", [])

                            trials.append({
                                "nct_id": id_mod.get("nctId", ""),
                                "title": id_mod.get("briefTitle", "No title"),
                                "summary": (desc_mod.get("briefSummary", "") or "")[:400],
                                "eligibility": (elig_mod.get("eligibilityCriteria", "") or "")[:600],
                                "min_age": elig_mod.get("minimumAge", ""),
                                "max_age": elig_mod.get("maximumAge", ""),
                                "sex": elig_mod.get("sex", "ALL"),
                                "phase": ", ".join(phases) if phases else "N/A",
                                "sponsor": sponsor_mod.get("leadSponsor", {}).get("name", ""),
                                "conditions": conditions_mod.get("conditions", [])[:3],
                                "interventions": interventions,
                                "locations": locations,
                            })
                        except Exception:
                            continue
        except Exception:
            pass

    if not trials:
        return {"trials": [], "error": "Could not fetch trials from any source. Please try again."}

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
- match: exactly one of: Strong Match, Possible Match, Weak Match
- reasons: array of 2-3 strings explaining why it fits
- concerns: array of 0-2 strings
- recommendation: 1-2 sentence string

Sort by score descending. Only include score above 20. Max 8 trials."""

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            GROQ_BASE,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": GROQ_MODEL,
                "messages": [
                    {"role": "system", "content": "You are a JSON API. Output only raw JSON arrays. No markdown. No explanation. Start with [ end with ]."},
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
        text = re.sub(r'^```json\s*', '', text)
        text = re.sub(r'^```\s*', '', text)
        text = re.sub(r'\s*```$', '', text).strip()

        try:
            matches = json.loads(text)
        except Exception:
            m = re.search(r'\[[\s\S]*\]', text)
            if not m:
                return {"error": f"Could not parse response: {text[:300]}"}
            try:
                matches = json.loads(m.group())
            except Exception:
                return {"error": f"Could not parse response: {text[:300]}"}

        trial_map = {t["nct_id"]: t for t in req.trials}
        for m in matches:
            m["trial"] = trial_map.get(m["nct_id"], {})

        return {"matches": matches}


app.mount("/", StaticFiles(directory="static", html=True), name="static")
