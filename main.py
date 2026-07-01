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
from fastapi.responses import HTMLResponse
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
GROQ_MODEL = "openai/gpt-oss-120b"

NCI_BASE = "https://clinicaltrialsapi.cancer.gov/api/v2/trials"
CLINICALTRIALS_BASE = "https://clinicaltrials.gov/api/v2/studies"


FRONTEND_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Trial Match</title>
  <style>
    * {
      box-sizing: border-box;
    }

    body {
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      background: #f4f7fb;
      color: #172033;
    }

    .header {
      background: linear-gradient(135deg, #1f4fd8, #0f766e);
      color: white;
      padding: 32px 20px;
      text-align: center;
    }

    .header h1 {
      margin: 0;
      font-size: 34px;
    }

    .header p {
      margin: 10px 0 0;
      opacity: 0.95;
    }

    .container {
      max-width: 1150px;
      margin: 24px auto;
      padding: 0 16px;
    }

    .grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 18px;
    }

    @media (max-width: 850px) {
      .grid {
        grid-template-columns: 1fr;
      }
    }

    .card {
      background: white;
      border-radius: 14px;
      padding: 20px;
      box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
      border: 1px solid #e5e7eb;
    }

    label {
      display: block;
      margin-top: 12px;
      margin-bottom: 6px;
      font-weight: 700;
      color: #263248;
    }

    input,
    textarea {
      width: 100%;
      padding: 11px;
      border-radius: 9px;
      border: 1px solid #cbd5e1;
      font-size: 14px;
      background: #fff;
    }

    textarea {
      min-height: 230px;
      resize: vertical;
      font-family: Consolas, Monaco, monospace;
    }

    button {
      margin-top: 14px;
      padding: 12px 18px;
      border: none;
      border-radius: 9px;
      background: #1f4fd8;
      color: white;
      font-weight: 700;
      cursor: pointer;
      font-size: 15px;
    }

    button:hover {
      background: #193fb0;
    }

    button.secondary {
      background: #0f766e;
    }

    button.secondary:hover {
      background: #0b5f59;
    }

    .status {
      margin-top: 12px;
      font-size: 14px;
      color: #475569;
    }

    .trial,
    .match {
      border: 1px solid #e2e8f0;
      border-radius: 12px;
      padding: 14px;
      margin-top: 12px;
      background: #fbfdff;
    }

    .trial h3,
    .match h3 {
      margin: 0 0 8px;
      color: #0f172a;
    }

    .meta {
      color: #475569;
      font-size: 13px;
      line-height: 1.5;
    }

    .pill {
      display: inline-block;
      padding: 4px 9px;
      border-radius: 999px;
      background: #e0f2fe;
      color: #075985;
      margin: 4px 4px 4px 0;
      font-size: 12px;
      font-weight: 700;
    }

    .score {
      font-size: 22px;
      font-weight: 800;
      color: #0f766e;
    }

    .error {
      color: #b91c1c;
      background: #fee2e2;
      border: 1px solid #fecaca;
      padding: 10px;
      border-radius: 8px;
      margin-top: 12px;
      white-space: pre-wrap;
    }

    .success {
      color: #166534;
      background: #dcfce7;
      border: 1px solid #bbf7d0;
      padding: 10px;
      border-radius: 8px;
      margin-top: 12px;
    }

    .full {
      margin-top: 18px;
    }

    .small {
      font-size: 12px;
      color: #64748b;
      margin-top: 8px;
      line-height: 1.5;
    }
  </style>
</head>
<body>
  <div class="header">
    <h1>Trial Match</h1>
    <p>Search recruiting clinical trials and match them against a patient profile using AI.</p>
  </div>

  <div class="container">
    <div class="grid">
      <div class="card">
        <h2>1. Search Clinical Trials</h2>

        <label for="condition">Condition</label>
        <input id="condition" value="breast cancer" placeholder="Example: breast cancer" />

        <label for="location">Location / Country</label>
        <input id="location" value="" placeholder="Optional. Example: United States" />

        <label for="maxResults">Max Results</label>
        <input id="maxResults" type="number" min="1" max="50" value="10" />

        <button onclick="searchTrials()">Search Trials</button>

        <div id="searchStatus" class="status"></div>
      </div>

      <div class="card">
        <h2>2. Patient Profile</h2>

        <label for="patientProfile">Patient JSON</label>
        <textarea id="patientProfile">{
  "age": 55,
  "sex": "Female",
  "condition": "Breast cancer",
  "stage": "Stage II",
  "biomarkers": "HER2 positive",
  "prior_treatments": "Surgery and chemotherapy",
  "location": "United States",
  "notes": "Looking for recruiting interventional trials"
}</textarea>

        <button class="secondary" onclick="matchTrials()">Match Patient to Trials</button>

        <div id="matchStatus" class="status"></div>

        <div class="small">
          This tool is for research support only. It does not provide medical advice or replace clinician review.
        </div>
      </div>
    </div>

    <div class="card full">
      <h2>Search Results</h2>
      <div id="trials"></div>
    </div>

    <div class="card full">
      <h2>AI Matches</h2>
      <div id="matches"></div>
    </div>
  </div>

  <script>
    let loadedTrials = [];

    function escapeHtml(value) {
      return String(value || "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }

    function showError(elementId, message) {
      document.getElementById(elementId).innerHTML =
        '<div class="error">' + escapeHtml(message) + '</div>';
    }

    function showSuccess(elementId, message) {
      document.getElementById(elementId).innerHTML =
        '<div class="success">' + escapeHtml(message) + '</div>';
    }

    async function searchTrials() {
      const condition = document.getElementById("condition").value.trim();
      const location = document.getElementById("location").value.trim();
      const maxResults = document.getElementById("maxResults").value || 10;

      if (!condition) {
        showError("searchStatus", "Please enter a condition.");
        return;
      }

      document.getElementById("searchStatus").innerHTML = "Searching clinical trials...";
      document.getElementById("trials").innerHTML = "";
      document.getElementById("matches").innerHTML = "";

      try {
        const params = new URLSearchParams({
          condition: condition,
          max_results: maxResults
        });

        if (location) {
          params.append("location", location);
        }

        const response = await fetch("/search-trials?" + params.toString());
        const data = await response.json();

        if (!response.ok || data.error) {
          throw new Error(data.error || "Search failed.");
        }

        loadedTrials = data.trials || [];

        if (!loadedTrials.length) {
          showError("searchStatus", "No trials found.");
          return;
        }

        showSuccess("searchStatus", "Found " + loadedTrials.length + " trial(s).");

        const html = loadedTrials.map((trial, index) => {
          const conditions = (trial.conditions || []).map(c => '<span class="pill">' + escapeHtml(c) + '</span>').join("");
          const interventions = (trial.interventions || []).map(i => '<span class="pill">' + escapeHtml(i) + '</span>').join("");

          return `
            <div class="trial">
              <h3>${index + 1}. ${escapeHtml(trial.title)}</h3>
              <div class="meta">
                <b>NCT ID:</b> ${escapeHtml(trial.nct_id || "N/A")}<br/>
                <b>Phase:</b> ${escapeHtml(trial.phase || "N/A")}<br/>
                <b>Sponsor:</b> ${escapeHtml(trial.sponsor || "N/A")}<br/>
                <b>Age Range:</b> ${escapeHtml(trial.min_age || "N/A")} - ${escapeHtml(trial.max_age || "N/A")}<br/>
                <b>Sex:</b> ${escapeHtml(trial.sex || "ALL")}<br/>
                <b>Locations:</b> ${escapeHtml((trial.locations || []).join(", ") || "N/A")}
              </div>
              <div>${conditions}</div>
              <div>${interventions}</div>
              <p>${escapeHtml(trial.summary || "")}</p>
            </div>
          `;
        }).join("");

        document.getElementById("trials").innerHTML = html;

      } catch (error) {
        loadedTrials = [];
        showError("searchStatus", error.message || "Search failed.");
      }
    }

    async function matchTrials() {
      if (!loadedTrials.length) {
        showError("matchStatus", "Please search trials first.");
        return;
      }

      let patient;

      try {
        patient = JSON.parse(document.getElementById("patientProfile").value);
      } catch (error) {
        showError("matchStatus", "Invalid patient JSON. Please correct it.");
        return;
      }

      document.getElementById("matchStatus").innerHTML = "Matching patient to trials...";
      document.getElementById("matches").innerHTML = "";

      try {
        const response = await fetch("/match", {
          method: "POST",
          headers: {
            "Content-Type": "application/json"
          },
          body: JSON.stringify({
            patient: patient,
            trials: loadedTrials
          })
        });

        const data = await response.json();

        if (!response.ok || data.error) {
          throw new Error(data.error || "Matching failed.");
        }

        const matches = data.matches || [];

        if (!matches.length) {
          showError("matchStatus", "No suitable matches found above the score threshold.");
          return;
        }

        showSuccess("matchStatus", "Generated " + matches.length + " match(es).");

        const html = matches.map((item, index) => {
          const reasons = (item.reasons || []).map(r => "<li>" + escapeHtml(r) + "</li>").join("");
          const concerns = (item.concerns || []).map(c => "<li>" + escapeHtml(c) + "</li>").join("");
          const trial = item.trial || {};

          return `
            <div class="match">
              <h3>${index + 1}. ${escapeHtml(trial.title || item.nct_id)}</h3>
              <div class="score">${escapeHtml(item.score)} / 100</div>
              <div class="meta">
                <b>Match:</b> ${escapeHtml(item.match)}<br/>
                <b>NCT ID:</b> ${escapeHtml(item.nct_id)}
              </div>

              <h4>Reasons</h4>
              <ul>${reasons}</ul>

              <h4>Concerns</h4>
              <ul>${concerns || "<li>No major concerns listed.</li>"}</ul>

              <h4>Recommendation</h4>
              <p>${escapeHtml(item.recommendation)}</p>
            </div>
          `;
        }).join("");

        document.getElementById("matches").innerHTML = html;

      } catch (error) {
        showError("matchStatus", error.message || "Matching failed.");
      }
    }
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def root():
    return HTMLResponse(content=FRONTEND_HTML)


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
        pass

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


# Optional: serve additional static assets from /static if the folder exists.
static_dir = Path("static")

if static_dir.exists() and static_dir.is_dir():
    app.mount("/static", StaticFiles(directory="static"), name="static")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
