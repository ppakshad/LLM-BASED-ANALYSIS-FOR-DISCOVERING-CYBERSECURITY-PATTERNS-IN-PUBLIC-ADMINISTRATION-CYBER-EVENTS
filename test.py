import json
import math
import time
import re
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from tqdm import tqdm
from openai import OpenAI


# ============================================================
# Configuration
# ============================================================

# WARNING:
# Hard-coding an API key inside the source code is not recommended.
# Use this only for local testing, and do not upload/share this file.

API_KEY = "publickey"
MODEL = "gpt-4.1-mini"

if not API_KEY or API_KEY == "PASTE_YOUR_NEW_API_KEY_HERE":
    raise ValueError("Please paste your OpenAI API key into the API_KEY variable.")

client = OpenAI(api_key=API_KEY)

# Input JSON file must be in the project directory.
INPUT_JSON_PATH = "cyber_events_2026-05-01.json"

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

PREPARED_DATA_PATH = OUTPUT_DIR / "prepared_public_administration_3000.xlsx"
ANNOTATION_OUTPUT_PATH = OUTPUT_DIR / "llm_m1_m2_m3_annotations.xlsx"
MTOTAL_OUTPUT_PATH = OUTPUT_DIR / "llm_mtotal_hidden_patterns.xlsx"
FINAL_REPORT_PATH = OUTPUT_DIR / "final_event_m1_m2_m3_mtotal.xlsx"

N_RECORDS = 3000
BATCH_SIZE = 30

SELECTED_COLUMNS = [
    "event_date",
    "organization",
    "motive",
    "event_subtype",
    "magnitude",
    "scope",
    "description",
    "country",
    "actor_country",
]


# ============================================================
# Utility Functions
# ============================================================

def normalize_value(value: Any) -> str:
    """
    Normalize missing or non-informative values.
    """
    if value is None:
        return "Undetermined"

    if isinstance(value, float) and pd.isna(value):
        return "Undetermined"

    value_str = str(value).strip()

    if value_str == "":
        return "Undetermined"

    if value_str.lower() in {
        "none",
        "null",
        "none reported",
        "nan",
        "n/a",
        "unknown",
        "undetermined"
    }:
        return "Undetermined"

    return value_str


def clean_text(value: Any) -> str:
    """
    Clean text while preserving meaning.
    """
    value = normalize_value(value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def read_json_dataset(path: str) -> pd.DataFrame:
    """
    Read the raw CyberEvents JSON dataset.

    Supported formats:
    1. List of JSON objects: [{...}, {...}]
    2. Single JSON object: {...}
    """
    input_path = Path(path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        data = [data]

    if not isinstance(data, list):
        raise ValueError("Expected JSON file to contain a list of records or a single JSON object.")

    return pd.DataFrame(data)


def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter Public Administration records, select the target features,
    normalize values, assign EVENT_ID, and keep the first 3000 records.
    """
    if "industry" not in df.columns:
        raise ValueError("The dataset must contain an 'industry' column.")

    public_df = df[
        df["industry"].astype(str).str.strip().str.lower() == "public administration"
    ].copy()

    if public_df.empty:
        raise ValueError("No records found with industry = Public Administration.")

    missing_cols = [col for col in SELECTED_COLUMNS if col not in public_df.columns]
    if missing_cols:
        raise ValueError(f"Missing selected columns in dataset: {missing_cols}")

    # Use slug as EVENT_ID if available. Otherwise, generate artificial IDs.
    if "slug" in public_df.columns:
        event_ids = public_df["slug"].apply(clean_text).tolist()
    else:
        event_ids = [f"PA_{i + 1:06d}" for i in range(len(public_df))]

    public_df = public_df[SELECTED_COLUMNS].copy()

    for col in SELECTED_COLUMNS:
        public_df[col] = public_df[col].apply(clean_text)

    public_df.insert(0, "EVENT_ID", event_ids)

    # Replace missing or duplicate EVENT_ID values.
    seen = set()
    final_event_ids = []

    for i, eid in enumerate(public_df["EVENT_ID"].tolist()):
        if eid == "Undetermined" or eid in seen:
            eid = f"PA_{i + 1:06d}"
        seen.add(eid)
        final_event_ids.append(eid)

    public_df["EVENT_ID"] = final_event_ids

    # Select first 3000 records.
    public_df = public_df.head(N_RECORDS).copy()

    return public_df


def make_batches(df: pd.DataFrame, batch_size: int) -> List[pd.DataFrame]:
    """
    Split dataframe into batches.
    """
    batches = []
    total_batches = math.ceil(len(df) / batch_size)

    for i in range(total_batches):
        start = i * batch_size
        end = start + batch_size
        batches.append(df.iloc[start:end].copy())

    return batches


def batch_to_records(batch_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Convert a dataframe batch into a list of dictionaries.
    """
    return batch_df.to_dict(orient="records")


def extract_json_array(text: str) -> List[Dict[str, Any]]:
    """
    Extract and parse a JSON array from LLM output.
    """
    text = text.strip()

    try:
        parsed = json.loads(text)

        if isinstance(parsed, list):
            return parsed

        if isinstance(parsed, dict):
            if "records" in parsed and isinstance(parsed["records"], list):
                return parsed["records"]
            if "data" in parsed and isinstance(parsed["data"], list):
                return parsed["data"]

    except json.JSONDecodeError:
        pass

    # Fallback: try to extract first JSON array from the output.
    match = re.search(r"\[\s*\{.*\}\s*\]", text, re.DOTALL)

    if match:
        return json.loads(match.group(0))

    raise ValueError("Could not parse a valid JSON array from LLM output.")

def call_llm(prompt: str, max_retries: int = 3, sleep_seconds: int = 5) -> str:
    """
    Call the OpenAI Responses API with retry logic and timeout.
    """
    last_error = None

    for attempt in range(max_retries):
        try:
            print(f"Calling LLM... attempt {attempt + 1}/{max_retries}, prompt length={len(prompt)} chars")

            response = client.responses.create(
                model=MODEL,
                input=prompt,
                timeout=120
            )

            print("LLM response received.")
            return response.output_text

        except Exception as e:
            last_error = e
            print(f"API error on attempt {attempt + 1}/{max_retries}: {e}")
            time.sleep(sleep_seconds)

    raise RuntimeError(f"LLM call failed after {max_retries} attempts: {last_error}")

def safe_list_to_string(value: Any) -> str:
    """
    Convert list/dict values into Excel-friendly strings.
    """
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def save_excel_safely(df: pd.DataFrame, path: Path) -> None:
    """
    Save dataframe to Excel after converting list/dict values.
    """
    temp_df = df.copy()

    for col in temp_df.columns:
        temp_df[col] = temp_df[col].apply(safe_list_to_string)

    temp_df.to_excel(path, index=False)


# ============================================================
# Prompt Builders
# ============================================================

def build_m1_m2_m3_prompt(records: List[Dict[str, Any]]) -> str:
    """
    Build prompt for Stage 1: M1, M2, M3 annotation.
    """
    records_json = json.dumps(records, ensure_ascii=False, indent=2)

    return f"""
You are a senior cybersecurity analyst specializing in Public Administration cyber events.

You will receive a batch of cyber event records. Each record contains:
EVENT_ID, event_date, organization, motive, event_subtype, magnitude, scope, description, country, actor_country.

Your task is to generate three semantic indicators for each record:

M1: Incident Security Semantics
- Explain the security meaning of the incident itself.
- Use description, scope, magnitude, event_subtype, and motive.
- Identify meanings such as ransomware, data exposure, unauthorized access, service disruption, vulnerability exploitation, sabotage, espionage, operational degradation, or other relevant security behavior.

M2: Governance Function Criticality
- Explain what public-sector function is affected.
- Use organization, country, description, and scope.
- Identify functions such as citizen services, public safety, law enforcement, taxation, legal administration, emergency response, public records, municipal operations, government communication, digital public platforms, or general administration.

M3: Cyber Intelligence Gap and Inference
- Explain whether important information is missing, null, unclear, or Undetermined.
- Use motive, event_subtype, magnitude, scope, description, organization, country, actor_country, and event_date.
- Explain whether the remaining evidence still supports useful cyber intelligence inference.

Important rules:
- Do not invent facts.
- Do not confirm attribution or motive unless explicitly stated.
- If something is inferred, clearly describe it as an inference.
- Use only the given record fields.
- Return valid JSON only.
- Return one object per EVENT_ID.
- Do not include markdown or explanations outside JSON.

Required JSON format:
[
  {{
    "EVENT_ID": "example_id",
    "M1": "short but meaningful text",
    "M2": "short but meaningful text",
    "M3": "short but meaningful text",
    "M1_EVIDENCE": ["phrase or field evidence"],
    "M2_EVIDENCE": ["phrase or field evidence"],
    "M3_EVIDENCE": ["phrase or field evidence"],
    "CONFIDENCE": "High/Medium/Low",
    "MANUAL_REVIEW": false
  }}
]

Records:
{records_json}
""".strip()


def build_mtotal_prompt(records: List[Dict[str, Any]]) -> str:
    """
    Build prompt for Stage 2: M_TOTAL generation.
    """
    records_json = json.dumps(records, ensure_ascii=False, indent=2)

    return f"""
You are a senior cybersecurity professor and cyber threat intelligence analyst.

You will receive semantic annotations for Public Administration cyber events.
Each row contains EVENT_ID, M1, M2, and M3.

Your task is to generate M_TOTAL for each record.

M_TOTAL means the higher-level hidden cybersecurity pattern derived by conceptually integrating:
- M1: the security meaning of the incident
- M2: the affected public-sector function
- M3: the cyber intelligence gap and inference value

M_TOTAL should describe the hidden cybersecurity pattern in one concise phrase or sentence.

Use the following pattern taxonomy when possible:
P1: Operational Disruption of Public Functions
P2: Public Safety or Law Enforcement Exposure
P3: Citizen-Service Dependency Disruption
P4: Sensitive Governance Exposure
P5: Data Exposure in Public-Sector Context
P6: High-Value Intelligence Gap
P7: Cross-Border or Geopolitical Cyber Signal
P8: Routine or Low-Semantic Administrative Incident

Decision guidance:
- Use P1 if M1 indicates outage, disruption, ransomware, operational degradation, shutdown, or service unavailability.
- Use P2 if M2 indicates public safety, police, sheriff, fire, dispatch, emergency, 911, correctional, jail, or law enforcement.
- Use P3 if M2 indicates citizen-facing services such as tax, payment, permits, public portal, job applications, benefits, licenses, public records, or municipal services.
- Use P4 if M2 indicates ministry, parliament, court, election, intelligence, defense, senior official, tax authority, national agency, or diplomatic/consular services.
- Use P5 if M1 indicates data exposure, stolen records, leaked files, exposed emails, database compromise, personal data, employee data, or citizen data.
- Use P6 if M3 indicates missing or undetermined motive, actor_country, subtype, scope, or magnitude while M1 or M2 still indicates meaningful impact.
- Use P7 if M3 indicates cross-border actor-country/victim-country difference, political espionage, sabotage, protest, or geopolitical targeting.
- Use P8 only if M1, M2, and M3 do not support any meaningful hidden cybersecurity pattern.

Important rules:
- Do not invent evidence.
- Use only M1, M2, and M3.
- Multiple pattern codes are allowed.
- Return valid JSON only.
- Return one object per EVENT_ID.
- Do not include markdown or explanations outside JSON.

Required JSON format:
[
  {{
    "EVENT_ID": "example_id",
    "M_TOTAL_CODES": ["P1", "P6"],
    "PRIMARY_PATTERN": "P1",
    "M_TOTAL": "short conceptual hidden cybersecurity pattern",
    "JUSTIFICATION": "brief explanation based on M1, M2, and M3",
    "CONFIDENCE": "High/Medium/Low",
    "MANUAL_REVIEW": false
  }}
]

Records:
{records_json}
""".strip()


# ============================================================
# Stage 1: Generate M1, M2, M3
# ============================================================

def run_stage_1_annotations(prepared_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate M1, M2, and M3 for each prepared record.
    Saves progress after every batch.
    """
    if ANNOTATION_OUTPUT_PATH.exists():
        print(f"Found existing annotation file: {ANNOTATION_OUTPUT_PATH}")
        existing_df = pd.read_excel(ANNOTATION_OUTPUT_PATH)
        processed_ids = set(existing_df["EVENT_ID"].astype(str))
    else:
        existing_df = pd.DataFrame()
        processed_ids = set()

    remaining_df = prepared_df[
        ~prepared_df["EVENT_ID"].astype(str).isin(processed_ids)
    ].copy()

    if remaining_df.empty:
        print("Stage 1 already completed.")
        return existing_df

    batches = make_batches(remaining_df, BATCH_SIZE)
    all_outputs = [] if existing_df.empty else existing_df.to_dict(orient="records")

    for batch_index, batch_df in enumerate(tqdm(batches, desc="Stage 1: M1/M2/M3")):
        records = batch_to_records(batch_df)
        prompt = build_m1_m2_m3_prompt(records)

        try:
            raw_output = call_llm(prompt)
            parsed_output = extract_json_array(raw_output)

            for item in parsed_output:
                item["BATCH_STAGE_1"] = batch_index + 1

            all_outputs.extend(parsed_output)

            temp_df = pd.DataFrame(all_outputs)
            save_excel_safely(temp_df, ANNOTATION_OUTPUT_PATH)

        except Exception as e:
            print(f"Failed Stage 1 batch {batch_index + 1}: {e}")

            error_path = OUTPUT_DIR / f"stage1_failed_batch_{batch_index + 1}.json"
            with open(error_path, "w", encoding="utf-8") as f:
                json.dump(records, f, ensure_ascii=False, indent=2)

    return pd.DataFrame(all_outputs)


# ============================================================
# Stage 2: Generate M_TOTAL
# ============================================================

def run_stage_2_mtotal(annotation_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate M_TOTAL for each annotated record.
    Saves progress after every batch.
    """
    required_cols = ["EVENT_ID", "M1", "M2", "M3"]
    missing_cols = [col for col in required_cols if col not in annotation_df.columns]

    if missing_cols:
        raise ValueError(f"Annotation dataset is missing columns: {missing_cols}")

    if MTOTAL_OUTPUT_PATH.exists():
        print(f"Found existing M_TOTAL file: {MTOTAL_OUTPUT_PATH}")
        existing_df = pd.read_excel(MTOTAL_OUTPUT_PATH)
        processed_ids = set(existing_df["EVENT_ID"].astype(str))
    else:
        existing_df = pd.DataFrame()
        processed_ids = set()

    remaining_df = annotation_df[
        ~annotation_df["EVENT_ID"].astype(str).isin(processed_ids)
    ][required_cols].copy()

    if remaining_df.empty:
        print("Stage 2 already completed.")
        return existing_df

    batches = make_batches(remaining_df, BATCH_SIZE)
    all_outputs = [] if existing_df.empty else existing_df.to_dict(orient="records")

    for batch_index, batch_df in enumerate(tqdm(batches, desc="Stage 2: M_TOTAL")):
        records = batch_to_records(batch_df)
        prompt = build_mtotal_prompt(records)

        try:
            raw_output = call_llm(prompt)
            parsed_output = extract_json_array(raw_output)

            for item in parsed_output:
                item["BATCH_STAGE_2"] = batch_index + 1

            all_outputs.extend(parsed_output)

            temp_df = pd.DataFrame(all_outputs)
            save_excel_safely(temp_df, MTOTAL_OUTPUT_PATH)

        except Exception as e:
            print(f"Failed Stage 2 batch {batch_index + 1}: {e}")

            error_path = OUTPUT_DIR / f"stage2_failed_batch_{batch_index + 1}.json"
            with open(error_path, "w", encoding="utf-8") as f:
                json.dump(records, f, ensure_ascii=False, indent=2)

    return pd.DataFrame(all_outputs)


# ============================================================
# Final Report
# ============================================================

def create_final_report(annotation_df: pd.DataFrame, mtotal_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create the final Excel file required for the paper:
    EVENT-ID, M1, M2, M3, M(TOTAL).
    """
    required_annotation_cols = ["EVENT_ID", "M1", "M2", "M3"]
    required_mtotal_cols = ["EVENT_ID", "M_TOTAL"]

    missing_annotation = [
        col for col in required_annotation_cols
        if col not in annotation_df.columns
    ]

    missing_mtotal = [
        col for col in required_mtotal_cols
        if col not in mtotal_df.columns
    ]

    if missing_annotation:
        raise ValueError(f"Annotation dataset is missing columns: {missing_annotation}")

    if missing_mtotal:
        raise ValueError(f"M_TOTAL dataset is missing columns: {missing_mtotal}")

    report_df = annotation_df[required_annotation_cols].merge(
        mtotal_df[required_mtotal_cols],
        on="EVENT_ID",
        how="left"
    )

    report_df = report_df.rename(columns={
        "EVENT_ID": "EVENT-ID",
        "M_TOTAL": "M(TOTAL)"
    })

    save_excel_safely(report_df, FINAL_REPORT_PATH)
    print(f"Final report saved to: {FINAL_REPORT_PATH}")

    return report_df


# ============================================================
# Main
# ============================================================

def main() -> None:
    print("Reading raw dataset...")
    raw_df = read_json_dataset(INPUT_JSON_PATH)
    print(f"Raw dataset records: {len(raw_df)}")

    print("Preparing Public Administration dataset...")
    prepared_df = prepare_dataset(raw_df)
    print(f"Prepared Public Administration records: {len(prepared_df)}")

    save_excel_safely(prepared_df, PREPARED_DATA_PATH)
    print(f"Prepared dataset saved to: {PREPARED_DATA_PATH}")

    print("Running Stage 1: M1/M2/M3 annotation...")
    annotation_df = run_stage_1_annotations(prepared_df)
    print(f"Annotation records generated: {len(annotation_df)}")
    print(f"Annotation dataset saved to: {ANNOTATION_OUTPUT_PATH}")

    print("Running Stage 2: M_TOTAL hidden pattern generation...")
    mtotal_df = run_stage_2_mtotal(annotation_df)
    print(f"M_TOTAL records generated: {len(mtotal_df)}")
    print(f"M_TOTAL dataset saved to: {MTOTAL_OUTPUT_PATH}")

    print("Creating final Excel report with EVENT-ID, M1, M2, M3, and M(TOTAL)...")
    final_report_df = create_final_report(annotation_df, mtotal_df)
    print(f"Final report records: {len(final_report_df)}")

    print("Done.")


if __name__ == "__main__":
    main()