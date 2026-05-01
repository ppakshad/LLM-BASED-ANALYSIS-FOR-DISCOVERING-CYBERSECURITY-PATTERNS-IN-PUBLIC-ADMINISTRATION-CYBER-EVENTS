# LLM-Based Cybersecurity Event Analysis and Hidden Pattern Discovery

This repository contains a Python-based research prototype for analyzing cybersecurity event records using Large Language Models (LLMs). The project focuses on Public Administration cyber events and generates semantic cybersecurity indicators to identify hidden security patterns that may not be directly visible in raw incident records.

## Project Overview

Cybersecurity event datasets often contain structured fields such as event date, organization, motive, event subtype, magnitude, scope, description, victim country, and actor country. However, these raw fields may not fully explain the deeper cybersecurity meaning of an incident.

This project uses an LLM-based analysis pipeline to transform cyber event records into higher-level semantic indicators:

- **M1: Incident Security Semantics**
- **M2: Governance Function Criticality**
- **M3: Cyber Intelligence Gap and Inference**
- **M(TOTAL): Hidden Cybersecurity Pattern**

The goal is to support deeper cybersecurity interpretation, especially for Public Administration incidents, by identifying patterns related to operational disruption, citizen-service dependency, public safety exposure, sensitive governance exposure, data exposure, intelligence gaps, and geopolitical signals.

## Repository Title

**LLM-Based Cybersecurity Event Analysis and Hidden Pattern Discovery**

## Description

A data-centric cybersecurity analysis project that uses Large Language Models to examine Public Administration cyber event records, generate semantic security indicators, and discover hidden cybersecurity patterns beyond basic incident-level reporting.

## Main Features

- Reads raw CyberEvents data from a JSON file
- Filters records related to the Public Administration sector
- Selects key cybersecurity-relevant fields
- Cleans and normalizes missing or unclear values
- Assigns unique event identifiers
- Processes up to 3000 cyber event records
- Uses OpenAI LLM API for semantic annotation
- Generates M1, M2, and M3 indicators for each event
- Generates M(TOTAL) as a higher-level hidden cybersecurity pattern
- Saves intermediate and final results as Excel files
- Supports batch processing with retry logic
- Saves progress after each batch to avoid losing completed work

## Methodology

The project follows a two-stage LLM-assisted cybersecurity analysis workflow.

### Stage 1: Semantic Indicator Generation

For each Public Administration cyber event, the model generates three semantic indicators:

### M1: Incident Security Semantics

M1 explains the cybersecurity meaning of the incident itself. It considers fields such as:

- Description
- Scope
- Magnitude
- Event subtype
- Motive

Examples of M1 meanings include:

- Ransomware
- Data exposure
- Unauthorized access
- Service disruption
- Vulnerability exploitation
- Sabotage
- Espionage
- Operational degradation

### M2: Governance Function Criticality

M2 explains the affected public-sector or governance function. It considers fields such as:

- Organization
- Country
- Description
- Scope

Examples of M2 meanings include:

- Citizen services
- Public safety
- Law enforcement
- Taxation
- Legal administration
- Emergency response
- Public records
- Municipal operations
- Government communication
- Digital public platforms

### M3: Cyber Intelligence Gap and Inference

M3 explains whether important intelligence fields are missing, unclear, null, or undetermined. It also evaluates whether the available information still supports useful cybersecurity inference.

M3 considers fields such as:

- Motive
- Event subtype
- Magnitude
- Scope
- Description
- Organization
- Country
- Actor country
- Event date

### Stage 2: Hidden Pattern Generation

After M1, M2, and M3 are generated, the project creates **M(TOTAL)**.

M(TOTAL) represents the higher-level hidden cybersecurity pattern derived from the integration of M1, M2, and M3.

The project uses the following pattern taxonomy:

| Code | Hidden Cybersecurity Pattern |
|---|---|
| P1 | Operational Disruption of Public Functions |
| P2 | Public Safety or Law Enforcement Exposure |
| P3 | Citizen-Service Dependency Disruption |
| P4 | Sensitive Governance Exposure |
| P5 | Data Exposure in Public-Sector Context |
| P6 | High-Value Intelligence Gap |
| P7 | Cross-Border or Geopolitical Cyber Signal |
| P8 | Routine or Low-Semantic Administrative Incident |

## Input Dataset

The script expects a JSON file in the project directory.

Default input file name:

```text
cyber_events_2026-05-01.json ==> https://cybereventsdatabase.org/dashboard
