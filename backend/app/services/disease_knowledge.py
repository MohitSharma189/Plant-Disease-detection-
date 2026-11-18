import json
from functools import lru_cache
from pathlib import Path
from typing import Any

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "disease_knowledge.json"

MANDATORY_DISCLAIMER = (
    "This information is provided for educational purposes. Model predictions are not confirmed "
    "diagnoses. Disease symptoms can resemble those of other conditions. Consult a qualified agricultural "
    "expert or local agricultural extension service before making treatment decisions."
)


@lru_cache(maxsize=1)
def load_knowledge_base() -> dict[str, dict[str, Any]]:
    """Load the disease knowledge base from the local JSON data repository."""
    if not DATA_PATH.exists():
        return {}
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def list_supported_classes() -> list[str]:
    """Return all 38 exact class identifiers supported in the knowledge base."""
    return list(load_knowledge_base().keys())


def get_disease_info(class_name: str | None) -> dict[str, Any]:
    """
    Retrieve structured agricultural knowledge for a specific model class identifier.
    Gracefully handles unknown or missing classes by returning an informative fallback.
    """
    if not class_name:
        class_name = "Unknown"

    kb = load_knowledge_base()

    if class_name in kb:
        entry = dict(kb[class_name])
        entry["available"] = True
        entry["disclaimer"] = MANDATORY_DISCLAIMER
        entry["title"] = "Plant Health & Crop Care Information" if entry.get("is_healthy") else "Disease Information"
        return entry

    # Fallback for unindexed or invalid class identifiers
    raw_crop, _, raw_cond = class_name.partition("___")
    crop_display = raw_crop.replace("_", " ").replace(",", "").strip() or "Unspecified Crop"
    cond_display = raw_cond.replace("_", " ").strip() or "Unindexed Condition"
    is_healthy = "healthy" in class_name.lower()

    return {
        "class_id": class_name,
        "crop": crop_display,
        "condition": cond_display,
        "is_healthy": is_healthy,
        "available": False,
        "title": "Disease Information Unavailable",
        "message": f"Disease profile for '{class_name}' is not indexed in the local knowledge base.",
        "disclaimer": MANDATORY_DISCLAIMER,
        "overview": {
            "common_name": cond_display,
            "scientific_name": "Not specified in local database",
            "pathogen_type": "Unclassified / Pending expert verification",
            "description": (
                f"Detailed reference information for {crop_display} ({cond_display}) is currently being "
                "compiled from verified agricultural extension sources. Please consult your local university "
                "agricultural extension service or a certified crop advisor for diagnostic and management guidance."
            ),
        },
        "symptoms": {
            "leaf_symptoms": "Specific visual symptom criteria are not indexed for this class identifier.",
            "lesion_appearance": "Lesion characteristics require confirmation by a plant pathology laboratory.",
            "progression": "Progression rates depend on ambient microclimate and pathogen identity.",
            "other_parts_affected": "Field inspection of roots, stems, and fruits is advised.",
        },
        "causes_and_spread": {
            "causal_organism": "Causal organism under extension review.",
            "favorable_conditions": "Environmental thresholds not indexed.",
            "transmission_modes": "Modes of transmission not indexed.",
            "susceptibility_factors": "Cultivar susceptibility data not indexed.",
        },
        "management": {
            "cultural_practices": "Follow general good agronomic practices including balanced fertilization and crop rotation.",
            "sanitation": "Isolate or remove severely blighted plant material to reduce local inoculum.",
            "environmental_control": "Ensure adequate airflow and avoid prolonged foliar wetness.",
            "professional_advice": "Consult your local county extension educator or diagnostic clinic before applying treatments.",
            "general_treatments": "No chemical or biological interventions are recommended without verified laboratory identification.",
        },
        "prevention": {
            "crop_hygiene": "Maintain clean field sanitation and disinfect pruning shears.",
            "irrigation_practices": "Avoid overhead watering; irrigate at the soil level using drip irrigation.",
            "monitoring_and_scouting": "Scout crops regularly for developing foliar symptoms.",
            "resistant_varieties": "Check local extension cultivar trial reports for regionally adapted varieties.",
        },
        "references": [
            {
                "institution": "USDA National Institute of Food and Agriculture (NIFA)",
                "title": "Cooperative Extension System Directory",
                "url": "https://www.nifa.usda.gov/land-grant-colleges-and-universities-partner-website-directory",
            }
        ],
    }
