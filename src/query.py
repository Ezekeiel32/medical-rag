import re
from typing import Optional, Tuple


def parse_patient_query(text: str) -> Tuple[Optional[str], Optional[str]]:
    pattern = re.compile(r"patient\s*id\s*[:=\-]?\s*([A-Za-z0-9\-]+)", re.IGNORECASE)
    match = pattern.search(text)
    patient_id = match.group(1) if match else None

    lowered = text.lower()
    if any(k in lowered for k in ("medication", "medications", "drugs")):
        return patient_id, "medications"
    if any(k in lowered for k in ("diagnosis", "diagnoses", "conditions")):
        return patient_id, "diagnoses"
    if any(k in lowered for k in ("procedure", "procedures")):
        return patient_id, "procedures"
    return patient_id, None


