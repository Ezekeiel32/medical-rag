import json
import os
from typing import Dict, List, Any, Optional


def load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def find_patient_id_from_reference(ref: str) -> Optional[str]:
    # Expected formats: "Patient/<id>", "urn:uuid:<id>", or plain id
    if not isinstance(ref, str):
        return None
    if ref.startswith("urn:uuid:"):
        return ref.split(":")[-1]
    if "/" in ref:
        parts = ref.split("/")
        return parts[-1]
    return ref


def index_fhir_directory(root_dir: str) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    index: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if not filename.endswith(".json"):
                continue
            file_path = os.path.join(dirpath, filename)
            try:
                data = load_json(file_path)
            except Exception:
                continue

            resource_type = data.get("resourceType")
            if not resource_type:
                continue

            if resource_type == "Bundle":
                entries = data.get("entry", [])
                for entry in entries:
                    resource = entry.get("resource", {})
                    _add_resource_to_index(index, resource)
            else:
                _add_resource_to_index(index, data)
    return index


def _add_resource_to_index(index: Dict[str, Dict[str, List[Dict[str, Any]]]], resource: Dict[str, Any]) -> None:
    resource_type = resource.get("resourceType")
    if not resource_type:
        return

    if resource_type == "Patient":
        patient_id = resource.get("id")
        if not patient_id:
            return
        index.setdefault(patient_id, {})
        index[patient_id].setdefault("Patient", []).append(resource)
        return

    # MedicationRequest: subject.reference -> Patient/{id}
    if resource_type in ("MedicationRequest", "MedicationStatement"):
        subject = resource.get("subject", {})
        ref = subject.get("reference")
        pid = find_patient_id_from_reference(ref)
        if not pid:
            return
        index.setdefault(pid, {})
        index[pid].setdefault(resource_type, []).append(resource)
        return

    # Other clinical resources may also reference patient via subject or patient
    ref_fields = [
        ("subject", "reference"),
        ("patient", "reference"),
    ]
    for top_key, key in ref_fields:
        obj = resource.get(top_key)
        if isinstance(obj, dict) and key in obj:
            pid = find_patient_id_from_reference(obj[key])
            if pid:
                index.setdefault(pid, {})
                index[pid].setdefault(resource_type, []).append(resource)
                return


def extract_medications_for_patient(index: Dict[str, Dict[str, List[Dict[str, Any]]]], patient_id: str) -> List[Dict[str, Any]]:
    patient_bucket = index.get(patient_id, {})
    meds: List[Dict[str, Any]] = []

    for rtype in ("MedicationRequest", "MedicationStatement"):
        for res in patient_bucket.get(rtype, []) or []:
            med_name = None
            codeable = res.get("medicationCodeableConcept")
            if isinstance(codeable, dict):
                med_name = codeable.get("text")
                if not med_name:
                    coding = codeable.get("coding") or []
                    if coding and isinstance(coding, list):
                        med_name = coding[0].get("display") or coding[0].get("code")

            dosage_texts: List[str] = []
            for di in res.get("dosageInstruction", []) or []:
                if isinstance(di, dict):
                    if di.get("text"):
                        dosage_texts.append(di["text"]) 

            meds.append({
                "resourceType": res.get("resourceType"),
                "status": res.get("status"),
                "intent": res.get("intent"),
                "medication": med_name,
                "dosage": "; ".join(dosage_texts) if dosage_texts else None,
                "authoredOn": res.get("authoredOn"),
            })

    # Deduplicate entries by (medication, dosage, authoredOn)
    unique = []
    seen = set()
    for m in meds:
        key = (m.get("medication"), m.get("dosage"), m.get("authoredOn"))
        if key in seen:
            continue
        seen.add(key)
        unique.append(m)
    return unique


