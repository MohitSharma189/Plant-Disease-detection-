"""
Test suite for Additional Feature: Disease Understanding and Agricultural Knowledge.
Validates:
1. Database integrity: All 38 model classes present with required structured fields and extension references.
2. Service functions: loading, class listing, lookup, unindexed fallback.
3. API endpoints:
   - GET /disease-info
   - GET /disease-info/{class_name}
   - POST /predict (disease_info attachment)
   - POST /screen-plant (single_leaf and whole_plant modes)
4. Context-aware presentation rules and mandatory educational disclaimer text.
"""

import json
import os
import sys
import unittest
from pathlib import Path
from PIL import Image
import io

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from backend.app.services.disease_knowledge import (
    load_knowledge_base,
    list_supported_classes,
    get_disease_info,
    MANDATORY_DISCLAIMER,
)
from backend.app.services.screening import screen_plant_image
from backend.app.services.models import model_registry
from fastapi.testclient import TestClient
from backend.app.main import app

class TestDiseaseKnowledgeBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db_path = ROOT_DIR / "backend" / "app" / "data" / "disease_knowledge.json"
        with open(cls.db_path, "r", encoding="utf-8") as f:
            cls.data = json.load(f)
        
        # Load class mapping to verify exact 38 class match
        mapping_path = ROOT_DIR / "saved_models" / "class_mapping.json"
        with open(mapping_path, "r", encoding="utf-8") as f:
            mapping_data = json.load(f)
            cls.classes = mapping_data if isinstance(mapping_data, list) else mapping_data.get("classes", [])

    def test_database_has_exact_38_classes(self):
        self.assertEqual(len(self.data), 38, f"Expected 38 classes, got {len(self.data)}")
        for class_id in self.classes:
            self.assertIn(class_id, self.data, f"Class {class_id} missing from disease_knowledge.json")

    def test_required_fields_present_in_all_classes(self):
        required_top_level = ["crop", "condition", "is_healthy", "overview", "symptoms", "causes_and_spread", "management", "prevention", "references"]
        overview_fields = ["common_name", "scientific_name", "pathogen_type", "description"]
        symptoms_fields = ["leaf_symptoms", "lesion_appearance", "progression", "other_parts_affected"]
        causes_fields = ["causal_organism", "favorable_conditions", "transmission_modes", "susceptibility_factors"]
        management_fields = ["cultural_practices", "sanitation", "environmental_control", "professional_advice", "general_treatments"]
        prevention_fields = ["crop_hygiene", "irrigation_practices", "monitoring_and_scouting", "resistant_varieties"]

        healthy_count = 0
        disease_count = 0

        for class_name, info in self.data.items():
            for key in required_top_level:
                self.assertIn(key, info, f"Missing key '{key}' in class '{class_name}'")
            
            # Check overview
            for key in overview_fields:
                self.assertIn(key, info["overview"], f"Missing overview key '{key}' in '{class_name}'")
                self.assertTrue(len(info["overview"][key].strip()) > 0, f"Empty overview key '{key}' in '{class_name}'")
            
            # Check symptoms
            for key in symptoms_fields:
                self.assertIn(key, info["symptoms"], f"Missing symptoms key '{key}' in '{class_name}'")
                self.assertTrue(len(info["symptoms"][key].strip()) > 0, f"Empty symptoms key '{key}' in '{class_name}'")
            
            # Check causes
            for key in causes_fields:
                self.assertIn(key, info["causes_and_spread"], f"Missing causes key '{key}' in '{class_name}'")
                self.assertTrue(len(info["causes_and_spread"][key].strip()) > 0, f"Empty causes key '{key}' in '{class_name}'")
            
            # Check management
            for key in management_fields:
                self.assertIn(key, info["management"], f"Missing management key '{key}' in '{class_name}'")
                self.assertTrue(len(info["management"][key].strip()) > 0, f"Empty management key '{key}' in '{class_name}'")
            
            # Check prevention
            for key in prevention_fields:
                self.assertIn(key, info["prevention"], f"Missing prevention key '{key}' in '{class_name}'")
                self.assertTrue(len(info["prevention"][key].strip()) > 0, f"Empty prevention key '{key}' in '{class_name}'")
            
            # Check references
            refs = info["references"]
            self.assertIsInstance(refs, list, f"References must be a list in '{class_name}'")
            self.assertGreaterEqual(len(refs), 1, f"At least 1 extension reference required for '{class_name}'")
            for ref in refs:
                self.assertIn("institution", ref, f"Missing 'institution' in reference for '{class_name}'")
                self.assertIn("title", ref, f"Missing 'title' in reference for '{class_name}'")
                self.assertIn("url", ref, f"Missing 'url' in reference for '{class_name}'")
                self.assertTrue(ref["url"].startswith("http"), f"Invalid reference URL in '{class_name}': {ref['url']}")

            if info["is_healthy"]:
                healthy_count += 1
            else:
                disease_count += 1

        self.assertEqual(healthy_count, 12, f"Expected 12 healthy classes, found {healthy_count}")
        self.assertEqual(disease_count, 26, f"Expected 26 disease classes, found {disease_count}")

    def test_mandatory_disclaimer_verbatim(self):
        expected_disclaimer = (
            "This information is provided for educational purposes. Model predictions are not confirmed "
            "diagnoses. Disease symptoms can resemble those of other conditions. Consult a qualified agricultural "
            "expert or local agricultural extension service before making treatment decisions."
        )
        self.assertEqual(MANDATORY_DISCLAIMER, expected_disclaimer)


class TestDiseaseKnowledgeService(unittest.TestCase):
    def test_list_supported_classes(self):
        supported = list_supported_classes()
        self.assertEqual(len(supported), 38)
        self.assertIn("Apple___Apple_scab", supported)
        self.assertIn("Tomato___healthy", supported)

    def test_get_disease_info_disease_class(self):
        info = get_disease_info("Apple___Apple_scab")
        self.assertTrue(info["available"])
        self.assertEqual(info["title"], "Disease Information")
        self.assertFalse(info["is_healthy"])
        self.assertEqual(info["crop"], "Apple")
        self.assertIn("Venturia inaequalis", info["overview"]["scientific_name"])
        self.assertIn("disclaimer", info)
        self.assertEqual(info["disclaimer"], MANDATORY_DISCLAIMER)

    def test_get_disease_info_healthy_class(self):
        info = get_disease_info("Tomato___healthy")
        self.assertTrue(info["available"])
        self.assertEqual(info["title"], "Plant Health & Crop Care Information")
        self.assertTrue(info["is_healthy"])
        self.assertEqual(info["crop"], "Tomato")
        self.assertIn("healthy", info["overview"]["description"].lower())
        self.assertEqual(info["disclaimer"], MANDATORY_DISCLAIMER)

    def test_get_disease_info_unindexed_class_fallback(self):
        info = get_disease_info("UnknownCrop___MysteryDisease")
        self.assertFalse(info["available"])
        self.assertEqual(info["title"], "Disease Information Unavailable")
        self.assertIn("not indexed", info["message"].lower())
        self.assertEqual(info["disclaimer"], MANDATORY_DISCLAIMER)
        self.assertIn("references", info)
        self.assertGreaterEqual(len(info["references"]), 1)


class TestDiseaseKnowledgeAPIEndpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model_registry.load()
        cls.client = TestClient(app)

    def test_get_all_disease_info_classes(self):
        response = self.client.get("/disease-info")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["total_classes"], 38)
        self.assertEqual(len(data["supported_classes"]), 38)

    def test_get_single_disease_info_valid(self):
        response = self.client.get("/disease-info/Corn_(maize)___Common_rust_")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["available"])
        self.assertEqual(data["crop"], "Corn (Maize)")
        self.assertEqual(data["condition"], "Common Rust")
        self.assertEqual(data["title"], "Disease Information")

    def test_get_single_disease_info_healthy(self):
        response = self.client.get("/disease-info/Potato___healthy")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["available"])
        self.assertTrue(data["is_healthy"])
        self.assertEqual(data["title"], "Plant Health & Crop Care Information")

    def test_get_single_disease_info_fallback(self):
        response = self.client.get("/disease-info/NonExistent_Class")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertFalse(data["available"])
        self.assertIn("disclaimer", data)

    def test_predict_endpoint_attaches_disease_info(self):
        # Create a simple synthetic RGB image
        img = Image.new("RGB", (256, 256), color=(45, 120, 60))
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="JPEG")
        img_bytes.seek(0)

        response = self.client.post(
            "/predict",
            files={"file": ("test_leaf.jpg", img_bytes.getvalue(), "image/jpeg")}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("disease_info", data)
        self.assertIsNotNone(data["disease_info"])
        self.assertEqual(data["disease_info"]["disclaimer"], MANDATORY_DISCLAIMER)

    def test_screen_plant_single_leaf_attaches_disease_info(self):
        img = Image.new("RGB", (256, 256), color=(45, 120, 60))
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="JPEG")
        img_bytes.seek(0)

        response = self.client.post(
            "/screen-plant",
            data={"mode": "single_leaf"},
            files={"file": ("test_leaf.jpg", img_bytes.getvalue(), "image/jpeg")}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("screening", data)
        self.assertIn("disease_info", data["screening"])
        self.assertIsNotNone(data["screening"]["disease_info"])
        self.assertEqual(data["screening"]["disease_info"]["disclaimer"], MANDATORY_DISCLAIMER)

    def test_screen_plant_whole_plant_mode(self):
        img = Image.new("RGB", (512, 512), color=(50, 110, 50))
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="JPEG")
        img_bytes.seek(0)

        response = self.client.post(
            "/screen-plant",
            data={"mode": "whole_plant"},
            files={"file": ("test_plant.jpg", img_bytes.getvalue(), "image/jpeg")}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("screening", data)
        # Either candidate regions were identified with attached disease_info,
        # or no candidates were identified and disease_info is None (triggering the no-candidates panel).
        self.assertIn("disease_info", data["screening"])


if __name__ == "__main__":
    unittest.main()
