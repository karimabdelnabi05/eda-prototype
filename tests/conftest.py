"""
Shared test fixtures for the EDA test suite.
"""

import json
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir():
    """Path to the test fixtures directory."""
    return FIXTURES_DIR


@pytest.fixture
def sample_financial_text():
    """Load the sample financial report text."""
    path = FIXTURES_DIR / "sample_financial_report.txt"
    return path.read_text(encoding="utf-8")


@pytest.fixture
def financial_ground_truth():
    """Load the financial report ground truth."""
    path = FIXTURES_DIR / "ground_truth_financial.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def sample_compiled_class():
    """A pre-built compiled class for testing runtime components without LLM calls."""
    return '''
class TestFinancialReport:
    def __init__(self):
        self.revenue = {"total": 128500000, "cloud": 72300000, "enterprise": 38100000}
        self.capex = {
            "Q1": {"amount": 32000000, "driver": "R&D"},
            "Q2": {"amount": 41000000, "driver": "Acquisitions"},
            "Q3": {"amount": 45000000, "driver": "Cloud Infrastructure"},
        }
        self.travel_policy = {
            "Executive": {"max_flight": 5000, "class": "Business"},
            "Senior Manager": {"max_flight": 2500, "class": "Premium Economy"},
            "Standard": {"max_flight": 1200, "class": "Economy"},
        }
        self.employees = {"total": 2847, "engineering": 1240, "sales": 685}

    def get_total_revenue(self) -> int:
        return self.revenue["total"]

    def get_capex(self, quarter: str) -> dict:
        if quarter not in self.capex:
            return {"error": f"Quarter {quarter} not found"}
        return self.capex[quarter]

    def check_travel_compliance(self, role: str, flight_class: str = "", cost: float = 0) -> bool:
        policy = self.travel_policy.get(role)
        if not policy:
            return False
        return cost <= policy["max_flight"]

    def get_headcount(self, department: str = "") -> int:
        if department:
            return self.employees.get(department, 0)
        return self.employees["total"]

    def get_summary(self) -> dict:
        return {
            "title": "Acme Corp Q3 2024 Financial Report",
            "total_revenue": 128500000,
            "net_income": 18200000,
            "total_employees": 2847,
        }

    def list_available_methods(self) -> list:
        return [
            "get_total_revenue() - Returns total Q3 revenue",
            "get_capex(quarter) - Returns capex data for a quarter",
            "check_travel_compliance(role, flight_class, cost) - Check travel policy",
            "get_headcount(department) - Returns employee count",
            "get_summary() - Returns document summary",
        ]
'''
