import copy
import re

import numpy as np
import pytest

from applicant_eval.models import unit
from applicant_eval.schema import Role


def pytest_addoption(parser):
    parser.addoption("--run-model-tests", action="store_true", default=False)


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-model-tests"):
        for item in items:
            if "model" in item.keywords:
                item.add_marker(pytest.mark.skip(reason="pass --run-model-tests for real models"))


@pytest.fixture
def role_data():
    return {
        "id": "role",
        "title": "Test role",
        "description": "api sql production research",
        "criteria": [
            {"id": "api", "description": "Build API", "hypothesis": "api", "required": True},
            {"id": "sql", "description": "Write SQL", "hypothesis": "sql", "required": True},
            {"id": "production", "description": "Production", "hypothesis": "production"},
            {"id": "research", "description": "Research", "hypothesis": "research"},
        ],
        "profiles": [
            {
                "id": "production",
                "description": "Production route",
                "criteria": ["production"],
                "examples": ["api sql production", "api production"],
                "near_misses": ["api future"],
            },
            {
                "id": "research",
                "description": "Research route",
                "criteria": ["research"],
                "examples": ["api sql research", "api research"],
            },
        ],
    }


@pytest.fixture
def role(role_data):
    return Role.from_dict(copy.deepcopy(role_data))


class TinyModels:
    """Deterministic test double. Semantic correctness is tested separately with real models."""

    metadata = {"test_double": True}
    vocabulary = ["api", "sql", "production", "research", "future", "other"]

    def spans(self, text):
        yield 0, len(text)

    def encode(self, texts):
        rows = []
        for text in texts:
            tokens = set(re.findall(r"[a-z]+", text.lower()))
            rows.append([float(word in tokens) for word in self.vocabulary])
        return unit(rows)

    def verify(self, pairs):
        return np.array(
            [
                [0.99, 0.005, 0.005]
                if "no_" + h in p
                else [0.005, 0.99, 0.005]
                if "has_" + h in p
                else [0.005, 0.005, 0.99]
                for p, h in pairs
            ]
        )


@pytest.fixture
def models():
    return TinyModels()


@pytest.fixture(scope="session")
def real_models():
    from applicant_eval.models import LocalModels

    return LocalModels(offline=True)
