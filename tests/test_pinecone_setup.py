from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import setup_pinecone_index as setup  # noqa: E402


class ReadCapacityOnDemandResponse(SimpleNamespace):
    pass


def compatible_index(**overrides):
    data = {
        "schema": {"fields": {"chunk_vector": {"type": "dense_vector", "dimension": 1536, "metric": "cosine"}}},
        "deployment": {"deployment_type": "managed", "cloud": "aws", "region": "us-east-1"},
        "read_capacity": ReadCapacityOnDemandResponse(),
        "status": {"ready": True, "state": "Ready"},
    }
    data.update(overrides)
    return SimpleNamespace(**data)


class FakeIndexes:
    def __init__(self, names=(), description=None, error=None, descriptions=None, create_response=None):
        self.names = names
        self.description = description or compatible_index()
        self.error = error
        self.descriptions = list(descriptions or [])
        self.create_response = create_response
        self.created = []
        self.described = []

    def list(self):
        if self.error:
            raise self.error
        return [SimpleNamespace(name=name) for name in self.names]

    def describe(self, name):
        self.described.append(name)
        if self.error:
            raise self.error
        if self.descriptions:
            return self.descriptions.pop(0)
        return self.description

    def create(self, **kwargs):
        self.created.append(kwargs)
        return self.create_response


class FakeClient:
    def __init__(self, indexes):
        self.indexes = indexes


def test_missing_api_key_is_clear_and_does_not_construct_client(monkeypatch):
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)
    with pytest.raises(setup.SetupError, match="PINECONE_API_KEY"):
        setup.run(create=False)


def test_check_reports_missing_index_without_creating_it():
    indexes = FakeIndexes(names=["unrelated-one", "unrelated-two"])
    ok, message = setup.run(create=False, api_key="test-key", client=FakeClient(indexes))
    assert not ok
    assert "--create" in message
    assert indexes.created == []
    assert indexes.described == []


def test_existing_compatible_index_is_validated_without_modification():
    indexes = FakeIndexes(names=[setup.INDEX_NAME, "unrelated-index"])
    ok, message = setup.run(create=True, api_key="test-key", client=FakeClient(indexes))
    assert ok and "compatible" in message
    assert indexes.created == []
    assert indexes.described == [setup.INDEX_NAME]


@pytest.mark.parametrize("field, bad_value", [
    ("deployment", {"deployment_type": "pod", "cloud": "aws", "region": "us-east-1"}),
    ("schema", {"fields": {"chunk_vector": {"type": "dense_vector", "dimension": 768, "metric": "cosine"}}}),
    ("read_capacity", SimpleNamespace(mode="Dedicated")),
])
def test_existing_incompatible_index_fails_without_modifying(field, bad_value):
    indexes = FakeIndexes(names=[setup.INDEX_NAME], description=compatible_index(**{field: bad_value}))
    with pytest.raises(setup.SetupError, match="incompatible settings"):
        setup.run(create=False, api_key="test-key", client=FakeClient(indexes))
    assert indexes.created == []


def test_creation_is_explicit_idempotent_and_uses_requested_schema():
    indexes = FakeIndexes()
    ok, message = setup.run(create=True, api_key="test-key", client=FakeClient(indexes))
    assert ok and "created and validated" in message
    assert len(indexes.created) == 1
    request = indexes.created[0]
    assert request["name"] == "career-market-knowledge"
    assert request["schema"]["fields"]["chunk_vector"] == {
        "type": "dense_vector", "dimension": 1536, "metric": "cosine",
    }
    assert request["deployment"] == {"deployment_type": "managed", "cloud": "aws", "region": "us-east-1"}
    assert "read_capacity" not in request  # Pinecone default is OnDemand.
    assert indexes.described == [setup.INDEX_NAME]  # Description fetched after create.


def test_pinecone_v10_to_dict_response_validates_exact_reported_configuration():
    class SDKResponse:
        def __init__(self, payload):
            self.payload = payload

        def to_dict(self):
            return self.payload

    response = SDKResponse({
        "name": setup.INDEX_NAME,
        "schema": {"fields": {"chunk_vector": {"type": "dense_vector", "dimension": 1536, "metric": "cosine"}}},
        "deployment": {"deployment_type": "managed", "cloud": "aws", "region": "us-east-1"},
        "read_capacity": {"mode": "OnDemand"},
        "status": {"ready": True, "state": "Ready"},
    })
    assert setup.validate_index(response) == []
    indexes = FakeIndexes(names=[setup.INDEX_NAME], description=response)
    ok, message = setup.run(create=False, api_key="test-key", client=FakeClient(indexes))
    assert ok and "compatible" in message
    assert indexes.created == []


def test_create_waits_through_provisioning_and_validates_final_description(monkeypatch):
    provisioning = compatible_index(status={"ready": False, "state": "Initializing"})
    ready = compatible_index()
    indexes = FakeIndexes(descriptions=[provisioning, ready])
    monkeypatch.setattr(setup.time, "sleep", lambda _seconds: None)
    ok, message = setup.run(create=True, api_key="test-key", client=FakeClient(indexes))
    assert ok and "created and validated" in message
    assert len(indexes.created) == 1
    assert len(indexes.described) == 2


def test_create_does_not_recreate_existing_index_even_if_valid():
    indexes = FakeIndexes(names=[setup.INDEX_NAME], description=compatible_index())
    ok, message = setup.run(create=True, api_key="test-key", client=FakeClient(indexes))
    assert ok and "compatible" in message
    assert indexes.created == []


def test_pinecone_api_failure_is_reported_without_exposing_details():
    indexes = FakeIndexes(error=RuntimeError("service error"))
    with pytest.raises(setup.SetupError, match="API operation failed") as error:
        setup.run(create=False, api_key="test-key", client=FakeClient(indexes))
    assert "service error" not in str(error.value)
