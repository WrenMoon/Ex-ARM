import pytest
from utils.ProbeIdentifier import ProbeIdentifier


def reference():
    return {"objects": ["cube@1", "sphere@0.5"], "grasps": [
        {"per_object_angles": {"cube@1": 10, "sphere@0.5": 12},
         "per_object_contact": {"cube@1": True, "sphere@0.5": True}},
        {"per_object_angles": {"cube@1": 80, "sphere@0.5": 80},
         "per_object_contact": {"cube@1": False, "sphere@0.5": True}}]}


def test_complementary_miss_and_contact():
    identifier = ProbeIdentifier(reference())
    assert identifier.identify([{"angle": 10.5, "contact": True}, {"angle": 80, "contact": False}])["object_id"] == "cube@1"
    assert identifier.identify([{"angle": 12, "contact": True}, {"angle": 80, "contact": True}])["object_id"] == "sphere@0.5"


def test_ambiguous_and_no_match():
    ref = reference()
    ref["grasps"][1]["per_object_contact"]["cube@1"] = True
    identifier = ProbeIdentifier(ref, 1)
    assert identifier.identify([{"angle": 11, "contact": True}, {"angle": 80, "contact": True}])["status"] == "ambiguous"
    assert identifier.identify([{"angle": 30, "contact": True}, {"angle": 80, "contact": True}])["status"] == "no_match"


@pytest.mark.parametrize("observations", [[], [{"angle": float("nan"), "contact": True}] * 2,
                                          [{"angle": 10, "contact": "false"}] * 2])
def test_reject_malformed_observations(observations):
    with pytest.raises(ValueError):
        ProbeIdentifier(reference()).identify(observations)


@pytest.mark.parametrize("tolerance", [-1, float("nan"), float("inf")])
def test_reject_invalid_tolerance(tolerance):
    with pytest.raises(ValueError):
        ProbeIdentifier(reference(), tolerance)
