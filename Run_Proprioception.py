"""Grasp an object with the real hand and recognize its class and size."""

from datetime import datetime, timezone
import json
from pathlib import Path

from Grasp import grasp
from utils.Constants import Connection, Proprioception
from utils.ProprioceptionModel import ProprioceptionModel


def recognize(leap_hand, model):
    final_angles = grasp(leap_hand)
    result = model.predict(final_angles)

    if result["class_name"] == "unknown":
        print(f"Object: unknown ({result['reason']})")
    else:
        print(f"Object: {result['class_name']}")
        print(f"{result['size_name']}: {result['size_m'] * 1000:.1f} mm")
    print(f"Class confidence: {result['confidence']:.1%}")

    output = dict(time_utc=datetime.now(timezone.utc).isoformat(),
                  final_angles_deg=final_angles, prediction=result)
    path = Path(Proprioception.recognition_result_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(f"Result saved to {path}")
    print(result)
    return output


def main():
    if Proprioception.hand_mode != "real":
        raise ValueError("The final recognition program requires the real hand")
    if not Path(Proprioception.model_path).exists():
        raise FileNotFoundError("Train the proprioception model before running the real hand")
    model = ProprioceptionModel.load()  # Check the model before commanding the hand.
    from utils.ExARM import ExArm

    leap_hand = ExArm(
        mode=Proprioception.hand_mode,
        ids=Connection.ids,
        port=Connection.Port,
        baudrate=Connection.baudrate,
        offsets=Connection.offsets,
        model_path=Connection.model_path,
    )
    try:
        recognize(leap_hand, model)
    finally:
        leap_hand.close()


if __name__ == "__main__":
    main()
