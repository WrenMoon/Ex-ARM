"""A small neural network for object class and size from final joint angles."""

import csv
import json
from pathlib import Path

import numpy as np

from utils.Constants import Proprioception


def model_settings():
    grip = Proprioception.grip
    return json.dumps(dict(
        normalization="ignore_fixed_joints",
        start_angles=grip["start_angles"], max_angles=grip["max_angles"],
        step_sizes=grip["step_sizes"], objects=Proprioception.objects,
        mount_translation_m=Proprioception.mount_translation_m,
        mount_rotation_rpy_deg=Proprioception.mount_rotation_rpy_deg,
        sim_joint_offsets_deg=Proprioception.sim_joint_offsets_deg,
        scale_start_percent=Proprioception.scale_start_percent,
        scale_stop_percent=Proprioception.scale_stop_percent,
        scale_step_percent=Proprioception.scale_step_percent,
        measurement_noise_deg=Proprioception.measurement_noise_deg,
        noisy_samples_per_grasp=Proprioception.noisy_samples_per_grasp,
        validation_stride=Proprioception.validation_stride,
        random_seed=Proprioception.random_seed,
        network_hidden_size=Proprioception.network_hidden_size,
        network_epochs=Proprioception.network_epochs,
        network_learning_rate=Proprioception.network_learning_rate,
        size_loss_weight=Proprioception.size_loss_weight,
        unknown_distance_multiplier=Proprioception.unknown_distance_multiplier,
    ), sort_keys=True)


def normalize_angles(angles):
    angles = np.asarray(angles, dtype=float)
    if angles.shape[-1] != 16 or not np.isfinite(angles).all():
        raise ValueError("Expected 16 finite final joint angles")
    start = np.asarray(Proprioception.grip["start_angles"], dtype=float)
    span = np.asarray(Proprioception.grip["max_angles"], dtype=float) - start
    # Fixed joints contain no simulated object information.
    span[span == 0] = np.inf
    return (angles - start) / span


class ProprioceptionModel:
    def __init__(self):
        classes = Proprioception.objects
        self.class_names = [entry["class_name"] for entry in classes]
        self.size_names = [entry["size_name"] for entry in classes]
        self.reference_sizes = np.asarray([entry["reference_size_m"] for entry in classes])
        rng = np.random.default_rng(Proprioception.random_seed)
        hidden = Proprioception.network_hidden_size
        self.weights1 = rng.normal(0, np.sqrt(2 / 16), (16, hidden))
        self.bias1 = np.zeros(hidden)
        self.weights2 = rng.normal(0, np.sqrt(2 / hidden), (hidden, len(classes) + 1))
        self.bias2 = np.zeros(len(classes) + 1)
        self.bias2[-1] = 1  # The middle of the scale sweep is 100%.
        self.reference_angles = np.zeros((0, 16))
        self.reference_labels = np.zeros(0, dtype=int)
        self.unknown_distance = 0

    def outputs(self, angles):
        x = normalize_angles(angles)
        hidden = np.tanh(np.einsum("...i,ij->...j", x, self.weights1,
                                   optimize=False) + self.bias1)
        # One output per class, followed by the object's scale factor.
        output = np.einsum("...i,ij->...j", hidden, self.weights2,
                           optimize=False) + self.bias2
        logits = output[..., :-1]
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum(axis=-1, keepdims=True)
        return probabilities, output[..., -1], hidden

    def train(self, angles, class_indices, scale_factors):
        x = normalize_angles(angles)
        labels = np.asarray(class_indices, dtype=int)
        factors = np.asarray(scale_factors, dtype=float)
        if x.ndim != 2 or len(x) != len(labels) or len(x) != len(factors):
            raise ValueError("Training arrays have different lengths")
        parameters = [self.weights1, self.bias1, self.weights2, self.bias2]
        momentum = [np.zeros_like(value) for value in parameters]
        variance = [np.zeros_like(value) for value in parameters]
        learning_rate = Proprioception.network_learning_rate

        for epoch in range(1, Proprioception.network_epochs + 1):
            # Combine classification error with size-prediction error.
            probabilities, predicted_factors, hidden = self.outputs(angles)
            difference = probabilities.copy()
            difference[np.arange(len(labels)), labels] -= 1
            difference /= len(labels)
            size_gradient = (2 * Proprioception.size_loss_weight
                             * (predicted_factors - factors) / len(labels))
            output_gradient = np.column_stack((difference, size_gradient))
            gradient2 = np.einsum("ni,nj->ij", hidden, output_gradient, optimize=False)
            bias_gradient2 = output_gradient.sum(axis=0)
            hidden_gradient = (np.einsum("no,ho->nh", output_gradient, self.weights2,
                                         optimize=False) * (1 - hidden ** 2))
            gradient1 = np.einsum("ni,nj->ij", x, hidden_gradient, optimize=False)
            bias_gradient1 = hidden_gradient.sum(axis=0)

            # Adam adjusts the two layers using the gradients above.
            for index, gradient in enumerate((gradient1, bias_gradient1,
                                              gradient2, bias_gradient2)):
                momentum[index] = 0.9 * momentum[index] + 0.1 * gradient
                variance[index] = 0.999 * variance[index] + 0.001 * gradient ** 2
                corrected_momentum = momentum[index] / (1 - 0.9 ** epoch)
                corrected_variance = variance[index] / (1 - 0.999 ** epoch)
                parameters[index] -= learning_rate * corrected_momentum / (
                    np.sqrt(corrected_variance) + 1e-8)

    def predict(self, angles):
        angles = np.asarray(angles, dtype=float)
        if angles.shape != (16,):
            raise ValueError("Expected 16 final joint angles")
        if len(self.reference_angles) == 0:
            raise ValueError("Model has no simulated reference grasps")
        if len(self.reference_labels) != len(self.reference_angles):
            raise ValueError("Model reference classes are missing")
        reference_distances = np.linalg.norm(
            normalize_angles(self.reference_angles) - normalize_angles(angles), axis=1)
        closest = int(np.argmin(reference_distances))
        distance = float(reference_distances[closest])
        if distance > self.unknown_distance:
            return dict(class_name="unknown", size_name=None, size_m=None,
                        confidence=0.0, distance=distance, reason="far from known grasps")
        probabilities, factor, _ = self.outputs(angles)
        class_index = int(np.argmax(probabilities))
        confidence = float(probabilities[class_index])
        if confidence < Proprioception.unknown_probability_min:
            return dict(class_name="unknown", size_name=None, size_m=None,
                        confidence=confidence, distance=distance, reason="low confidence")
        if class_index != self.reference_labels[closest]:
            return dict(class_name="unknown", size_name=None, size_m=None,
                        confidence=confidence, distance=distance,
                        reason="neural network and nearest grasp disagree")
        class_distances = [
            float(np.min(np.linalg.norm(
                self.reference_angles[self.reference_labels == index] - angles,
                axis=1))) for index in range(len(self.class_names))
        ]
        nearest_classes = sorted(class_distances)
        if nearest_classes[1] - nearest_classes[0] < Proprioception.unknown_class_margin_deg:
            return dict(class_name="unknown", size_name=None, size_m=None,
                        confidence=confidence, distance=distance,
                        reason="classes have similar grasp angles")
        factor = float(np.clip(factor, Proprioception.scale_stop_percent / 100,
                               Proprioception.scale_start_percent / 100))
        return dict(class_name=self.class_names[class_index],
                    size_name=self.size_names[class_index],
                    size_m=float(factor * self.reference_sizes[class_index]),
                    confidence=confidence, distance=distance, reason=None)

    def save(self, path=None):
        path = Path(path or Proprioception.model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, weights1=self.weights1, bias1=self.bias1,
                 weights2=self.weights2, bias2=self.bias2,
                 class_names=np.asarray(self.class_names),
                 size_names=np.asarray(self.size_names),
                 reference_sizes=self.reference_sizes,
                 reference_angles=self.reference_angles,
                 reference_labels=self.reference_labels,
                 unknown_distance=self.unknown_distance,
                 settings=model_settings())

    @classmethod
    def load(cls, path=None):
        with np.load(path or Proprioception.model_path, allow_pickle=False) as saved:
            if str(saved["settings"]) != model_settings():
                raise ValueError("Grasp or object settings changed; retrain the model")
            model = cls()
            model.weights1 = saved["weights1"].copy()
            model.bias1 = saved["bias1"].copy()
            model.weights2 = saved["weights2"].copy()
            model.bias2 = saved["bias2"].copy()
            model.class_names = saved["class_names"].tolist()
            model.size_names = saved["size_names"].tolist()
            model.reference_sizes = saved["reference_sizes"].copy()
            model.reference_angles = saved["reference_angles"].copy()
            model.reference_labels = saved["reference_labels"].copy()
            model.unknown_distance = float(saved["unknown_distance"])
        return model


def load_simulations(path=None):
    with open(path or Proprioception.simulation_path, newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError("No simulated grasps found")
    angles = np.asarray([[float(row[f"joint_{joint}"]) for joint in range(16)]
                         for row in rows])
    class_names = [entry["class_name"] for entry in Proprioception.objects]
    labels = np.asarray([class_names.index(row["class"]) for row in rows])
    factors = np.asarray([float(row["scale_percent"]) / 100 for row in rows])
    return angles, labels, factors


def add_measurement_noise(angles, labels, factors, seed):
    rng = np.random.default_rng(seed)
    copies = Proprioception.noisy_samples_per_grasp
    repeated_angles = np.repeat(angles, copies + 1, axis=0)
    noise = rng.normal(0, Proprioception.measurement_noise_deg,
                       repeated_angles.shape)
    noise[::copies + 1] = 0
    return (repeated_angles + noise,
            np.repeat(labels, copies + 1),
            np.repeat(factors, copies + 1))


def train_model():
    angles, labels, factors = load_simulations()
    stride = Proprioception.validation_stride
    scale_indices = np.rint((Proprioception.scale_start_percent - factors * 100)
                             / abs(Proprioception.scale_step_percent)).astype(int)
    validation = scale_indices % stride == 0
    if not validation.any() or validation.all():
        raise ValueError("Scale sweep needs both training and validation sizes")
    training = ~validation

    noisy_angles, noisy_labels, noisy_factors = add_measurement_noise(
        angles[training], labels[training], factors[training],
        Proprioception.random_seed)
    validation_model = ProprioceptionModel()
    validation_model.train(noisy_angles, noisy_labels, noisy_factors)
    probabilities, predicted_factors, _ = validation_model.outputs(angles[validation])
    predicted_classes = np.argmax(probabilities, axis=1)
    accuracy = float(np.mean(predicted_classes == labels[validation]))
    size_errors = np.abs(predicted_factors - factors[validation])
    reference_sizes = validation_model.reference_sizes[labels[validation]]
    size_mae_mm = float(np.mean(size_errors * reference_sizes) * 1000)

    known_distances = np.min(np.linalg.norm(
        normalize_angles(angles[validation])[:, None, :]
        - normalize_angles(angles[training])[None, :, :], axis=2), axis=1)
    unknown_distance = (float(np.max(known_distances))
                        * Proprioception.unknown_distance_multiplier)
    unknown_distance = max(unknown_distance, 1e-6)
    validation_model.reference_angles = angles[training]
    validation_model.reference_labels = labels[training]
    validation_model.unknown_distance = unknown_distance
    validation_results = [validation_model.predict(row)
                          for row in angles[validation]]
    accepted = [result["class_name"] for result in validation_results]
    accepted_count = sum(name != "unknown" for name in accepted)
    accepted_correct = sum(name == validation_model.class_names[label]
                           for name, label in zip(accepted, labels[validation]))

    validation_path = Path(Proprioception.validation_path)
    validation_path.parent.mkdir(parents=True, exist_ok=True)
    with validation_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["class", "scale_percent", "size_m", "network_class",
                         "network_confidence", "result_class", "result_size_m",
                         "unknown_reason"] + [f"joint_{joint}" for joint in range(16)])
        for row, label, factor, probabilities_row, result in zip(
                angles[validation], labels[validation], factors[validation],
                probabilities, validation_results):
            writer.writerow([validation_model.class_names[label], factor * 100,
                             factor * validation_model.reference_sizes[label],
                             validation_model.class_names[int(np.argmax(probabilities_row))],
                             float(np.max(probabilities_row)), result["class_name"],
                             result["size_m"], result["reason"], *row])

    all_angles, all_labels, all_factors = add_measurement_noise(
        angles, labels, factors, Proprioception.random_seed)
    samples_path = Path(Proprioception.training_samples_path)
    samples_path.parent.mkdir(parents=True, exist_ok=True)
    with samples_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["stage", "class", "size_name", "size_m", "scale_percent"]
                        + [f"joint_{joint}" for joint in range(16)])
        for stage, sample_angles, sample_labels, sample_factors in (
                ("validation_model", noisy_angles, noisy_labels, noisy_factors),
                ("final_model", all_angles, all_labels, all_factors)):
            for row, label, factor in zip(sample_angles, sample_labels, sample_factors):
                object_info = Proprioception.objects[label]
                writer.writerow([stage, object_info["class_name"],
                                 object_info["size_name"],
                                 factor * object_info["reference_size_m"],
                                 factor * 100, *row])
    model = ProprioceptionModel()
    model.train(all_angles, all_labels, all_factors)
    model.reference_angles = angles
    model.reference_labels = labels
    model.unknown_distance = unknown_distance
    model.save()
    report = dict(validation_accuracy=accuracy, validation_size_mae_mm=size_mae_mm,
                  accepted_count=accepted_count, accepted_correct=accepted_correct,
                  validation_count=int(validation.sum()),
                  unknown_distance=unknown_distance, simulation_count=len(angles),
                  settings=json.loads(model_settings()))
    report_path = Path(Proprioception.training_report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report
