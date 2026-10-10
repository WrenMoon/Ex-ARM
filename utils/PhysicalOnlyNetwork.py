"""A small neural classifier trained only on physical grasp angles."""

import hashlib
import json
from pathlib import Path

import numpy as np

from utils.Constants import Connection, PhysicalOnly, Proprioception


def network_settings():
    return json.dumps(dict(
        grip=Proprioception.grip,
        motor_ids=Connection.ids,
        motor_offsets_deg=Connection.offsets,
        hidden_size=PhysicalOnly.network_hidden_size,
        epochs=PhysicalOnly.network_epochs,
        patience=PhysicalOnly.network_patience,
        learning_rate=PhysicalOnly.network_learning_rate,
        weight_decay=PhysicalOnly.network_weight_decay,
        noise_deg=PhysicalOnly.network_noise_deg,
        finger_noise_deg=PhysicalOnly.network_finger_noise_deg,
        noisy_copies=PhysicalOnly.network_noisy_copies,
        validation_folds=PhysicalOnly.network_validation_folds,
        random_seed=PhysicalOnly.network_random_seed,
        temperature_min=PhysicalOnly.network_temperature_min,
        temperature_max=PhysicalOnly.network_temperature_max,
        min_probability=PhysicalOnly.network_min_probability,
        min_margin=PhysicalOnly.network_min_margin,
        distance_multiplier=PhysicalOnly.network_distance_multiplier,
    ), sort_keys=True)


def sample_fingerprint(samples):
    content = json.dumps(samples, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(content.encode()).hexdigest()


def normalize_angles(angles):
    angles = np.asarray(angles, dtype=float)
    if angles.shape[-1] != 16 or not np.isfinite(angles).all():
        raise ValueError("Expected 16 finite joint angles")
    start = np.asarray(Proprioception.grip["start_angles"], dtype=float)
    span = np.asarray(Proprioception.grip["max_angles"], dtype=float) - start
    span[span == 0] = np.inf
    return (angles - start) / span


def add_angle_noise(angles, labels, seed):
    copies = PhysicalOnly.network_noisy_copies
    if copies < 0:
        raise ValueError("The number of noisy copies cannot be negative")
    rng = np.random.default_rng(seed)
    repeated = np.repeat(angles, copies + 1, axis=0)
    noise = rng.normal(0, PhysicalOnly.network_noise_deg, repeated.shape)
    finger_noise = rng.normal(0, PhysicalOnly.network_finger_noise_deg,
                              (len(repeated), 4))
    noise += np.repeat(finger_noise, 4, axis=1)
    noise[::copies + 1] = 0
    return repeated + noise, np.repeat(labels, copies + 1)


class PhysicalOnlyNetwork:
    def __init__(self, class_names, seed=None):
        self.class_names = list(class_names)
        rng = np.random.default_rng(PhysicalOnly.network_random_seed if seed is None else seed)
        hidden = PhysicalOnly.network_hidden_size
        self.weights1 = rng.normal(0, np.sqrt(1 / 16), (16, hidden))
        self.bias1 = np.zeros(hidden)
        self.weights2 = rng.normal(0, np.sqrt(1 / hidden), (hidden, len(class_names)))
        self.bias2 = np.zeros(len(class_names))
        self.reference_angles = np.empty((0, 16))
        self.max_distance = 0.0
        self.temperature = 1.0
        self.fingerprint = ""

    def logits(self, angles):
        values = normalize_angles(angles)
        hidden = np.tanh(values @ self.weights1 + self.bias1)
        return hidden @ self.weights2 + self.bias2

    def outputs(self, angles):
        values = normalize_angles(angles)
        hidden = np.tanh(values @ self.weights1 + self.bias1)
        logits = (hidden @ self.weights2 + self.bias2) / self.temperature
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        probabilities = np.exp(shifted)
        probabilities /= np.sum(probabilities, axis=-1, keepdims=True)
        return probabilities, hidden, values

    def fit(self, angles, labels, epochs, seed, validation=None):
        noisy_angles, noisy_labels = add_angle_noise(angles, labels, seed)
        values = normalize_angles(noisy_angles)
        counts = np.bincount(labels, minlength=len(self.class_names))
        weights = 1 / counts[noisy_labels]
        weights /= np.sum(weights)

        parameters = [self.weights1, self.bias1, self.weights2, self.bias2]
        momentum = [np.zeros_like(parameter) for parameter in parameters]
        variance = [np.zeros_like(parameter) for parameter in parameters]
        best_parameters = None
        best_loss = np.inf
        best_epoch = epochs
        waiting = 0

        for epoch in range(1, epochs + 1):
            hidden = np.tanh(values @ self.weights1 + self.bias1)
            logits = hidden @ self.weights2 + self.bias2
            shifted = logits - np.max(logits, axis=1, keepdims=True)
            probabilities = np.exp(shifted)
            probabilities /= np.sum(probabilities, axis=1, keepdims=True)
            gradient = probabilities
            gradient[np.arange(len(noisy_labels)), noisy_labels] -= 1
            gradient *= weights[:, None]

            regularization = PhysicalOnly.network_weight_decay
            gradient2 = hidden.T @ gradient + regularization * self.weights2
            bias_gradient2 = np.sum(gradient, axis=0)
            hidden_gradient = (gradient @ self.weights2.T) * (1 - hidden ** 2)
            gradient1 = values.T @ hidden_gradient + regularization * self.weights1
            bias_gradient1 = np.sum(hidden_gradient, axis=0)

            for index, update in enumerate((gradient1, bias_gradient1,
                                             gradient2, bias_gradient2)):
                momentum[index] = 0.9 * momentum[index] + 0.1 * update
                variance[index] = 0.999 * variance[index] + 0.001 * update ** 2
                corrected_momentum = momentum[index] / (1 - 0.9 ** epoch)
                corrected_variance = variance[index] / (1 - 0.999 ** epoch)
                parameters[index] -= PhysicalOnly.network_learning_rate * corrected_momentum / (
                    np.sqrt(corrected_variance) + 1e-8)

            if validation is None:
                continue
            validation_angles, validation_labels = validation
            validation_probabilities = self.outputs(validation_angles)[0]
            likelihoods = validation_probabilities[
                np.arange(len(validation_labels)), validation_labels]
            loss = float(-np.mean(np.log(np.maximum(likelihoods, 1e-12))))
            if loss < best_loss - 1e-6:
                best_loss = loss
                best_epoch = epoch
                best_parameters = [parameter.copy() for parameter in parameters]
                waiting = 0
            else:
                waiting += 1
                if waiting >= PhysicalOnly.network_patience:
                    break

        if best_parameters is not None:
            for parameter, best in zip(parameters, best_parameters):
                parameter[:] = best
        return best_epoch

    def evaluate(self, probabilities, distance):
        probabilities = np.asarray(probabilities, dtype=float)
        best = int(np.argmax(probabilities))
        ranked = np.sort(probabilities)
        confidence = float(ranked[-1])
        margin = float(ranked[-1] - ranked[-2])
        if distance > self.max_distance:
            class_name = "unknown"
            reason = "far from training grasps"
        elif confidence < PhysicalOnly.network_min_probability:
            class_name = "uncertain"
            reason = "low class probability"
        elif margin < PhysicalOnly.network_min_margin:
            class_name = "uncertain"
            reason = "similar class probabilities"
        else:
            class_name = self.class_names[best]
            reason = None
        return dict(class_name=class_name, best_class=self.class_names[best],
                    confidence=confidence, margin=margin, distance=float(distance),
                    reason=reason, probabilities={name: float(value)
                                                for name, value in zip(self.class_names,
                                                                       probabilities)})

    def predict(self, angles):
        if len(self.reference_angles) == 0:
            raise ValueError("The neural model has no physical reference grasps")
        probabilities = self.outputs(angles)[0]
        distances = np.linalg.norm(normalize_angles(self.reference_angles)
                                   - normalize_angles(angles), axis=1)
        distance = float(np.min(distances))
        return self.evaluate(probabilities, distance)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, weights1=self.weights1, bias1=self.bias1,
                 weights2=self.weights2, bias2=self.bias2,
                 class_names=np.asarray(self.class_names),
                 reference_angles=self.reference_angles,
                 max_distance=self.max_distance,
                 temperature=self.temperature,
                 fingerprint=self.fingerprint, settings=network_settings())

    @classmethod
    def load(cls, path, fingerprint):
        with np.load(path, allow_pickle=False) as saved:
            if str(saved["settings"]) != network_settings():
                raise ValueError("The grasp or neural settings changed. Retrain the neural model")
            if str(saved["fingerprint"]) != fingerprint:
                raise ValueError("The physical trials changed. Retrain the neural model")
            model = cls(saved["class_names"].tolist())
            model.weights1 = saved["weights1"].copy()
            model.bias1 = saved["bias1"].copy()
            model.weights2 = saved["weights2"].copy()
            model.bias2 = saved["bias2"].copy()
            model.reference_angles = saved["reference_angles"].copy()
            model.max_distance = float(saved["max_distance"])
            model.temperature = float(saved["temperature"])
            model.fingerprint = fingerprint
        return model


def train_network(samples, class_names, model_path, report_path):
    counts = {name: sum(sample["class_name"] == name for sample in samples)
              for name in class_names}
    if min(counts.values()) < 3:
        raise ValueError("The neural model needs at least three physical trials per class")

    angles = np.asarray([sample["angles_deg"] for sample in samples], dtype=float)
    labels = np.asarray([class_names.index(sample["class_name"]) for sample in samples],
                        dtype=int)
    folds = min(PhysicalOnly.network_validation_folds, min(counts.values()))
    if folds < 2:
        raise ValueError("At least two validation folds are needed")

    validation_predictions = []
    validation_distances = []
    validation_logits = []
    validation_labels = []
    best_epochs = []
    for fold in range(folds):
        validation = np.zeros(len(samples), dtype=bool)
        for class_index in range(len(class_names)):
            indices = np.flatnonzero(labels == class_index)
            validation[indices[fold::folds]] = True
        training = ~validation
        model = PhysicalOnlyNetwork(class_names,
                                    seed=PhysicalOnly.network_random_seed + fold)
        best_epoch = model.fit(angles[training], labels[training],
                               PhysicalOnly.network_epochs,
                               PhysicalOnly.network_random_seed + fold,
                               validation=(angles[validation], labels[validation]))
        best_epochs.append(best_epoch)
        probabilities = model.outputs(angles[validation])[0]
        prediction = np.argmax(probabilities, axis=1)
        validation_predictions.extend(zip(labels[validation].tolist(), prediction.tolist()))
        validation_logits.extend(model.logits(angles[validation]))
        validation_labels.extend(labels[validation])
        references = normalize_angles(angles[training])
        for row in normalize_angles(angles[validation]):
            validation_distances.append(float(np.min(np.linalg.norm(references - row, axis=1))))

    epochs = max(1, int(np.median(best_epochs)))
    final_model = PhysicalOnlyNetwork(class_names)
    final_model.fit(angles, labels, epochs, PhysicalOnly.network_random_seed + 100)
    heldout_logits = np.asarray(validation_logits)
    heldout_labels = np.asarray(validation_labels, dtype=int)
    temperatures = np.geomspace(PhysicalOnly.network_temperature_min,
                                PhysicalOnly.network_temperature_max, 100)
    losses = []
    for temperature in temperatures:
        shifted = heldout_logits / temperature
        shifted -= np.max(shifted, axis=1, keepdims=True)
        probabilities = np.exp(shifted)
        probabilities /= np.sum(probabilities, axis=1, keepdims=True)
        correct_probabilities = probabilities[np.arange(len(heldout_labels)),
                                              heldout_labels]
        losses.append(-np.mean(np.log(np.maximum(correct_probabilities, 1e-12))))
    final_model.temperature = float(temperatures[int(np.argmin(losses))])
    final_model.reference_angles = angles.copy()
    final_model.max_distance = max(
        float(np.quantile(validation_distances, 0.95))
        * PhysicalOnly.network_distance_multiplier, 1e-6)
    final_model.fingerprint = sample_fingerprint(samples)
    final_model.save(model_path)

    correct = sum(actual == predicted for actual, predicted in validation_predictions)
    confusion = {actual: {predicted: 0 for predicted in class_names}
                 for actual in class_names}
    for actual, predicted in validation_predictions:
        confusion[class_names[actual]][class_names[predicted]] += 1
    accepted = 0
    accepted_correct = 0
    unknown = 0
    uncertain = 0
    for logits, label, distance in zip(heldout_logits, heldout_labels,
                                       validation_distances):
        shifted = logits / final_model.temperature
        shifted -= np.max(shifted)
        probabilities = np.exp(shifted)
        probabilities /= np.sum(probabilities)
        result = final_model.evaluate(probabilities, distance)
        if result["class_name"] in class_names:
            accepted += 1
            accepted_correct += result["class_name"] == class_names[label]
        elif result["class_name"] == "unknown":
            unknown += 1
        else:
            uncertain += 1
    report = dict(classes=class_names, trial_counts=counts,
                  physical_trials=len(samples), validation_folds=folds,
                  validation_correct=correct,
                  validation_total=len(validation_predictions),
                  validation_confusion=confusion,
                  validation_accepted=accepted,
                  validation_accepted_correct=accepted_correct,
                  validation_unknown=unknown,
                  validation_uncertain=uncertain,
                  final_training_epochs=epochs,
                  temperature=final_model.temperature,
                  distance_limit=final_model.max_distance,
                  note="Validation uses real trials before noise augmentation. "
                       "Unknown-object rejection has not been validated on unseen classes.")
    Path(report_path).write_text(json.dumps(report, indent=2) + "\n")
    return final_model, report
