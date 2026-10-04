from utils.ProprioceptionSimulation import run_simulations
from utils.ProprioceptionModel import train_model


def main():
    run_simulations()
    results = train_model()
    print(f"Simulated grasps: {results['simulation_count']}")
    print(f"Class accuracy on held-out scales: {results['validation_accuracy']:.1%}")
    print(f"Size error on held-out scales: {results['validation_size_mae_mm']:.2f} mm")
    print(f"Accepted held-out grasps: {results['accepted_count']} / {results['validation_count']}")
    print(f"Correct accepted grasps: {results['accepted_correct']} / {results['accepted_count']}")
    print(f"Unknown-distance threshold: {results['unknown_distance']:.3f}")


if __name__ == "__main__":
    main()
