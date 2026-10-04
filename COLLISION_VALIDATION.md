# Experimental collision-model validation

The original model/configuration and the selected five-probe references are unchanged.
Experimental models are not hardware motion instructions.

## Reproduce the v2 check

From the repository directory:

```bash
venv/bin/python -u Validate_Collision_Model.py \
  --results /tmp/leap-fixed-library-grasps.json \
  --model /tmp/leap-decomposed-hand-v2/hand_decomposed.urdf \
  --output /tmp/leap-v2-validation.json \
  --fresh-results /tmp/leap-v2-fresh-grasps.json
```

Exit status 1 means replay or discrimination failed. Exit status 0 means replay and
simulated discrimination completed, **not** collision clearance or hardware safety.

The script uses a diagnostic subclass that continues through self-collisions to
measure them. It does not modify the production simulator's collision policy.
Object-contact, joint-limit, approach and return checks remain enabled. It resets
to zero for each independent diagnostic, not as a physically safe recovery motion.
All MuJoCo-generated hand penetration pairs are reported, including new pairs,
deeper baseline overlaps and same-body decomposition-piece contacts. Same-body
piece overlaps are not interference between independent links. Probe contact
refinement trial poses are included in the diagnostics.

Fresh signatures do not have to match original stopping angles. They are tested
for separation with the saved threshold and for contact coverage. Identification
is checked against regenerated references, not held-out physical observations.
Any replay error prevents a complete recognition claim. All fresh measurements
are marked `path_verified: false`; the experimental output must not be used to
command hardware. Old library measurements are removed from the experimental
result because they belong to a different collision model.

## Current v2 blocker

The first complete attempted run failed all 315 paths during approach. The model
reports fixed palm/object contact at zero, so no probe can start. For cube@1.5,
palm pieces 3, 4, 5, 8 and 12 give negative geometry-query distances of about
0.414 mm. The updated report records zero-pose fixed-object distances for every
object. No calibration offset or collision tolerance was changed to hide this.

The seven-pose comparison was not sufficient to detect this regression because
it inspected only previously problematic hand body pairs, not palm/object
clearance. CoACD preprocessing/approximation is a possible contributor; it is not
yet established whether the reported distances reflect actual STL interference,
approximation error or collision-query behavior.

Do not adopt v2. Next, independently compare the palm decomposition against the
original palm STL near the mounted-object interface and verify the negative
geometry distances. Correct the model generation if confirmed; do not simply
move the object or suppress contact. Then repeat full-path validation, regenerate
references, and rerun probe selection only if the existing five no longer work.
Until that blocker is resolved, there is no complete-path assessment of v2's
remaining MCP overlaps and no hardware clearance certification.
