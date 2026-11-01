# Project profile

## SortSmart — AI-Powered Robotic Recycling

**Team:** SortSmart  
**Track:** Physical AI / robotic material recovery  
**Prototype target:** discarded AAA batteries  
**Platform:** NVIDIA DGX Spark, NVIDIA Isaac Sim, FANUC ER-4iA, RGB/depth cameras  
**Perception:** fine-tuned YOLO instance segmentation  

### One-line description

A closed-loop robotic recycling system that learns to identify, localize and safely recover valuable materials from clutter, using a digital twin to accelerate iteration before physical deployment.

### Problem

Mixed waste contains reusable materials and hazardous objects that are difficult to identify and manipulate reliably. Manual sorting is slow and exposes workers to risk; fixed automation struggles when products, lighting and waste composition change.

### Solution

SortSmart combines a reusable data loop with calibrated robotic manipulation:

1. Capture phone, table-camera and wrist-camera images.
2. Label target instances and include clutter-only negative samples.
3. Fine-tune a segmentation model and review misses/false positives.
4. Recreate real object appearances and randomized scenes in Isaac Sim.
5. Convert masks into calibrated position/yaw, apply safety gates, then grasp and place.
6. Feed difficult cases and new material categories into the next cycle.

### Evidence

- Fine-tuning improved held-out recall from **65.0% to 95.0%** and reduced missed batteries from **14 to 2**.
- The digital twin completed **12/12 randomized end-to-end trials**, including pickup, transport and placement.
- The physical cell successfully detected, localized and grasped an AAA battery; the remaining failure was loss of grip from smooth stock fingers.
- A known-height depth check measured a 55–56 mm box at a median **54.87 mm** with **0.08 mm** standard deviation.

### Industrial potential

AAA batteries are the first recovery target, not the boundary of the system. The same workflow can be extended to other batteries, metals, paper and valuable components. Simulation provides a repeatable sandbox for perception and motion validation; an operational ontology can later coordinate facility-level inventory, routing, exceptions, human approvals and continuous improvement.

### Immediate next steps

- 3D-print or rubberize high-friction gripper fingers.
- Collect more damaged, occluded and multi-class recycling examples.
- Expand randomized simulation and physical-cell evaluation.
- Connect cell-level evidence to facility-level operational decisions and review workflows.

