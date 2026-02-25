# MHR Optimization Pipeline Flow

This document summarizes the complete flow implemented in `example_optimization.py`.

## High-Level Goal

Optimize a trainable mesh (initialized with angular pose displacement) to match a zero-pose target mesh using a Torch Chamfer-distance objective.

## End-to-End Flowchart

```mermaid
flowchart TD
    A[Start Script] --> B[Select Device CPU/GPU]
    B --> C[Load MHR Model and Character]

    C --> D[Create Target Mesh]
    D --> D1[Initialize zero params: pose=0, shape=0, scale=0]
    D1 --> D2[Forward pass no_grad]
    D2 --> D3[Target vertices]

    C --> E[Create Initial Mesh]
    E --> E1[Initialize trainable params]
    E1 --> E2[Set pose displacement from zero pose]
    E2 --> E3[Forward pass no_grad]
    E3 --> E4[Initial vertices]

    D3 --> F[Save pre-optimization artifacts]
    E4 --> F
    E1 --> F1[Save params_before.pt]
    F --> F2[Save target/initial .pt + .obj]

    D3 --> G[Compute Initial Chamfer]
    E4 --> G

    G --> H[Optimization Loop]
    H --> H1[Forward predicted vertices]
    H1 --> H2[Chamfer loss pred vs target]
    H2 --> H3[Add parameter regularization]
    H3 --> H4[Backward + grad clip]
    H4 --> H5[Optimizer step]
    H5 --> H6{More iterations?}
    H6 -- Yes --> H
    H6 -- No --> I[Get Final vertices]

    I --> J[Compute Final Chamfer]
    I --> K[Save post-optimization artifacts]
    K --> K1[Save params_after.pt]
    K --> K2[Save final .pt + .obj]
    K --> K3[Render mesh_comparison.png]

    J --> L[Print metrics + output path]
    L --> M[End]
```

## Core Stages

1. **Model Initialization**
   - Load `MHR` from assets.
   - Access `character_torch` for skinning + mesh topology (`faces`).

2. **Target Mesh Construction**
   - Build `DifferentiableMHRParameters` with all-zero pose/shape/scale.
   - Run forward pass to get target vertices.

3. **Initial Mesh Construction**
   - Build trainable parameters.
   - Apply angular displacement to pose from zero-pose baseline.
   - Run forward pass to get initial vertices.

4. **Optimization**
   - Loss = Torch Chamfer distance + small L2 regularization on trainable params.
   - Optimize with independent learning rates per parameter group.
   - Use gradient clipping for stability.

5. **Outputs**
   - Save before/after params: `params_before.pt`, `params_after.pt`.
   - Save meshes as tensors and OBJ: target/initial/final.
   - Save one visual summary image: `mesh_comparison.png`.
   - Print initial/final Chamfer for quick quality check.

## Files Produced

All outputs are written to:

- `outputs/mhr_optimization/`

Typical artifacts:

- `params_before.pt`
- `params_after.pt`
- `target_vertices.pt`, `initial_vertices.pt`, `final_vertices.pt`
- `target_mesh.obj`, `initial_mesh.obj`, `final_mesh.obj`
- `mesh_comparison.png`
