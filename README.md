# Real-Time Driver Monitoring System (DSDM-M1)

Welcome to the **Real-Time Driver Monitoring System (DSDM-M1)** repository. This project aims to build a robust, camera-based in-cabin Driver Monitoring System (DMS) that detects drowsiness, sleep, distraction, and inattention in real-time.

## Project Overview

The blueprint for this project breaks the problem into a three-phase implementation:
1. **Phase 1: Baseline Prototype** — A complete rule-based detection chain using geometric features (EAR, MAR, Head Pose) without custom training.
2. **Phase 2: Improved Temporal Model** — Enhancing detection with lightweight trained models (e.g., gaze classifiers, GRU/TCN for drowsiness, and phone-behavior detection).
3. **Phase 3: Robust Real-Time Deployment** — Edge optimization using ONNX/TensorRT and robustness testing against lighting, visual occlusions, and diverse driver populations.

## Datasets

The project utilizes several public datasets for training and benchmarking our distraction and drowsiness detection models.

### Primary Dataset: MRL Eye Dataset
We currently have the **MRL Eye Dataset** loaded for training the eye state (open/closed) classifier. This dataset is crucial for the foundational "Eye Aspect Ratio (EAR)" and blink detection heuristics on which the drowsiness scoring relies.
good 
**Dataset Location:**
The dataset has been structured locally into training directories containing cropped eye images:
* `Close-Eyes/`: Contains **41,948** images of closed eyes.
* `Open-Eyes/`: Contains **42,950** images of open eyes.

*Total Images: 84,898*

**Formatting Details:**
The MRL Eye Dataset consists of large-scale human eye images under different lighting conditions. This immense variability (infrared vs. RGB, high vs. low resolution, varying camera sensors) allows our eye-state detection model to become highly robust against the "noise variables" typical in an automotive cabin environment, such as reflections, eyeglasses, or poor lighting.

## Setup & Implementation

Further implementation modules will be committed according to the project's phasing plan. Please refer to the documentation and blueprints located in this repository for an in-depth understanding of the temporal alerting state machine, metric calculations, and edge optimization rules aligned with Euro NCAP standards.
