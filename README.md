# Benefit-Gated Adaptive Inference for Resource-Efficient Wearable Human Activity Recognition
<img width="2752" height="1536" alt="image" src="https://github.com/user-attachments/assets/2fccd0f2-49ef-4643-9919-1625e655c410" />
This repository implements the methodology proposed in the paper "Benefit-Gated Adaptive Inference for Resource-Efficient Wearable Human Activity Recognition"

## Paper Overview

**Abstract**: Continuous wearable sensing requires activity-recognition models to operate under tight latency and energy constraints, making it inefficient to execute the same amount of computation for every sensor window. Existing early-exit approaches reduce this cost by routing samples according to intermediate confidence, entropy, or related uncertainty measures. These criteria quantify the reliability of the current prediction, but they do not directly estimate the marginal value of executing additional computation. This paper presents a benefit-gated adaptive inference framework for resource-aware on-device human activity recognition (HAR). A compact shared MiniMamba backbone produces an early representation and prediction, and a lightweight gate estimates the sample-wise predictive benefit of executing the remaining temporal blocks. The gate is supervised from paired early- and full-route outcomes, explicitly accounting for error correction, harmful over-processing, and meaningful reductions in sample-level cross-entropy loss. The resulting policy allocates deeper computation only to windows for which additional processing is predicted to be useful. Across five public wearable HAR datasets, dynamic inference matches or exceeds Full-only Macro-F1 on three datasets and remains within 0.17 percentage points of it on the other two, while reducing average FLOPs by 48.9%. Under compute-matched routing across all five datasets, benefit-based selection consistently outperforms confidence-, entropy-, and margin-based routing at identical full-execution ratios and computational cost. Raspberry Pi 4B measurements further show latency reductions of 47.0–52.0% and energy reductions of 50.6–53.1% relative to Full-only inference. These results show that sample-wise predictive benefit estimation can provide an effective basis for adaptive computation allocation in resource-constrained wearable and edge sensing, reducing unnecessary processing while preserving near-full-route recognition performance.

## Dataset
This repository does not include datasets. Please download them from the official sources below and configure the dataset path accordingly.
- **UCI-HAR** dataset is available at https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
- **MotionSense** dataset is available at https://www.kaggle.com/datasets/malekzadeh/motionsense-dataset
- **MHEALTH** dataset is available at https://archive.ics.uci.edu/dataset/319/mhealth+dataset
- **WISDM** dataset is available at https://www.cis.fordham.edu/wisdm/dataset.php
- **PAMAP2** dataset is available at https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring

## Codebase Overview
- `model.py`: Implementation of the proposed benefit-gated adaptive inference framework for resource-efficient wearable human activity recognition, including a compact shared MiniMamba backbone, early and full inference routes, sample-wise benefit estimation, dynamic conditional execution, benefit-target construction, and the complete training objective.

## Citing this Repository
If you use this code in your research, please cite:
```
@article{BGAI,
  title   = {Benefit-Gated Adaptive Inference for Resource-Efficient Wearable Human Activity Recognition},
  author  = {Jimin Kim and Myung-Kyu Yi},
  journal = {},
  volume  = {},
  number  = {},
  pages   = {},
  year    = {},
  publisher = {}
}
```

## License
This project is licensed under the MIT License.
See the [LICENSE](LICENSE) file for details.

## Contact
For questions or issues, please contact:
  - Jimin Kim: sispo3314@gmail.com
