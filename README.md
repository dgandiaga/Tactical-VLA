# Tactical-VLA: Causal Attention for Autonomous Mission Planning

### **Architecture Overview**
This repository implements a **Vision-Language-Action (VLA) Backbone** designed for aerial autonomous platforms (Jetson AGX Orin). Unlike standard LLMs, this architecture is optimized for **continuous 6-DOF control** while maintaining the temporal reasoning of a Transformer.

---

## **Core Design Principles**

### **1. Causal Self-Attention (Temporal Autoregression)**
We implement **Multi-Head Attention (MHA)** with a lower-triangular mask. 
* **The Physics:** In mission planning, the drone must not "cheat" by looking at future waypoints. Every action $\vec{a}_t$ must be a strict function of the history $\mathcal{H} = \{s_0, s_1, \dots, s_t\}$.
* **The Math:** We use **4D-Tensor Vectorization** `(B, nh, T, hs)` to process multiple attention heads in parallel, ensuring high-throughput inference on embedded NPU/GPU hardware.



### **2. Numerical Stability & Gradient Flow**
To ensure the model learns complex tactical maneuvers without degrading:
* **Scaled Dot-Product:** Alignment scores are scaled by $\frac{1}{\sqrt{d_k}}$ to prevent **Softmax Saturation**. This keeps gradients in the "sloped" region of the activation, avoiding the "Vanishing Gradient" trap common in high-dimensional latent spaces.
* **Pre-Norm Residual Highways:** Following modern SOTA (Qwen/Llama), we apply `LayerNorm` *before* MHA and FFN. This preserves the "Identity Path," allowing the gradient to flow unimpeded from the Action Head back to the Vision Bridge.



### **3. Physical Grounding (The Action Head)**
Unlike text-based models that use Softmax for discrete token selection, this controller uses a **Tanh-constrained Regression Head**.
* **Safety & Avionics:** The output is squashed to $(-1, 1)$, which is then mapped to real-world kinematic limits (e.g., max pitch rate, thrust N). This prevents "hallucinated" commands that would violate the flight envelope of the platform.

---

## **Technical Implementation Details**

| Feature | Implementation | Engineering Justification |
| :--- | :--- | :--- |
| **Normalization** | `LayerNorm` | Agnostic to batch size; critical for single-sample ROS inference. |
| **Non-Linearity** | `GELU` | Smoother derivative than ReLU; prevents "Dead Neurons" during tactical training. |
| **Matmul** | `Transpose(-2, -1)` | Dimension-agnostic logic for robust 4D batch matrix multiplication. |
| **Memory** | `register_buffer` | Pre-computed causal masks to avoid GPU re-allocation during the control loop. |

---

## **Future Roadmap: The Project Vision**

1. **Distillation:** Implement **Linear Alignment** (https://arxiv.org/abs/2312.16886) to enable **Knowledge Distillation** from 30B Flagship models to Edge-optimized Student models for continous control.
2. **Aerostack 2.0 Integration:** Wrapping this backbone in a ROS2 node to evaluate Foundation Model-based mission planning against traditional Behavior Trees.
3. **Jetson Optimization:** Applying INT8 Quantization and TensorRT pruning to fit the distilled model on the AGX Orin for real-time onboard execution.

---
