# Liver-Tumor-Segmentation-Project
---

### **Introduction**

The Liver Tumor Segmentation project utilizes deep learning techniques to automate and improve the accuracy of liver tumor segmentation from CT scan images. Accurate segmentation is vital for diagnosis, treatment planning, and monitoring tumor progression. The project's codebase implements a structured pipeline encompassing data preprocessing, model development, training, evaluation, and results interpretation.

The primary objective was to process medical images, train models, and achieve precise tumor segmentation using the U-Net and ResNet-50 architectures, entirely driven by code logic.

---

### **1. Project Motivation**

Medical imaging, particularly CT scans, plays a critical role in liver tumor diagnosis. Manual segmentation is labor-intensive and subjective, motivating the implementation of automated segmentation using deep learning models. The code is designed to:

- Enhance diagnostic accuracy.
- Automate segmentation to assist radiologists.
- Improve efficiency and consistency in medical imaging.

---

### **2. Code Structure and Dataset Integration**

The codebase is modular, ensuring readability, maintainability, and scalability.

#### **2.1 Dataset Loading**
- Utilizes Python's `nibabel` library to load `.nii` files.
- Extracts CT scans and corresponding tumor masks into structured arrays.
- The data is organized into training and testing subsets.

#### **2.2 Data Preprocessing Pipeline**
The preprocessing code is critical for converting raw medical images into model-compatible formats:

- **Orientation Correction:** Rotates images to align with standard anatomical views.
- **Intensity Windowing:** Implements a windowing function to adjust pixel intensities, emphasizing liver tissues.
- **Histogram Equalization:** Applies histogram-based scaling to normalize intensity distributions.
- **Multi-Channel Conversion:** Combines different intensity windows to create richer, multi-dimensional input features.
- **Augmentation:** Random rotations, flips, and intensity shifts are applied using `imgaug` to increase dataset diversity.

---

### **3. Model Development**

The code implements two deep learning models using TensorFlow/Keras:

#### **3.1 U-Net Model**
The U-Net architecture, designed for biomedical image segmentation, is implemented with:

- An **encoder-decoder structure** with skip connections.
- Custom loss functions tailored to medical image segmentation tasks.
- Convolutional layers with ReLU activation for feature extraction.
- Up-sampling layers to reconstruct segmentation masks.

**Key Code Highlights:**
- Defined with `tf.keras.Sequential`.
- Integrated skip connections via `concatenate` operations.
- Optimized using the **Adam optimizer** with a learning rate of 1e-4.

#### **3.2 ResNet-50 Model**
ResNet-50, a classification model, was adapted for segmentation:

- Pre-trained on ImageNet and fine-tuned for liver segmentation.
- Custom upsampling layers added after feature extraction.
- Transfer learning implemented via `include_top=False`.

**Key Code Highlights:**
- Layer freezing and unfreezing via `model.layers[i].trainable`.
- Custom segmentation head added after ResNet's feature extractor.

---

### **4. Model Training and Optimization**

The training logic ensures systematic model learning:

- **Batch Generation:** Custom `DataGenerator` class to handle large volumes of CT scan slices.
- **Loss Calculation:** Binary cross-entropy loss combined with Dice loss.
- **Optimization:** Adaptive learning rates via `ReduceLROnPlateau` callback.
- **Regularization:** L2 regularization applied to convolutional layers to mitigate overfitting.
- **Monitoring:** Utilized `TensorBoard` and `Matplotlib` for loss/accuracy visualization.

**Training Parameters:**
- Epochs: **50 (U-Net)**, **10 (ResNet-50)**.
- Batch Size: **16 (U-Net)**, **4 (ResNet-50)**.
- Validation Split: **20%**.

---

### **5. Model Evaluation and Metrics Implementation**

The evaluation code rigorously assesses model performance:

- **Dice Coefficient:** Implemented as `2 * (TP)/(2*(TP) + FP + FN)`.
- **Intersection over Union (IoU):** Calculated via logical mask operations.
- **ROC-AUC:** Leveraged `sklearn.metrics.roc_auc_score` for performance curves.

**Evaluation Code Insights:**
- Predictions compared against ground truth masks.
- Overlays generated to visualize predicted tumor boundaries.

---

### **6. Code-Driven Results Analysis**

| **Model** | **Mean Dice Coefficient** | **Mean IoU** | **ROC-AUC** |
|-----------|--------------------------|--------------|-------------|
| U-Net     | 0.8215                   | 0.7976       | 1.00        |
| ResNet-50 | 0.7711                   | 0.7318       | 0.99        |

**Key Insights:**
- U-Net's skip connections significantly improved segmentation performance.
- ResNet-50 showed instability in loss convergence, indicating architectural limitations for segmentation tasks.
- Model predictions visualized via `matplotlib.pyplot` confirmed U-Net's superior boundary delineation.

---

### **7. Error Analysis**

The codebase includes logic for error analysis:

- **False Positives:** High-intensity textures misclassified as tumors.
- **False Negatives:** Low-contrast tumors not detected.
- **Mitigation:** Adjusted class weights and experimented with `focal loss`.

---

### **8. Limitations and Future Code Enhancements**

- **Dataset Limitations:** Restricted to 200 CT scans.
- **Model Generalization:** Further tuning of hyperparameters (e.g., `learning_rate`, `batch_size`).
- **Integration Prospects:** Modify the pipeline to support MRI datasets.

---

### **Conclusion**

The code-driven implementation of liver tumor segmentation demonstrated that U-Net, with its robust architecture, outperforms ResNet-50 in terms of accuracy and reliability. This project highlights deep learning's potential to aid medical professionals by automating a crucial diagnostic task.

**Code Impact:**
- Improved segmentation precision with minimal manual intervention.
- Enhanced understanding of tumor morphology for clinical applications.

**Next Steps:**
- Increase dataset size for model generalization.
- Explore advanced architectures like **Attention U-Net**.
- Develop deployment scripts for potential real-time applications.

**In essence,** this project showcases the power of deep learning, transforming complex medical imaging tasks into efficient, reliable, and scalable processes.
