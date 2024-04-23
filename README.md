**Description:**

This repository explores image classification tasks using various popular deep learning architectures, including:

*   **EfficientNet:** A family of models that achieve state-of-the-art accuracy with efficient use of parameters. 
*   **ResNet (Residual Networks):**  Known for their ability to train very deep networks while avoiding vanishing gradient problems.
*   **VggNet:** A classic architecture that achieved significant performance with its simple yet effective design.
*   **DenseNet:** Features dense connections between layers, leading to improved information flow and parameter efficiency. 
*   **Vision Transformer (ViT):** Leverages the Transformer architecture from NLP for image classification tasks, achieving competitive results.

**Contents:**

*   **imagesplit.py:** This script handles splitting the image dataset into training, validation, and testing sets.
*   **grad\_CAM:** This folder contains implementations or utilities related to Gradient-weighted Class Activation Mapping (Grad-CAM), a technique for visualizing the regions in an image that contribute most to a model's prediction.
*   Folders for each model (EfficientNet, ResNet, VggNet, DenseNet, Vision Transformer) contain:
    *   Model definition scripts
    *   Training scripts
    *   Evaluation scripts

**Getting Started:**

1.  **Clone the repository:** `git clone https://github.com/your-username/your-repo.git`
2.  **Install dependencies:** Create a virtual environment and install the required libraries.
3.  **Dataset preparation:** Organize image data into appropriate folders for training, validation, and testing. (See the given txt)
4.  **Run the scripts:**  Follow the instructions within each model's folder to train and evaluate the models. 
