# Homer and Bart Character Classifier

This project implements a simple image classifier to distinguish between Homer Simpson and Bart Simpson from "The Simpsons" animated series. The classification is based on color-based feature extraction from their iconic appearances.

## Features Used

The classifier extracts the following features by detecting specific color ranges in the images:
- Percentage of Homer's brown mouth pixels.
- Percentage of Homer's blue pants pixels.
- Percentage of Homer's gray shoes pixels.
- Percentage of Bart's orange/red t-shirt pixels (TODO: Needs implementation in code).
- Percentage of Bart's blue shorts pixels.
- Percentage of Bart's white/red sneakers pixels (TODO: Needs implementation in code).

These extracted features are then used to train a simple machine learning model (e.g., a classifier from `scikit-learn`).

## Getting Started

### Prerequisites

- Python 3.x
- OpenCV (`cv2`)
- Matplotlib (`matplotlib`)
- Pandas (`pandas`)
- Scikit-learn (`sklearn`)

You can install these packages using pip:
```bash
pip install opencv-python matplotlib pandas scikit-learn
