import onnxruntime as ort
import numpy as np

class ExerkineONNXClassifier:
    def __init__(self, model_path):
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        
    def preprocess_spatial_data(self, lr_expression_matrix):
        """
        Convert L-R interaction matrix to 28x28 image
        lr_expression_matrix: spatial coordinates with L-R pair expressions
        """
        # Reshape/interpolate to 28x28
        # Normalize to [0, 1] or standardize
        pass
    
    def predict(self, spatial_features):
        """Classify spatial interaction patterns"""
        input_data = spatial_features.reshape(1, 28, 28, 1).astype(np.float32)
        outputs = self.session.run(None, {self.input_name: input_data})
        return outputs[0]  # 10-class probabilities
