import cv2
import torch
import numpy as np
import faiss
import json
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from huggingface_hub import login

HF_TOKEN = ""
login(token=HF_TOKEN)


class OODCameraDetector:
    def __init__(self, model_name="facebook/dinov3-vits16-pretrain-lvd1689m"):
        """
        Initializes the Out-of-Distribution (OOD) detector.
        Loads the DINOv3 model, FAISS dictionary (centroids), and the reference database.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔧 Initializing Engine on {self.device}...")

        print("📥 Loading DINOv3...")
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float32 if self.device.type == "cpu" else torch.float16,
            low_cpu_mem_usage=True
        ).to(self.device).eval()
        self.num_registers = getattr(self.model.config, "num_register_tokens", 4)

        with open('model/model_small_config.json', 'r') as f:
            config = json.load(f)
            self.K_NN = config["K_NN"]
            self.DISTANCE_THRESHOLD = config["DISTANCE_THRESHOLD"]
            self.CHAOS_THRESHOLD = config["CHAOS_THRESHOLD"]
            self.num_clusters = config["num_clusters"]

        print("📥 Loading FAISS Dictionary...")
        centroids = np.load('model/faiss_dictionary_small_k1024.npy')
        self.index = faiss.IndexFlatL2(centroids.shape[1])
        self.index.add(centroids)

        print("📥 Loading Reference Database...")
        db = np.load('model/knn_reference_small_database.npz')
        self.ref_histograms = db['histograms']
        self.ref_labels = db['labels']  # These are now strings!

        print("✅ Engine Ready!")

    def chi_square_distance(self, histA, histB):
        """
        Calculates the Chi-Square distance between a histogram and a set of reference histograms.
        Useful for comparing visual word frequency distributions.
        """
        eps = 1e-10
        return 0.5 * np.sum(((histA - histB) ** 2) / (histA + histB + eps), axis=1)

    def predict(self, image_frame):
        """
        Processes an input image frame and predicts whether it belongs to a known or unknown (OOD) class.
        Returns:
            label (str): The predicted class name or "UNKNOWN (OOD)".
            closest_distance (float): The distance to the nearest neighbor in the reference database.
            unique_classes_count (int): The number of unique classes among the K nearest neighbors (Chaos).
            is_ood (bool): True if the image is considered out-of-distribution.
        """
        pil_img = Image.fromarray(image_frame)

        with torch.no_grad():
            inputs = self.processor(images=pil_img, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            patch_tokens = outputs.last_hidden_state[0, 1 + self.num_registers:].cpu().to(torch.float32).numpy()

        _, visual_words = self.index.search(patch_tokens, 1)
        hist = np.bincount(visual_words.flatten(), minlength=self.num_clusters)
        hist = hist / (np.sum(hist) + 1e-10)

        distances = self.chi_square_distance(hist, self.ref_histograms)
        nearest_indices = np.argsort(distances)[:self.K_NN]
        k_distances = distances[nearest_indices]
        k_labels = self.ref_labels[nearest_indices]  # e.g., ['faces', 'faces', 'Motorbikes', ...]

        # Dual-Condition Heuristic
        closest_distance = k_distances[0]
        unique_classes = np.unique(k_labels)
        unique_classes_count = len(unique_classes)

        is_ood = (closest_distance > self.DISTANCE_THRESHOLD) and (unique_classes_count > self.CHAOS_THRESHOLD)

        if is_ood:
            return "UNKNOWN (OOD)", closest_distance, unique_classes_count, True

        # Get majority vote from strings
        values, counts = np.unique(k_labels, return_counts=True)
        predicted_class_name = values[np.argmax(counts)]

        return str(predicted_class_name).upper(), closest_distance, unique_classes_count, False


# =====================================================================
# 2. The Webcam User Interface
# =====================================================================
class WebcamApp:
    def __init__(self):
        """
        Initializes the webcam application and the OOD detector.
        Sets up the video capture using the default camera.
        """
        self.detector = OODCameraDetector()
        
        # Open the default camera (0 is usually the built-in webcam)
        self.cap = cv2.VideoCapture(0)
        
        # Check if camera opened successfully
        if not self.cap.isOpened():
            raise IOError("Cannot open webcam. Check your camera permissions.")
            
        # Set camera resolution (optional, helps with standardizing view)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def run(self):
        """
        Starts the main loop to capture webcam frames, run them through the OOD detector,
        and display the results with a UI overlay.
        """
        print("🎥 Starting Camera... Press 'q' to quit.")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Run the prediction
            # Note: This will block the frame until DINOv3 finishes computing
            label, dist, chaos, is_ood = self.detector.predict(rgb_frame)
            
            # --- UI Overlay ---
            # Choose color: Red for OOD, Green for Known
            color = (0, 0, 255) if is_ood else (0, 255, 0) # BGR format
            
            # Draw Main Label
            cv2.putText(frame, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            
            stats_text = f"Dist (T): {dist:.3f} | Chaos (C): {chaos}/7"
            cv2.putText(frame, stats_text, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, stats_text, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1) # Text outline
            
            cv2.imshow('Bachelor Thesis: OOD Detection Demo', frame)
            
            # Press 'q' on the keyboard to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        self.cleanup()

    def cleanup(self):
        """
        Releases the webcam resources and closes the application windows.
        """
        self.cap.release()
        cv2.destroyAllWindows()
        print("🛑 Camera closed.")

# =====================================================================
# 3. Execution
# =====================================================================
if __name__ == "__main__":
    app = WebcamApp()
    app.run()