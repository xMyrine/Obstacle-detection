import cv2
import gradio as gr
import numpy as np

from demo import OODCameraDetector


detector = OODCameraDetector()


def process_frame(image_frame):
    """Run OOD prediction on a single RGB frame and overlay results."""
    if image_frame is None:
        return None

    image_frame = np.copy(image_frame)

    label, dist, chaos, is_ood = detector.predict(image_frame)

    color = (255, 0, 0) if is_ood else (0, 255, 0)

    h, w = image_frame.shape[:2]

    cv2.putText(
        image_frame,
        label,
        (20, h - 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        color,
        3,
        cv2.LINE_AA,
    )

    stats_text = f"Dist: {dist:.3f} | Chaos: {chaos}"
    cv2.putText(
        image_frame,
        stats_text,
        (20, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return image_frame


interface = gr.Interface(
    fn=process_frame,
    inputs=gr.Image(sources=["webcam"], streaming=True),
    outputs="image",
    live=True,
    title="Bachelor Thesis: Mobile OOD Detection Demo",
    description="Real-time Out-of-Distribution (OOD) detection stream for local Wi-Fi testing.",
)


if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=7860, ssl_keyfile="key.pem", ssl_certfile="cert.pem", ssl_verify=False)
