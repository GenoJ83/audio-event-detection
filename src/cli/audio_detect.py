"""Command-line audio event classification tool."""
import argparse
from src.data.audio_generator import generate_synthetic_audio
from src.inference.stream_predictor import AudioEventDetector

def main():
    parser = argparse.ArgumentParser(description="Environmental Sound Detector")
    parser.add_argument("--class-sim", type=int, default=0, help="Simulate sound class (0-4)")
    args = parser.parse_args()

    waveform, sr, ground_truth = generate_synthetic_audio(class_idx=args.class_sim)
    detector = AudioEventDetector()
    result = detector.detect_event(waveform)

    print(f"Detected Acoustic Event: {result['event']} (Confidence: {result['confidence']:.2%})")

if __name__ == "__main__":
    main()
