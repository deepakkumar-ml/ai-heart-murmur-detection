import os
import glob
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model

# --- 1. CONFIGURATION ---
# Adjust these paths if your folders are named differently
MODEL_FOLDER = "Model files"
AUDIO_FOLDER = "Test files"

# --- 2. THE VERIFIED CLASS MAP (0, 1, 2) ---
# Based on your notebook: 0=Artifact, 1=Murmur, 2=Normal
CLASSES = ['Artifact (Noise)', 'Murmur (Heart Defect)', 'Normal (Healthy)']

# --- 3. EXACT TRAINING PARAMETERS ---
DURATION = 10   # Seconds (Fixed Window)
SAMPLE_RATE = 22050
FEATURES = 52   # MFCC count

def preprocess_audio_exact(file_path):
    """
    Replicates your 'load_file_data' function exactly:
    1. Load 10s audio.
    2. Pad with silence if short (Critical Step!).
    3. Extract 52 MFCCs and Average them.
    """
    try:
        # 1. Load Audio
        # We force the duration to 10s. If file is longer, it cuts it.
        X, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
        
        # 2. Pad if shorter than 10s
        # Calculate expected number of samples (22050 * 10 = 220,500)
        input_length = SAMPLE_RATE * DURATION
        dur = librosa.get_duration(y=X, sr=sr)
        
        if round(dur) < DURATION:
            # This mimics your 'librosa.util.fix_length' logic
            X = librosa.util.fix_length(data=X, size=input_length)
            
        # 3. Feature Extraction (Mean of 52 MFCCs)
        # Transpose (.T) ensures we average across time, keeping 52 features
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sr, n_mfcc=FEATURES).T, axis=0)
        
        # 4. Reshape for LSTM (1 sample, 52 features, 1 channel)
        final_input = np.reshape(mfccs, (1, FEATURES, 1))
        
        return final_input
        
    except Exception as e:
        print(f"❌ Error processing audio: {e}")
        return None

def run_diagnosis():
    print("\n" + "="*60)
    print("🫀  HEART MURMUR DETECTION: PRODUCTION ENGINE")
    print("="*60 + "\n")

    # A. Find Model
    # Looks for .keras OR .h5 files
    files = glob.glob(os.path.join(MODEL_FOLDER, "*.keras")) + \
            glob.glob(os.path.join(MODEL_FOLDER, "*.h5"))
    
    if not files:
        print(f"❌ Error: No model found in '{MODEL_FOLDER}'")
        return
    
    model_path = files[0]
    print(f"⚙️  Loading Brain: '{os.path.basename(model_path)}'...")
    model = load_model(model_path)
    print("✅ Model Loaded Successfully.\n")

    # B. Find Audio
    audio_files = glob.glob(os.path.join(AUDIO_FOLDER, "*.wav"))
    if not audio_files:
        print(f"❌ Error: No .wav files found in '{AUDIO_FOLDER}'")
        return
    
    # Pick the first file automatically for the test
    target_audio = audio_files[0]
    print(f"🎧 Analyzing Patient Data: '{os.path.basename(target_audio)}'")

    # C. Predict
    input_data = preprocess_audio_exact(target_audio)
    
    if input_data is not None:
        # Get probabilities [Artifact%, Murmur%, Normal%]
        prediction = model.predict(input_data, verbose=0)
        probs = prediction[0] # Extract the array from the batch
        
        # Who is the winner?
        winner_index = np.argmax(probs)
        winner_label = CLASSES[winner_index]
        confidence = probs[winner_index]

        # D. Report
        print("\n" + "-"*40)
        print("🩺  FINAL DIAGNOSIS REPORT")
        print("-" * 40)
        
        # Dynamic Icons
        if winner_index == 1:   # Murmur
            icon = "🔴"
            status = "Refer to Cardiologist"
        elif winner_index == 2: # Normal
            icon = "🟢"
            status = "No Action Needed"
        else:                   # Artifact
            icon = "⚠️ "
            status = "Retake Audio (Too Noisy)"
            
        print(f"{icon} RESULT: {winner_label}")
        print(f"📊 Confidence: {confidence*100:.2f}%")
        print(f"📝 Action: {status}")
        
        print("\n--- Probability Distribution ---")
        print(f"   [0] Artifact: {probs[0]*100:.2f}%")
        print(f"   [1] Murmur:   {probs[1]*100:.2f}%")
        print(f"   [2] Normal:   {probs[2]*100:.2f}%")
        print("-" * 40)

if __name__ == "__main__":
    run_diagnosis()
