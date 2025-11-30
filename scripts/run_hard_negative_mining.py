import os
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import config
import preprocessing
import data_loader
import model as model_module

def mine_hard_negatives(model, long_audio_files, confidence_range=(0.5, 0.85), max_samples=300):
    """
    Extract samples where model predicts primate but with medium confidence
    """
    
    hard_negatives = []
    min_conf, max_conf = confidence_range
    
    for audio_file in long_audio_files:
        if len(hard_negatives) >= max_samples:
            break
        
        print(f"Processing: {os.path.basename(audio_file)}")
        
        audio, sr = librosa.load(audio_file, sr=config.SAMPLE_RATE)
        windows, times = preprocessing.extract_sliding_windows(audio, sr)
        
        X_windows = []
        for window in windows:
            img = preprocessing.preprocess_audio(window, sr)
            img_norm = preprocessing.preprocess_for_model(img)
            X_windows.append(img_norm)
        
        X_windows = np.array(X_windows)
        predictions = model.predict(X_windows, batch_size=config.BATCH_SIZE, verbose=0)
        
        for i, (pred, (start_time, end_time)) in enumerate(zip(predictions, times)):
            predicted_class = np.argmax(pred)
            confidence = pred[predicted_class]
            
            if predicted_class != len(config.CLASS_NAMES) - 1:
                if min_conf <= confidence <= max_conf:
                    hard_negatives.append({
                        'audio': windows[i],
                        'file': os.path.basename(audio_file),
                        'start_time': start_time,
                        'end_time': end_time,
                        'confidence': confidence,
                        'predicted_species': config.CLASS_NAMES[predicted_class]
                    })
                    
                    if len(hard_negatives) >= max_samples:
                        break
        
        print(f"  Extracted: {len([h for h in hard_negatives if h['file'] == os.path.basename(audio_file)])} samples")
    
    return hard_negatives


def save_hard_negatives(hard_negatives, output_dir):
    """
    Save hard negative samples as audio files
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    for sample in hard_negatives:
        filename = (f"{os.path.splitext(sample['file'])[0]}_"
                   f"{sample['start_time']:.0f}s_"
                   f"conf{sample['confidence']:.3f}_"
                   f"{sample['predicted_species']}.wav")
        
        save_path = os.path.join(output_dir, filename)
        sf.write(save_path, sample['audio'], config.SAMPLE_RATE)
    
    metadata = []
    for sample in hard_negatives:
        metadata.append({
            'filename': f"{os.path.splitext(sample['file'])[0]}_{sample['start_time']:.0f}s_conf{sample['confidence']:.3f}_{sample['predicted_species']}.wav",
            'source_file': sample['file'],
            'start_time': sample['start_time'],
            'end_time': sample['end_time'],
            'confidence': sample['confidence'],
            'predicted_species': sample['predicted_species']
        })
    
    metadata_df = pd.DataFrame(metadata)
    metadata_path = os.path.join(output_dir, 'hard_negatives_metadata.csv')
    metadata_df.to_csv(metadata_path, index=False)
    
    species_counts = {}
    for sample in hard_negatives:
        species = sample['predicted_species']
        species_counts[species] = species_counts.get(species, 0) + 1
    
    print(f"\nBy predicted species:")
    for species, count in sorted(species_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {species}: {count}")
    
    return metadata_path


if __name__ == "__main__":
    
    print("Step 1: Loading model")
    model_path = os.path.join(config.MODEL_SAVE_DIR, 'best_model.h5')
    trained_model = model_module.load_trained_model(model_path)
    
    print("\nStep 2: Getting long audio files")
    long_audio_files = data_loader.get_long_audio_files()
    print(f"Found {len(long_audio_files)} files")
    
    print("\nStep 3: Mining hard negatives")
    hard_negatives = mine_hard_negatives(
        model=trained_model,
        long_audio_files=long_audio_files[:10],
        confidence_range=(0.5, 0.85),
        max_samples=300
    )
    
    print(f"\nExtracted {len(hard_negatives)} total samples")
    
    print("\nStep 4: Saving samples")
    output_dir = os.path.join(config.AUDIO_ROOT, 'hard_negative_candidates')
    metadata_path = save_hard_negatives(hard_negatives, output_dir)
    
    print(f"\nSaved to: {output_dir}")
    print(f"Metadata: {metadata_path}")
    
    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("1. Go to Google Drive: chimp-audio/audio/hard_negative_candidates/")
    print("2. Listen to these audio files")
    print("3. DELETE files that are actual primate calls")
    print("4. KEEP files that are bird calls or other sounds")
    print("5. Create new folder: chimp-audio/audio/verified_hard_negatives/")
    print("6. MOVE the kept files to verified_hard_negatives/")
    print("7. Edit config.py line 28-32:")
    print("   BACKGROUND_FOLDERS = [")
    print("       'background noise Clips 5sec',")
    print("       'wrong classified',")
    print("       'verified_hard_negatives'  # ADD THIS LINE")
    print("   ]")
    print("8. Run: import train; train.run_complete_training_pipeline()")
