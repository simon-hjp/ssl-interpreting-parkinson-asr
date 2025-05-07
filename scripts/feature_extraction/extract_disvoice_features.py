import os
import re
import sys
import glob
import joblib
import librosa
import argparse
import numpy as np
from tqdm import tqdm
from distutils.util import strtobool

sys.path.insert(1, os.path.abspath('tools/DisVoice/'))
from disvoice.glottal import Glottal
from disvoice.prosody import Prosody
from disvoice.phonation import Phonation
from disvoice.articulation import Articulation

import warnings
warnings.filterwarnings("ignore")

def extract_prosody():
    prosody = Prosody()

    output_dir = os.path.join(os.path.abspath(args.output_dir), 'disvoice', 'prosody')
    os.makedirs(output_dir, exist_ok=True)

    wav_files = glob.glob(os.path.join(os.path.abspath(args.wav_dir), '*.wav'))

    for wav_file in tqdm(wav_files):
        output_path = os.path.join(output_dir, os.path.basename(wav_file)).replace('.wav', '.npz')
        prosody_features = prosody.extract_features_file(wav_file, static=True, plots=False, fmt='npy')
        prosody_features = np.expand_dims(prosody_features, 0) if args.static else prosody_features

        # -- data curation
        prosody_features[np.isnan(prosody_features)] = 0
        # print(f'Prosody: {prosody_features.shape}')
        np.savez_compressed(output_path, data=prosody_features)

def extract_articulation():
    articulation = Articulation()

    output_dir = os.path.join(os.path.abspath(args.output_dir), 'disvoice', 'articulation')
    os.makedirs(output_dir, exist_ok=True)

    wav_files = glob.glob(os.path.join(os.path.abspath(args.wav_dir), '*.wav'))

    for wav_file in tqdm(wav_files):
        output_path = os.path.join(output_dir, os.path.basename(wav_file)).replace('.wav', '.npz')
        articulation_features = articulation.extract_features_file(wav_file, static=True, plots=False, fmt='npy')

        # -- data curation
        articulation_features[np.isnan(articulation_features)] = 0
        # print(f'Articulation: {articulation_features.shape}')
        np.savez_compressed(output_path, data=articulation_features)

def extract_phonation():
    phonation = Phonation()

    output_dir = os.path.join(os.path.abspath(args.output_dir), 'disvoice', 'phonation')
    os.makedirs(output_dir, exist_ok=True)

    wav_files = glob.glob(os.path.join(os.path.abspath(args.wav_dir), '*.wav'))

    for wav_file in tqdm(wav_files):
        output_path = os.path.join(output_dir, os.path.basename(wav_file)).replace('.wav', '.npz')
        phonation_features = phonation.extract_features_file(wav_file, static=True, plots=False, fmt='npy')

        # -- data curation
        phonation_features[np.isnan(phonation_features)] = 0
        # print(f'Phonation: {phonation_features.shape}')
        np.savez_compressed(output_path, data=phonation_features)

def extract_glottal():
    glottal = Glottal()

    output_dir = os.path.join(os.path.abspath(args.output_dir), 'disvoice', 'glottal')
    os.makedirs(output_dir, exist_ok=True)

    wav_files = glob.glob(os.path.join(os.path.abspath(args.wav_dir), '*.wav'))

    for wav_file in tqdm(wav_files):
        output_path = os.path.join(output_dir, os.path.basename(wav_file)).replace('.wav', '.npz')

        glottal_features = glottal.extract_features_file(wav_file, static=True, plots=False, fmt='npy')

        # -- data curation
        glottal_features[np.isnan(glottal_features)] = 0
        # print(f'Glottal: {glottal_features.shape}')
        np.savez_compressed(output_path, data=glottal_features)

if __name__ == "__main__":

    # -- command line arguments
    parser = argparse.ArgumentParser(description='DisVoice-based Feature Extraction', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--wav-dir', type=str, default='./data/gita/norm_audios/')
    parser.add_argument('--output-dir', required=True, type=str)
    parser.add_argument('--static', action='store_true', help='Use static feature representation') 
    args = parser.parse_args()

    # -- speech feature extraction
    print('Extracting disvoice features')
    extract_prosody()
    extract_articulation()
    extract_phonation()
    extract_glottal()
