import os
import sys
import random
import numpy as np
import soundfile as sf
import librosa
from librosa import effects as librosa_effects
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import json
import argparse
from datetime import datetime
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from utils.paths import get_raw_dir, get_synthetic_dir

class TITMLMixGenerator2Spk:

    def __init__(self, titml_dir, output_dir, target_sr=16000, seed=42):
        self.titml_dir = Path(titml_dir)
        self.output_dir = Path(output_dir)
        self.target_sr = target_sr
        random.seed(seed)
        np.random.seed(seed)
        print('=' * 60)
        print('TITML-IDN 2-Speaker Speech Separation Dataset Generator')
        print('=' * 60)
        self.speakers = self._collect_speakers()
        self._print_statistics()

    def _collect_speakers(self):
        speakers = defaultdict(list)
        speech_dir = self.titml_dir / 'Speech'
        if not speech_dir.exists():
            raise FileNotFoundError(f'Speech directory not found: {speech_dir}')
        for speaker_dir in sorted(speech_dir.iterdir()):
            if not speaker_dir.is_dir():
                continue
            speaker_id = speaker_dir.name
            audio_files = sorted(speaker_dir.glob('*.wav'))
            if len(audio_files) > 0:
                speakers[speaker_id] = audio_files
        return speakers

    def _print_statistics(self):
        print(f'\nDataset Statistics:')
        print(f'  Total speakers: {len(self.speakers)}')
        male_speakers = [s for s in self.speakers.keys() if s.startswith('m')]
        female_speakers = [s for s in self.speakers.keys() if s.startswith('f')]
        print(f'  Male speakers: {len(male_speakers)}')
        print(f'  Female speakers: {len(female_speakers)}')
        total_utterances = sum((len(files) for files in self.speakers.values()))
        avg_utterances = total_utterances / len(self.speakers)
        print(f'  Total utterances: {total_utterances}')
        print(f'  Avg utterances per speaker: {avg_utterances:.1f}')
        print(f'\nSample speakers:')
        for speaker_id, files in list(self.speakers.items())[:5]:
            print(f'  {speaker_id}: {len(files)} files')

    def load_audio(self, file_path, target_duration=None):
        try:
            audio, sr = sf.read(file_path)
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            if sr != self.target_sr:
                audio = librosa.resample(y=audio, orig_sr=sr, target_sr=self.target_sr)
            audio, _ = librosa_effects.trim(y=audio, top_db=30)
            if len(audio) < self.target_sr:
                return None
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio / max_val * 0.9
            if target_duration:
                target_len = int(target_duration * self.target_sr)
                if len(audio) > target_len:
                    start = random.randint(0, len(audio) - target_len)
                    audio = audio[start:start + target_len]
                elif len(audio) < target_len:
                    audio = np.pad(audio, (0, target_len - len(audio)))
            return audio
        except Exception as e:
            print(f'Error loading {file_path}: {e}')
            return None

    def mix_two_sources(self, audio1, audio2, snr_range=(-5, 5)):
        max_offset = int(1.0 * self.target_sr)
        offset2 = random.randint(0, max_offset)
        audio2_shifted = np.pad(audio2, (offset2, 0))
        use_min_length = random.choice([True, False])
        lengths = [len(audio1), len(audio2_shifted)]
        if use_min_length:
            target_len = min(lengths)
        else:
            target_len = max(lengths)

        def fit_audio(a, length):
            if len(a) > length:
                return a[:length]
            elif len(a) < length:
                return np.pad(a, (0, length - len(a)))
            return a
        audio1 = fit_audio(audio1, target_len)
        audio2_shifted = fit_audio(audio2_shifted, target_len)
        snr_db = random.uniform(*snr_range)
        audio1_power = np.mean(audio1 ** 2) + 1e-10
        audio2_power = np.mean(audio2_shifted ** 2) + 1e-10
        scale = np.sqrt(audio1_power / (audio2_power * 10 ** (snr_db / 10)))
        audio2_scaled = audio2_shifted * scale
        mixture = audio1 + audio2_scaled
        max_val = np.max(np.abs(mixture))
        if max_val > 1.0:
            scale_factor = 0.9 / max_val
            mixture = mixture * scale_factor
            audio1 = audio1 * scale_factor
            audio2_scaled = audio2_scaled * scale_factor
        return (mixture, audio1, audio2_scaled)

    def split_utterances(self, train_ratio=0.8, dev_ratio=0.1, test_ratio=0.1):
        train_utts = defaultdict(list)
        dev_utts = defaultdict(list)
        test_utts = defaultdict(list)
        for speaker_id, files in self.speakers.items():
            shuffled = list(files)
            random.shuffle(shuffled)
            n = len(shuffled)
            n_train = int(n * train_ratio)
            n_dev = int(n * dev_ratio)
            train_utts[speaker_id] = shuffled[:n_train]
            dev_utts[speaker_id] = shuffled[n_train:n_train + n_dev]
            test_utts[speaker_id] = shuffled[n_train + n_dev:]
        total_train = sum((len(v) for v in train_utts.values()))
        total_dev = sum((len(v) for v in dev_utts.values()))
        total_test = sum((len(v) for v in test_utts.values()))
        print(f'\nUtterance Split (all {len(self.speakers)} speakers in every split):')
        print(f'  Train: {total_train} utterances across {len(train_utts)} speakers')
        print(f'  Dev:   {total_dev} utterances across {len(dev_utts)} speakers')
        print(f'  Test:  {total_test} utterances across {len(test_utts)} speakers')
        return (train_utts, dev_utts, test_utts)

    def generate_mixtures_from_utterances(self, utterances_by_speaker, split_name, num_mixtures, target_duration=5.0, gender_balance=True):
        speaker_list = [sid for sid, utts in utterances_by_speaker.items() if len(utts) > 0]
        if len(speaker_list) < 2:
            raise ValueError(f'Need at least 2 speakers with utterances, got {len(speaker_list)}')
        output_split = self.output_dir / split_name
        for subdir in ('mix', 's1', 's2'):
            (output_split / subdir).mkdir(parents=True, exist_ok=True)
        print(f"\nGenerating {num_mixtures} 2-speaker mixtures for '{split_name}' split...")
        print(f'   Using {len(speaker_list)} speakers')
        print(f'   Target duration: {target_duration}s')
        males = [s for s in speaker_list if s.startswith('m')]
        females = [s for s in speaker_list if s.startswith('f')]
        success_count = 0
        attempt = 0
        max_attempts = num_mixtures * 5
        pbar = tqdm(total=num_mixtures, desc=f'{split_name}')
        while success_count < num_mixtures and attempt < max_attempts:
            attempt += 1
            if gender_balance and males and females:
                chosen = random.choice([(random.choice(males), random.choice(females)), (random.choice(males), random.choice(females)), tuple(random.sample(speaker_list, 2))][:-1] + [tuple(random.sample(speaker_list, 2))])
            else:
                chosen = tuple(random.sample(speaker_list, 2))
            spk1, spk2 = chosen
            audio1 = self.load_audio(random.choice(utterances_by_speaker[spk1]), target_duration=target_duration)
            audio2 = self.load_audio(random.choice(utterances_by_speaker[spk2]), target_duration=target_duration)
            if audio1 is None or audio2 is None:
                continue
            try:
                mixture, src1, src2 = self.mix_two_sources(audio1, audio2)
            except Exception as e:
                print(f'Error mixing: {e}')
                continue
            filename = f'{split_name}_{success_count:05d}.wav'
            sf.write(output_split / 'mix' / filename, mixture, self.target_sr)
            sf.write(output_split / 's1' / filename, src1, self.target_sr)
            sf.write(output_split / 's2' / filename, src2, self.target_sr)
            success_count += 1
            pbar.update(1)
        pbar.close()
        if success_count < num_mixtures:
            print(f'Warning: Generated {success_count}/{num_mixtures} mixtures')
        else:
            print(f'Generated {success_count} mixtures')
        self._save_metadata(output_split, success_count, speaker_list, target_duration)
        return success_count

    def _save_metadata(self, output_path, num_mixtures, speakers, target_duration):
        metadata = {'num_mixtures': num_mixtures, 'sample_rate': self.target_sr, 'num_sources': 2, 'num_speakers_total': len(speakers), 'speakers': speakers, 'source': 'TITML-IDN', 'duration_seconds': target_duration, 'total_duration_hours': num_mixtures * target_duration / 3600}
        with open(output_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

    def generate_dataset_info(self, train_count, dev_count, test_count, target_duration):
        info = {'dataset_name': 'TITML-2spk', 'description': '2-speaker speech separation dataset from TITML-IDN', 'created_at': datetime.now().isoformat(), 'source_dataset': 'TITML-IDN (Indonesian Speech Dataset)', 'configuration': {'sample_rate': self.target_sr, 'clip_duration_seconds': target_duration, 'num_speakers': 2, 'snr_range_db': [-5, 5], 'time_offset_max_seconds': 1.0}, 'splits': {'train': {'num_mixtures': train_count, 'duration_hours': train_count * target_duration / 3600}, 'dev': {'num_mixtures': dev_count, 'duration_hours': dev_count * target_duration / 3600}, 'test': {'num_mixtures': test_count, 'duration_hours': test_count * target_duration / 3600}}, 'total': {'num_mixtures': train_count + dev_count + test_count, 'duration_hours': (train_count + dev_count + test_count) * target_duration / 3600}, 'directory_structure': {'train': {'mix': '2-speaker mixtures', 's1': 'Speaker 1 (reference)', 's2': 'Speaker 2 (SNR-scaled, time-offset)'}, 'dev': 'Same structure as train', 'test': 'Same structure as train'}, 'speaker_info': {'total_speakers': len(self.speakers), 'male_speakers': len([s for s in self.speakers.keys() if s.startswith('m')]), 'female_speakers': len([s for s in self.speakers.keys() if s.startswith('f')])}}
        info_path = self.output_dir / 'dataset_info.json'
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
        print(f'\nDataset info saved to: {info_path}')
        return info

def main():
    parser = argparse.ArgumentParser(description='Generate TITML-2spk dataset for speech separation')
    parser.add_argument('--titml-dir', type=str, default=None, help='Path to TITML-IDN raw dataset directory. Falls back to $TSS_RAW_DIR or <project_root>/dataset/raw/TTML-IDN.')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory for generated dataset. Falls back to $TSS_SYNTHETIC_DIR/TITML-2spk or <project_root>/dataset/synthetic/TITML-2spk.')
    parser.add_argument('--target-duration', type=float, default=5.0, help='Duration of each clip in seconds (default: 5.0)')
    parser.add_argument('--target-hours', type=float, default=50.0, help='Target total dataset size in hours (default: 50.0)')
    parser.add_argument('--train-ratio', type=float, default=0.8, help='Ratio of training data (default: 0.8)')
    parser.add_argument('--dev-ratio', type=float, default=0.1, help='Ratio of dev data (default: 0.1)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--only-splits', type=str, nargs='+', choices=['train', 'dev', 'test'], default=None, help='Only generate specified splits (e.g. --only-splits dev test)')
    args = parser.parse_args()
    args.titml_dir = str(get_raw_dir(args.titml_dir))
    args.output_dir = str(get_synthetic_dir('TITML-2spk', args.output_dir))
    total_mixtures = int(args.target_hours * 3600 / args.target_duration)
    train_mixtures = int(total_mixtures * args.train_ratio)
    dev_mixtures = int(total_mixtures * args.dev_ratio)
    test_mixtures = total_mixtures - train_mixtures - dev_mixtures
    min_mixtures = 100
    if dev_mixtures < min_mixtures:
        deficit = min_mixtures - dev_mixtures
        if train_mixtures > min_mixtures * 10:
            train_mixtures -= deficit
            dev_mixtures += deficit
    if test_mixtures < min_mixtures:
        deficit = min_mixtures - test_mixtures
        if train_mixtures > min_mixtures * 10:
            train_mixtures -= deficit
            test_mixtures += deficit
    print('\n' + '=' * 60)
    print('Dataset Generation Configuration')
    print('=' * 60)
    print(f'TITML source: {args.titml_dir}')
    print(f'Output directory: {args.output_dir}')
    print(f'Target duration: {args.target_duration}s')
    print(f'Target total hours: {args.target_hours}h')
    print(f'Total mixtures: {total_mixtures}')
    print(f'  Train: {train_mixtures} ({args.train_ratio * 100:.0f}%)')
    print(f'  Dev:   {dev_mixtures} ({args.dev_ratio * 100:.0f}%)')
    print(f'  Test:  {test_mixtures} ({(1 - args.train_ratio - args.dev_ratio) * 100:.0f}%)')
    print('=' * 60)
    generator = TITMLMixGenerator2Spk(titml_dir=args.titml_dir, output_dir=args.output_dir, target_sr=16000, seed=args.seed)
    only_splits = set(args.only_splits) if args.only_splits else {'train', 'dev', 'test'}
    train_utts, dev_utts, test_utts = generator.split_utterances(train_ratio=args.train_ratio, dev_ratio=args.dev_ratio, test_ratio=1 - args.train_ratio - args.dev_ratio)
    train_count = generator.generate_mixtures_from_utterances(utterances_by_speaker=train_utts, split_name='train', num_mixtures=train_mixtures, target_duration=args.target_duration, gender_balance=True) if 'train' in only_splits else 0
    dev_count = generator.generate_mixtures_from_utterances(utterances_by_speaker=dev_utts, split_name='dev', num_mixtures=dev_mixtures, target_duration=args.target_duration, gender_balance=True) if 'dev' in only_splits else 0
    test_count = generator.generate_mixtures_from_utterances(utterances_by_speaker=test_utts, split_name='test', num_mixtures=test_mixtures, target_duration=args.target_duration, gender_balance=True) if 'test' in only_splits else 0
    generator.generate_dataset_info(train_count, dev_count, test_count, args.target_duration)
    print('\n' + '=' * 60)
    print('Dataset Generation Complete!')
    print('=' * 60)
    print(f'\nDataset location: {args.output_dir}')
    print('\nDataset structure:')
    print(f'  TITML-2spk/')
    if 'train' in only_splits:
        print(f'  ├── train/ ({train_count} mixtures, ~{train_count * args.target_duration / 3600:.1f} hours)')
    else:
        print('  Train mixtures not generated; training uses on-the-fly mixing.')
    print(f'  ├── dev/   ({dev_count} mixtures, ~{dev_count * args.target_duration / 3600:.1f} hours)')
    print(f'  ├── test/  ({test_count} mixtures, ~{test_count * args.target_duration / 3600:.1f} hours)')
    print(f'  └── dataset_info.json')
    print(f'\nStatic WAV mixtures: {train_count + dev_count + test_count}')
    print(f'Static WAV duration: ~{(train_count + dev_count + test_count) * args.target_duration / 3600:.1f} hours')
    if 'train' not in only_splits:
        print(f'Dynamic train per epoch: ~{train_mixtures * args.target_duration / 3600:.1f} hours ({train_mixtures} mixtures; generated during training)')
    print(f'Total dataset target: ~{args.target_hours:.1f} hours (train + dev + test)')
if __name__ == '__main__':
    main()
