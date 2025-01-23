import os
import torch
import numpy as np
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
import lightning as L 
import torch.utils.data as data

def read_utt2spk(file_path):
    utt2spk = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                raise ValueError(f"Line '{line.strip()}' is not properly formatted.")
            audio_name, speaker_id = parts
            utt2spk.append((audio_name, speaker_id))
    return utt2spk

def padding(audio, sr, target_duration):
    current_length = len(audio)
    target_length = target_duration * sr
    # 如果当前音频长度小于目标长度，进行填充
    if current_length < target_length:
        # 创建填充用的静音数组
        padded_audio = np.pad(audio, (0, target_length - current_length), mode='constant')
        return padded_audio
    else:
        # 如果音频长度大于目标长度，进行裁剪
        padded_audio = audio[:target_length]
        return padded_audio

class KeSpeechDataset(Dataset):
    def __init__(self, meta_dir, audio_dir, subset, target_duration=10, unique_labels=None):
        """
        Args:
            meta_dir (str): 元数据文件所在目录
            audio_dir (str): 音频文件所在目录
            subset (str): 数据子集名称，例如 "dev_phase1", "dev_phase2", "eval_combined/enroll_phase1-phase2"
            target_duration (int, optional): 目标音频时长（秒）。默认为 10 秒。
            unique_labels (dict, optional): 标签映射字典。默认为 None。
        """
        self.subset = subset
        self.audio_dir = audio_dir
        self.target_duration = target_duration
        meta_file_path = os.path.join(meta_dir, subset, "utt2spk")
        self.meta_subset = read_utt2spk(meta_file_path)

        if unique_labels is None:
            raise ValueError("unique_labels must be provided")
        self.unique_labels = unique_labels

        # 解析 subset 名称以确定包含的 phases
        if '-' in subset.split('/')[-1]:
            # 例如 "enroll_phase1-phase2"
            phase_part = subset.split('/')[-1].split('_')[-1]
            self.phases = phase_part.split('-')  # ['phase1', 'phase2']
        else:
            # 例如 "phase1" 或 "phase2"
            phase_part = subset.split('/')[-1].split('_')[-1]
            self.phases = [phase_part]  # ['phase1']

    def __len__(self):
        return len(self.meta_subset)

    def __getitem__(self, i):
        """
        返回:
            wav (torch.Tensor): 填充或裁剪后的音频数据
            label (torch.Tensor): 说话人类别索引
        """
        row_i = self.meta_subset[i]
        audio_name = row_i[0]
        speaker_id = row_i[1]

        # 根据 subset 中包含的 phases 查找音频文件
        wav_path = None
        for phase in self.phases:
            potential_path = os.path.join(self.audio_dir, speaker_id, phase, f"{audio_name}.wav")
            if os.path.exists(potential_path):
                wav_path = potential_path
                break  # 找到第一个存在的路径后跳出循环

        if wav_path is None:
            raise FileNotFoundError(f"Audio file for {audio_name} not found in phases {self.phases} under speaker {speaker_id}.")

        # 读取和处理音频
        wav, sr = sf.read(wav_path)
        wav = padding(wav, sr, self.target_duration)
        wav = torch.from_numpy(wav).float()

        # 编码说话人标签为整数
        label = self.unique_labels['speaker'].index(speaker_id)
        label = torch.tensor(label, dtype=torch.long)

        return wav, label

class KeSpeechDataModule(L.LightningDataModule):
    def __init__(self, meta_dir: str, audio_dir: str, batch_size: int = 16, num_workers: int = 0, pin_memory: bool = False, predict_subset=""):
        super().__init__()
        self.meta_dir = meta_dir
        self.audio_dir = audio_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.predict_subset = predict_subset
        self.unique_labels = None

    def setup(self, stage: str):
        if stage in ("fit", "validate", "test", "predict"):
            # 收集所有涉及的子集名称
            all_subsets = []
            if stage == "fit":
                all_subsets = ["dev_phase1"]
            elif stage == "validate":
                all_subsets = ["eval_combined/enroll_phase1-phase2"]
            elif stage == "test":
                all_subsets = ["eval_combined/test_phase1-phase2"]
            elif stage == "predict":
                all_subsets = [self.predict_subset]

            # 收集所有说话人 ID
            all_speakers = set()
            for subset in all_subsets:
                meta_file_path = os.path.join(self.meta_dir, subset, "utt2spk")
                utt2spk = read_utt2spk(meta_file_path)
                all_speakers.update([speaker_id for _, speaker_id in utt2spk])

            self.unique_labels = {'speaker': sorted(list(all_speakers))}

            # 根据 stage 初始化数据集
            if stage == "fit":
                full_train_set = KeSpeechDataset(
                    meta_dir=self.meta_dir,
                    audio_dir=self.audio_dir,
                    subset="dev_phase1",
                    unique_labels=self.unique_labels
                )
                train_set_size = int(len(full_train_set) * 0.8)
                valid_set_size = len(full_train_set) - train_set_size

                # 设置随机种子以确保可重复性
                generator = torch.Generator().manual_seed(42)

                # 拆分训练集为训练集和验证集
                self.train_set, self.valid_set = data.random_split(
                    full_train_set,
                    [train_set_size, valid_set_size],
                    generator=generator
                )
                
            if stage == "validate":
                self.valid_set = KeSpeechDataset(
                    meta_dir=self.meta_dir,
                    audio_dir=self.audio_dir,
                    subset="eval_combined/enroll_phase1-phase2",
                    unique_labels=self.unique_labels
                )
            if stage == "test":
                self.test_set = KeSpeechDataset(
                    meta_dir=self.meta_dir,
                    audio_dir=self.audio_dir,
                    subset="eval_combined/test_phase1-phase2",
                    unique_labels=self.unique_labels
                )
            if stage == "predict":
                self.predict_set = KeSpeechDataset(
                    meta_dir=self.meta_dir,
                    audio_dir=self.audio_dir,
                    subset=self.predict_subset,
                    unique_labels=self.unique_labels
                )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory
        )
