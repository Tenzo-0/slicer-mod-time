import numpy as np
from pydub import AudioSegment
import os

# This function is obtained from librosa.
def get_rms(
    y,
    *,
    frame_length=2048,
    hop_length=512,
    pad_mode="constant",
):
    padding = (int(frame_length // 2), int(frame_length // 2))
    y = np.pad(y, padding, mode=pad_mode)

    axis = -1
    out_strides = y.strides + (y.strides[axis],)
    x_shape_trimmed = list(y.shape)
    x_shape_trimmed[axis] -= frame_length - 1
    out_shape = tuple(x_shape_trimmed) + (frame_length,)
    xw = np.lib.stride_tricks.as_strided(
        y, shape=out_shape, strides=out_strides
    )
    if axis < 0:
        target_axis = axis - 1
    else:
        target_axis = axis + 1
    xw = np.moveaxis(xw, -1, target_axis)
    slices = [slice(None)] * xw.ndim
    slices[axis] = slice(0, None, hop_length)
    x = xw[tuple(slices)]
    power = np.mean(np.abs(x) ** 2, axis=-2, keepdims=True)
    return np.sqrt(power)


class Slicer:
    def __init__(self,
                 sr: int,
                 threshold: float = -40.,
                 min_length: int = 5000,
                 min_interval: int = 300,
                 hop_size: int = 20,
                 max_sil_kept: int = 5000):
        if not min_length >= min_interval >= hop_size:
            raise ValueError('The following condition must be satisfied: min_length >= min_interval >= hop_size')
        if not max_sil_kept >= hop_size:
            raise ValueError('The following condition must be satisfied: max_sil_kept >= hop_size')
        min_interval = sr * min_interval / 1000
        self.threshold = 10 ** (threshold / 20.)
        self.hop_size = round(sr * hop_size / 1000)
        self.win_size = min(round(min_interval), 4 * self.hop_size)
        self.min_length = round(sr * min_length / 1000 / self.hop_size)
        self.min_interval = round(min_interval / self.hop_size)
        self.max_sil_kept = round(sr * max_sil_kept / 1000 / self.hop_size)

    def _apply_slice(self, waveform, begin, end):
        # Calculate sample indices from frame indices
        start_sample = begin * self.hop_size
        if len(waveform.shape) > 1:
            end_sample = min(waveform.shape[1], end * self.hop_size)
            sliced = waveform[:, start_sample:end_sample]
        else:
            end_sample = min(waveform.shape[0], end * self.hop_size)
            sliced = waveform[start_sample:end_sample]
        return sliced, start_sample, end_sample

    def slice(self, waveform):
        if len(waveform.shape) > 1:
            samples = waveform.mean(axis=0)
        else:
            samples = waveform
        total_frames = (samples.shape[0] + self.hop_size - 1) // self.hop_size
        if total_frames <= self.min_length:
            # Return the entire waveform with its start and end sample indices.
            total_samples = waveform.shape[1] if len(waveform.shape) > 1 else waveform.shape[0]
            return [(waveform, 0, total_samples)]
        rms_list = get_rms(y=samples, frame_length=self.win_size, hop_length=self.hop_size).squeeze(0)
        sil_tags = []
        silence_start = None
        clip_start = 0
        for i, rms in enumerate(rms_list):
            if rms < self.threshold:
                if silence_start is None:
                    silence_start = i
                continue
            if silence_start is None:
                continue
            is_leading_silence = silence_start == 0 and i > self.max_sil_kept
            need_slice_middle = (i - silence_start >= self.min_interval) and (i - clip_start >= self.min_length)
            if not is_leading_silence and not need_slice_middle:
                silence_start = None
                continue
            if i - silence_start <= self.max_sil_kept:
                pos = rms_list[silence_start: i + 1].argmin() + silence_start
                if silence_start == 0:
                    sil_tags.append((0, pos))
                else:
                    sil_tags.append((pos, pos))
                clip_start = pos
            elif i - silence_start <= self.max_sil_kept * 2:
                pos = rms_list[i - self.max_sil_kept: silence_start + self.max_sil_kept + 1].argmin()
                pos += i - self.max_sil_kept
                pos_l = rms_list[silence_start: silence_start + self.max_sil_kept + 1].argmin() + silence_start
                pos_r = rms_list[i - self.max_sil_kept: i + 1].argmin() + i - self.max_sil_kept
                if silence_start == 0:
                    sil_tags.append((0, pos_r))
                    clip_start = pos_r
                else:
                    sil_tags.append((min(pos_l, pos), max(pos_r, pos)))
                    clip_start = max(pos_r, pos)
            else:
                pos_l = rms_list[silence_start: silence_start + self.max_sil_kept + 1].argmin() + silence_start
                pos_r = rms_list[i - self.max_sil_kept: i + 1].argmin() + i - self.max_sil_kept
                if silence_start == 0:
                    sil_tags.append((0, pos_r))
                else:
                    sil_tags.append((pos_l, pos_r))
                clip_start = pos_r
            silence_start = None
        # Deal with trailing silence.
        if silence_start is not None and (rms_list.shape[0] - silence_start >= self.min_interval):
            silence_end = min(rms_list.shape[0], silence_start + self.max_sil_kept)
            pos = rms_list[silence_start: silence_end + 1].argmin() + silence_start
            sil_tags.append((pos, rms_list.shape[0] + 1))
        # Apply slices and return tuple (chunk, start_sample, end_sample)
        if len(sil_tags) == 0:
            total_samples = waveform.shape[1] if len(waveform.shape) > 1 else waveform.shape[0]
            return [(waveform, 0, total_samples)]
        else:
            chunks = []
            # First chunk (if any) before first silence tag
            if sil_tags[0][0] > 0:
                chunk, start_sample, end_sample = self._apply_slice(waveform, 0, sil_tags[0][0])
                chunks.append((chunk, start_sample, end_sample))
            # Middle chunks between silence tags
            for i in range(len(sil_tags) - 1):
                chunk, start_sample, end_sample = self._apply_slice(waveform, sil_tags[i][1], sil_tags[i + 1][0])
                chunks.append((chunk, start_sample, end_sample))
            # Last chunk (if any) after the last silence tag
            if sil_tags[-1][1] < rms_list.shape[0]:
                chunk, start_sample, end_sample = self._apply_slice(waveform, sil_tags[-1][1], rms_list.shape[0])
                chunks.append((chunk, start_sample, end_sample))
            return chunks


def main():
    import os.path
    from argparse import ArgumentParser
    import librosa

    parser = ArgumentParser()
    parser.add_argument('audio', type=str, help='The audio to be sliced')
    parser.add_argument('--out', type=str, help='Output directory of the sliced audio clips')
    parser.add_argument('--db_thresh', type=float, required=False, default=-40,
                        help='The dB threshold for silence detection')
    parser.add_argument('--min_length', type=int, required=False, default=5000,
                        help='The minimum milliseconds required for each sliced audio clip')
    parser.add_argument('--min_interval', type=int, required=False, default=300,
                        help='The minimum milliseconds for a silence part to be sliced')
    parser.add_argument('--hop_size', type=int, required=False, default=10,
                        help='Frame length in milliseconds')
    parser.add_argument('--max_sil_kept', type=int, required=False, default=500,
                        help='The maximum silence length kept around the sliced clip, presented in milliseconds')
    args = parser.parse_args()
    out = args.out
    if out is None:
        out = os.path.dirname(os.path.abspath(args.audio))
    audio, sr = librosa.load(args.audio, sr=None, mono=False)
    slicer = Slicer(
        sr=sr,
        threshold=args.db_thresh,
        min_length=args.min_length,
        min_interval=args.min_interval,
        hop_size=args.hop_size,
        max_sil_kept=args.max_sil_kept
    )
    chunks = slicer.slice(audio)
    if not os.path.exists(out):
        os.makedirs(out)
    base = os.path.basename(args.audio).rsplit('.', maxsplit=1)[0]
    for i, (chunk, start_sample, end_sample) in enumerate(chunks):
        # Calculate start and end time (in seconds)
        start_time = start_sample / sr
        end_time = end_sample / sr
        print(f"Segment {i}: starts at {start_time:.2f}s, ends at {end_time:.2f}s")
        
        # If the chunk is multi-channel, ensure channels are set correctly.
        if len(chunk.shape) > 1:
            chunk = chunk.T  # reshape to (samples, channels)
            channels = chunk.shape[1]
        else:
            channels = 1

        # Convert float32 audio in [-1, 1] to int16 PCM data.
        chunk_int16 = (np.clip(chunk, -1.0, 1.0) * 32767).astype(np.int16)

        # Create a pydub AudioSegment from the raw PCM data.
        audio_segment = AudioSegment(
            data=chunk_int16.tobytes(),
            sample_width=chunk_int16.dtype.itemsize,
            frame_rate=sr,
            channels=channels
        )
        # Export using MP3 with a bitrate of 320k
        out_path = os.path.join(out, f'{base}_{i}.mp3')
        audio_segment.export(out_path, format='mp3', bitrate="320k")
        print(f'Exported: {out_path}')


if __name__ == '__main__':
    main()
