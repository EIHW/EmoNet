import tensorflow as tf

""" https://gist.github.com/keunwoochoi/f4854acb68acf791a49a051893bcd23b """
class LogMelgramLayer(tf.keras.layers.Layer):
    def __init__(
        self, num_fft, hop_length, num_mels, sample_rate, f_min=80, f_max=7600, eps=1e-6, return_decibel=False, mask_zero=True, **kwargs
    ):
        super(LogMelgramLayer, self).__init__(**kwargs)
        self.num_fft = num_fft
        self.hop_length = hop_length
        self.num_mels = num_mels
        self.sample_rate = sample_rate
        self.f_min = f_min
        self.f_max = f_max
        self.eps = eps
        self.return_decibel = return_decibel
        self.num_freqs = num_fft // 2 + 1
        self.mask_zero = mask_zero
        
        lin_to_mel_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=self.num_mels,
            num_spectrogram_bins=self.num_freqs,
            sample_rate=self.sample_rate,
            lower_edge_hertz=self.f_min,
            upper_edge_hertz=self.f_max,
        )

        self.lin_to_mel_matrix = lin_to_mel_matrix

    def build(self, input_shape):
        self.non_trainable_weights.append(self.lin_to_mel_matrix)
        super(LogMelgramLayer, self).build(input_shape)

    def call(self, input):
        """
        Args:
            input (tensor): Batch of mono waveform, shape: (None, N)
        Returns:
            log_melgrams (tensor): Batch of log mel-spectrograms, shape: (None, num_frame, mel_bins, channel=1)
        """
        def _tf_log10(x):
            numerator = tf.math.log(x)
            denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
            return numerator / denominator

        stfts = tf.signal.stft(
            input,
            frame_length=self.num_fft,
            frame_step=self.hop_length,
            pad_end=False,  # librosa test compatibility
        )
        mag_stfts = tf.abs(stfts)

        melgrams = tf.tensordot(  # assuming channel_first, so (b, c, f, t)
            tf.square(mag_stfts), self.lin_to_mel_matrix, axes=[2, 0]
        )
        if self.return_decibel:
            log_melgrams =  10 * _tf_log10((melgrams + self.eps) / tf.reduce_max(melgrams))
        else:
            log_melgrams = tf.math.log(melgrams + self.eps)
        return log_melgrams
    
    

    def get_config(self):
        config = {
            'num_fft': self.num_fft,
            'hop_length': self.hop_length,
            'num_mels': self.num_mels,
            'sample_rate': self.sample_rate,
            'f_min': self.f_min,
            'f_max': self.f_max,
            'eps': self.eps,
            'return_decibel': self.return_decibel,
            'mask_zero': self.mask_zero
        }
        base_config = super(LogMelgramLayer, self).get_config()
        return dict(list(config.items()) + list(base_config.items()))
    
    
class MFCCLayer(tf.keras.layers.Layer):
    def __init__(
        self, num_mfccs=13, **kwargs
    ):
        super(MFCCLayer, self).__init__(**kwargs)
        self.num_mfccs = num_mfccs
        

    def build(self, input_shape):
        super(MFCCLayer, self).build(input_shape)

    def call(self, input):
        """
        Args:
            input (tensor): Batch of log mel-spectrograms, shape: (None, num_frame, mel_bins, channel=1)
        Returns:
            mfccs (tensor): Batch of mfccs, shape: (None, num_frame, num_mfccs)
        """

        log_mel_spectrograms = input
        
        mfccs = tf.signal.mfccs_from_log_mel_spectrograms(
        log_mel_spectrograms)[..., :num_mfccs]
        return mfccs

    def get_config(self):
        config = {
            'num_mfccs': self.num_mfccs
            
        }
        base_config = super(MFCCLayer, self).get_config()
        return dict(list(config.items()) + list(base_config.items()))