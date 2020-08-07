from sksound.sounds import Sound
import numpy as np
from random import shuffle
import tensorflow as tf
'''
SoundProperties:
    source
    data
    rate
    numChannels
    totalSamples
    duration
    bitsPerSample
Sound info: get_info method. returns list with these values
    source (name of inFile)
    rate (sampleRate)
    numChannels (number of channels)
    totalSamples (number of total samples)
    duration (duration [sec])
    bitsPerSample (bits per sample)
'''
class tfSound(Sound):
    def __init__(self, *args, **kwargs):
        super(tfSound, self).__init__( *args, **kwargs)
    def sample(self, seconds, start=0):
        """
        returns a tfsound object that has specified number of seconds long
        at starting second
        """
        dis = self.rate * seconds
    
        new = self.data[start:dis+start]
            
        return tfSound(inData=new, inRate=self.rate)
    def chunks(self, seconds = 5, num = None):
        """
        returns list with tfSound objects that are specified seconds long
        or specific number of chunks
        seconds can be specified to None and automatically figure out how long each chunk is
        """
        k = []
        if num == None:
            for i in range(int(self.totalSamples / (5 * self.rate))):
                k.append(self.sample(seconds, i * self.rate * seconds))
        elif seconds ==None:
            samples_used = (self.totalSamples / num)
            for i in range(num):
                k.append(self.sample(seconds, i * samples_used))
        else:
            for i in range(num):
                k.append(self.sample(seconds, i * self.rate * seconds))
        return k
    def shuffled_chunks(self, *args):
        s = self.chunks(*args)
        return shuffle(s)
