from audio import tfSound
import tensorflow as tf
from random import shuffle

song = tfSound(r'C:\Users\jesus\Documents\WorkFiles\PythonCode\TF2_personal_projects\ArrivalPhobos.wav')
chunks_list = song.chunks(1, None)
shuffle(chunks_list)
tensor_list = [tf.convert_to_tensor(chunk.data) for chunk in chunks_list]

print(tensor_list[1])
#must figure out how this model is going to work
