from resemblyzer import VoiceEncoder, preprocess_wav
import numpy as np

enc = VoiceEncoder()
me = preprocess_wav("assets/my_voice_short.wav")
friend = preprocess_wav("assets/my_voice_long.wav")

emb_me = enc.embed_utterance(me)
emb_friend = enc.embed_utterance(friend)

sim = np.dot(emb_me, emb_friend)
print("Similarity:", sim)
