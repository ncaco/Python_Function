from pydub import AudioSegment
from Audio import getAudio

path = getAudio.select_audio_file()

song = AudioSegment.from_mp3(path)

print(song)

# PyDub handles time in milliseconds
ten_minutes = 10 * 60 * 1000

first_10_minutes = song[:ten_minutes]

first_10_minutes.export("result.mp3", format="mp3")