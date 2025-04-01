import base64
import requests
from openai import OpenAI
from openai import AzureOpenAI
from scipy.io import wavfile
import pandas as pd
import os

client = OpenAI(api_key='')

folder_path='./jailbreak_data/jailbreak_speech_persuasive/'
res=[]
df=pd.read_csv('jailbreak_gpt4o.csv')
for index, row in df.iterrows():
    # 1st dialogue turn
    print(index)
    #filename=row['path'].replace('mp3','wav')
    filename=str(index+1)+'.wav'
    path=os.path.join(folder_path, filename)
    # Read the WAV file in binary mode
    with open(path, 'rb') as wav_file:
        wav_data = wav_file.read()

    # Display the data in hex format (optional)
    #print(wav_data)
    encoded_string = base64.b64encode(wav_data).decode('utf-8')
    prompt='Please respond to the audio.'
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-audio-preview",
            modalities=["text", "audio"],
            audio={"voice": "alloy", "format": "wav"},
            messages=[
                {
                    "role": "user",
                    "content": [
                        { 
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": encoded_string,
                                "format": "wav"
                            }
                        }
                    ]
                },
            ]
        )

        #print(completion.choices[0].message)
        print(completion.choices[0].message.audio.transcript)
        res.append(completion.choices[0].message.audio.transcript)
    except Exception as e:
        # This block will catch any other exception
        print(f"An unexpected error occurred: {e}")
        res.append('error')
df['gpt4o_persuasive_res']=res
df.to_csv('jailbreak_gpt4o2.csv', index=False)