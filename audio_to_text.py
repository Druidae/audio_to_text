import whisper


def speech_recognition(model='base'):
    speech_model = whisper.load_model(model)
    result = speech_model.transcribe('data/kwork.aac', fp16=False)

    with open(f'text_data/kwork_{model}.txt', 'w') as file:
        file.write(result['text'])


def main():
    models = {1: 'tiny', 2: 'base', 3: 'small', 4: 'medium', 5: 'large'}

    for k, v in models.items():
        print(f'{k}:{v}')

    model = int(input("Select the model by passing the number from 1 to 5: "))

    if model not in models.keys():
        raise KeyError(f'[!] No model {model} in models list')

    print("Starting transcribing process, please wait...")
    speech_recognition(model=models[model])


if __name__ == '__main__':
    main()
