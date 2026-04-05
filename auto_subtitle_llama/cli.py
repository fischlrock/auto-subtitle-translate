import os
import ffmpeg
import whisper
import argparse
import warnings
import tempfile
from .utils import *
from typing import List, Tuple
from tqdm import tqdm

# Uncomment below and comment "from .utils import *", if executing cli.py directly
# import sys
# sys.path.append(".")
# from auto_subtitle_llama.utils import *

# deal with huggingface tokenizer warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("video", nargs="+", type=str,
                        help="paths to video files to transcribe")
    parser.add_argument("--model", default="turbo",
                        choices=whisper.available_models(), help="name of the Whisper model to use")
    parser.add_argument("--output_dir", "-o", type=str,
                        default="subtitled", help="directory to save the outputs")
    parser.add_argument("--output_srt", type=str2bool, default=True,
                        help="whether to output the .srt file along with the video files")
    parser.add_argument("--srt_only", type=str2bool, default=False,
                        help="only generate the .srt file and not create overlayed video")
    parser.add_argument("--verbose", type=str2bool, default=False,
                        help="whether to print out the progress and debug messages")

    parser.add_argument("--task", type=str, default="transcribe", choices=[
                        "transcribe", "translate"], help="whether to perform X->X speech recognition ('transcribe') or X->English translation ('translate')")
    parser.add_argument("--language", type=str, default="auto", choices=["auto","af","am","ar","as","az","ba","be","bg","bn","bo","br","bs","ca","cs","cy","da","de","el","en","es","et","eu","fa","fi","fo","fr","gl","gu","ha","haw","he","hi","hr","ht","hu","hy","id","is","it","ja","jw","ka","kk","km","kn","ko","la","lb","ln","lo","lt","lv","mg","mi","mk","ml","mn","mr","ms","mt","my","ne","nl","nn","no","oc","pa","pl","ps","pt","ro","ru","sa","sd","si","sk","sl","sn","so","sq","sr","su","sv","sw","ta","te","tg","th","tk","tl","tr","tt","uk","ur","uz","vi","yi","yo","zh"], 
    help="What is the origin language of the video? If unset, it is detected automatically.")
    parser.add_argument("--translate_to", type=str, default=None, choices=["af","am","ar","as","az","ba","be","bg","bn","bo","br","bs","ca","cs","cy","da","de","el","en","es","et","eu","fa","fi","fo","fr","gl","gu","ha","haw","he","hi","hr","ht","hu","hy","id","is","it","ja","jw","ka","kk","km","kn","ko","la","lb","ln","lo","lt","lv","mg","mi","mk","ml","mn","mr","ms","mt","my","ne","nl","nn","no","oc","pa","pl","ps","pt","ro","ru","sa","sd","si","sk","sl","sn","so","sq","sr","su","sv","sw","ta","te","tg","th","tk","tl","tr","tt","uk","ur","uz","vi","yi","yo","zh"],
    help="Final target language code; af=Afrikaans, ar=Arabic, bn=Bengali, cs=Czech, de=German, en=English, es=Spanish, fa=Persian, fi=Finnish, fr=French, gu=Gujarati, hi=Hindi, id=Indonesian, it=Italian, ja=Japanese, ko=Korean, lt=Lithuanian, lv=Latvian, ml=Malayalam, mr=Marathi, ms=Malay, ne=Nepali, nl=Dutch, pl=Polish, ps=Pashto, pt=Portuguese, ro=Romanian, ru=Russian, si=Sinhala, sv=Swedish, sw=Swahili, ta=Tamil, te=Telugu, th=Thai, tl=Tagalog, tr=Turkish, uk=Ukrainian, ur=Urdu, vi=Vietnamese, zh=Chinese")
    
    args = parser.parse_args().__dict__
    model_name: str = args.pop("model")
    output_dir: str = args.pop("output_dir")
    output_srt: bool = args.pop("output_srt")
    srt_only: bool = args.pop("srt_only")
    language: str = args.pop("language")
    translate_to: str = args.pop("translate_to")
    
    os.makedirs(output_dir, exist_ok=True)

    if model_name.endswith(".en"):
        warnings.warn(
            f"{model_name} is an English-only model, forcing English detection.")
        args["language"] = "en"
    # if translate task used and language argument is set, then use it
    elif language != "auto":
        args["language"] = language
    
    print(f"Loading {model_name} model")
    model = whisper.load_model(model_name)
    print(f"Finish loading {model_name} model")

    audios = get_audio(args.pop("video"))
    subtitles, detected_language = get_subtitles(
        audios, 
        output_srt or srt_only, 
        output_dir, 
        model,
        args, 
        translate_to=translate_to
    )

    if srt_only:
        return
    
    _translated_to = ""
    if translate_to:
        # for filename
        _translated_to = f"2{translate_to}"
        
    for path, srt_path in subtitles.items():
        out_path = os.path.join(output_dir, f"{filename(path)}_subtitled_{detected_language}{_translated_to}.mp4")

        print(f"Adding subtitles to {filename(path)}...")

        video = ffmpeg.input(path)
        audio = video.audio

        ffmpeg.concat(
            video.filter('subtitles', srt_path, force_style="FallbackName=NanumGothic,OutlineColour=&H40000000,BorderStyle=3", charenc="UTF-8"), audio, v=1, a=1
        ).output(out_path).run(quiet=True, overwrite_output=True)

        print(f"Saved subtitled video to {os.path.abspath(out_path)}.")


def get_audio(paths):
    temp_dir = tempfile.gettempdir()

    audio_paths = {}

    for path in paths:
        print(f"Extracting audio from {filename(path)}...")
        output_path = os.path.join(temp_dir, f"{filename(path)}.wav")

        ffmpeg.input(path).output(
            output_path,
            acodec="pcm_s16le", ac=1, ar="16k"
        ).run(quiet=True, overwrite_output=True)

        audio_paths[path] = output_path

    return audio_paths


def get_subtitles(audio_paths: list, output_srt: bool, output_dir: str, model:whisper.model.Whisper, args: dict, translate_to: str = None) -> Tuple[dict, str]:
    subtitles_path = {}

    for path, audio_path in audio_paths.items():
        srt_path = output_dir if output_srt else tempfile.gettempdir()
        srt_path = os.path.join(srt_path, f"{filename(path)}.srt")
        
        print(
            f"Generating subtitles for {filename(path)}... This might take a while."
        )

        warnings.filterwarnings("ignore")
        print("[Step1] detect language (Whisper)")
        # load audio and pad/trim it to fit 30 seconds
        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)
        # make log-Mel spectrogram
        mel = whisper.log_mel_spectrogram(audio, model.dims.n_mels).to(model.device)
        if args["language"]=="auto":
            # detect the spoken language
            _, probs = model.detect_language(mel)
            detected_language = max(probs, key=probs.get)
        else:
            detected_language = args["language"]  
        current_lang = LANG_CODE_MAPPER.get(detected_language, [])

        print(f"Detected Language: {detected_language}")
        print(f"Curent Language: {current_lang}") 
        
        print("[Step2] transcribe (Whisper)")
        if translate_to != None and detected_language != translate_to:
            print("[Step3] translate")
            args["task"]="translate"
        print(args)
        result = model.transcribe(audio_path, **args)

        with open(srt_path, "w", encoding="utf-8") as srt:
            write_srt(result["segments"], file=srt)
        print(f"srt file is saved: {srt_path}")
        subtitles_path[path] = srt_path

    return subtitles_path, detected_language


if __name__ == '__main__':
    main()
