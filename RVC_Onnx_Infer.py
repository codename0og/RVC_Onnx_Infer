import os
import glob
import soundfile

from infer.lib.infer_pack.onnx_inference import OnnxRVC


# PATHS 
root_path = os.path.dirname(os.path.abspath(__file__))  # setting the root dir for RVCp
input_path = os.path.join(root_path, "input")  # audio input
output_path = os.path.join(root_path, "output")  # inference output
onnx_models = os.path.join(root_path, "models") # ONNX models folder





# INFERENCE CONFIGURATION
sampling_rate = 48000  # Your model's sample rate;   32000, 40000, 48000
f0_up_key = 0  # pitch in semitones, either up or down.
f0_method = "dio"  # F0 pitch extraction method.  For now:   dio, pm and harvest are supported
hop_size = 64 # hop size for inference  -  smaller size = higher accuracy, yet, higher risk of catching noise residues. try: 64, 128 or 512 or try own / custom.
sid = 0 # Speaker ID, unusable atm.
vec_name = "vec-768-layer-12"  # pretrained ONNX variant of vec

# DEVICE SETTINGS
device = "cpu"  # options: dml, cuda, cpu

# Set your model's name                         / Here / 
model_path = os.path.join("onnx_models", "Your_Model.onnx")  # Your .ONNX model
output_folder = "output"  # Output folder for inferences
output_filename = "infer_output.wav"  # name for inference outputs






# Search for your .wav files in the input dir.
wav_files = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.lower().endswith('.wav')]
if not wav_files:
    raise FileNotFoundError("No WAV files found in the 'input' dir.")
wav_path = wav_files[0] # input for inference ( First found .wav from input dir )
out_path = os.path.join("output", output_filename) # ( Inference output lands into output dir )

model = OnnxRVC(
    model_path,
    vec_path=vec_name,
    sr=sampling_rate,
    hop_size=hop_size,
    device=device
    )

audio = model.inference(wav_path, sid, f0_method=f0_method, f0_up_key=f0_up_key)

try:
    soundfile.write(out_path, audio, sampling_rate)
    print("  INFERENCE SUCCESSFUL! CHECK 'output' FOLDER! ")
except Exception as e:
    print(f" AN ERROR OCCURRED: {e}")