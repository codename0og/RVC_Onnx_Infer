# Codename;0's improved RVC Onnx models Inference. 
## Ready to be used with RVC V2 onnx models. ( CPU, Cuda and DML support )<br />
### Todo:
- Adding index/faiss support
- Automating stuff / making i/o handling easier.
- Adding rmvpe f0 method
- Better automation and easier input/output managment + stuff picker.
- Possibly even a gui or web-ui ~ one day huh.
- Quite possibly a tflite model exporting for future Mobile-RVC-infer-port-project ( ***Not 100% sure yet, concept stage.*** )<br />
# Usage guide:

### 1. First, prior to any inferencing, you gotta obtain the: '**vec-768-layer-12.onnx**' file from:<br />
**https://huggingface.co/NaruseMioShirakana/MoeSS-SUBModel/tree/main**<br />

Place it here: RVC_Onnx_Infer/assets/vec
> reference: '**RVC_Onnx_Infer/assets/vec/vec-768-layer-12.onnx**'

⠀<br />
### 2. Your .onnx models land into '**onnx_models**' folder
( You set which one to use in the 30th line of '**RVC_Onnx_Infer.py**' script )
> model_path = os.path.join("onnx_models", "**Your_Model.onnx**")  # Your .ONNX model

⠀<br />
### 3. Your vocals for inference / acapella .wav goes into 'input' folder.
( Script will pick only the first found .wav in there, so, always have just 1 in there to avoid issues. )

⠀<br />
### 4. Your inference outputs will appear in the '**outpit**' folder.
( One at a time. Any consecutive inferences will overwrite the previous file so, copy / move it somewhere else.

⠀<br />
### 5. To switch the device to Cuda or DML, change "**cpu**" to any of the mentioned.g<br />
The 27th line of '**RVC_Onnx_Infer.py**' script;
> device = "**cpu**"  # options: **dml**, **cuda**, **cpu**

⠀<br />
⠀<br />
⠀<br />
# INITIAL RELEASE: v0.1a<br />
### Notes:
- Project is in an **early alpha-dev / test / debug state.**
- Currently only Dio F0 Pitch extraction until I figure out the rest.
- It is supporting **RVC V2 onnx models only.**<br />
(V1 models do not work unless you get 256-layer-9 vec onnx and modify the code appropriately.)
⠀<br />
- **CPU is set by default** as the main device for the sake of compatibility, need more testing.<br />
