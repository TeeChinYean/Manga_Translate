import os
import torch
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

ONNX_DIR = "data/models/onnx"
os.makedirs(ONNX_DIR, exist_ok=True)

def export_easyocr_craft():
    logger.info("Attempting to export EasyOCR CRAFT detector to ONNX...")
    try:
        import easyocr
        # Initialize reader (we only need the detector)
        reader = easyocr.Reader(['ja', 'en'], gpu=False)
        craft_net = reader.detector
        
        # CRAFT takes input [1, 3, H, W] where H and W are multiples of 32
        dummy_input = torch.randn(1, 3, 640, 640, device='cpu')
        out_path = os.path.join(ONNX_DIR, "craft_detector.onnx")
        
        logger.info("Tracing CRAFT model...")
        torch.onnx.export(
            craft_net, 
            dummy_input, 
            out_path,
            opset_version=12,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output', 'feature_map'],
            dynamic_axes={'input': {2: 'height', 3: 'width'}, 'output': {2: 'height', 3: 'width'}, 'feature_map': {2: 'height', 3: 'width'}}
        )
        logger.info(f"CRAFT exported successfully to {out_path}")
    except Exception as e:
        logger.error(f"Failed to export CRAFT: {e}")

def export_manga_ocr():
    logger.info("Attempting to export MangaOCR to ONNX...")
    try:
        # MangaOCR uses HuggingFace VisionEncoderDecoderModel
        from transformers import VisionEncoderDecoderModel
        
        # MangaOCR default model is 'kha-white/manga-ocr-base'
        model = VisionEncoderDecoderModel.from_pretrained("kha-white/manga-ocr-base")
        model.eval()
        
        # It's highly complex to export a full encoder-decoder with beam search directly via basic torch.onnx.export.
        # Optimum library is highly recommended for HuggingFace to ONNX.
        logger.warning("MangaOCR is a Transformer Encoder-Decoder. Basic export might fail or produce separate Encoder/Decoder ONNX files.")
        
        # Dummy input for ViT (typically 3x224x224)
        dummy_pixel_values = torch.randn(1, 3, 224, 224)
        
        out_path = os.path.join(ONNX_DIR, "manga_ocr_encoder.onnx")
        torch.onnx.export(
            model.encoder,
            dummy_pixel_values,
            out_path,
            opset_version=14,
            input_names=['pixel_values'],
            output_names=['last_hidden_state'],
            dynamic_axes={'pixel_values': {0: 'batch_size'}}
        )
        logger.info(f"MangaOCR Encoder exported successfully to {out_path}")
        
        # Note: Decoder requires past_key_values and input_ids for autoregressive generation. 
        # Writing C# inference for this is extremely complex without ONNXRuntime GenAI.
        logger.warning("Decoder export skipped: Requires deep integration with Optimum or ORT-GenAI for autoregressive decoding.")
        
    except Exception as e:
        logger.error(f"Failed to export MangaOCR: {e}")

def export_lama():
    logger.info("Attempting to export LaMa Inpainter to ONNX...")
    try:
        import sys
        # manga-image-translator (GPL-3.0) is NOT part of this repository: point MIT_SOURCE_DIR at your own checkout
        mit_path = os.getenv("MIT_SOURCE_DIR", "")
        if not mit_path or not os.path.isdir(mit_path):
            logger.error("Set MIT_SOURCE_DIR to a checkout of https://github.com/zyddnys/manga-image-translator to export LaMa.")
            return
        sys.path.insert(0, os.path.abspath(mit_path))
        from manga_translator.inpainting import get_inpainter
        
        # Assume it's LaMa (Valid choices: default, lama_large, lama_mpe, sd, none, original)
        inpainter = get_inpainter("lama_large")
        if inpainter is None or not hasattr(inpainter, "model"):
            logger.error("Could not load LaMa model from manga_translator.")
            return
            
        lama_model = inpainter.model.eval().cpu()
        
        # LaMa typically takes image [1, 3, H, W] and mask [1, 1, H, W]
        dummy_img = torch.randn(1, 3, 512, 512)
        dummy_mask = torch.randn(1, 1, 512, 512)
        
        out_path = os.path.join(ONNX_DIR, "lama_inpainter.onnx")
        
        # Some LaMa models take a dict, others take tensors directly. 
        # This will likely crash if the forward pass expects kwargs or dicts, which is common in BigLaMa.
        try:
            torch.onnx.export(
                lama_model,
                (dummy_img, dummy_mask),
                out_path,
                opset_version=11,
                input_names=['image', 'mask'],
                output_names=['inpainted']
            )
            logger.info(f"LaMa exported successfully to {out_path}")
        except Exception as forward_err:
            logger.error(f"LaMa forward pass tracing failed (often requires dict input): {forward_err}")
            
    except Exception as e:
        logger.error(f"Failed to export LaMa: {e}")

if __name__ == "__main__":
    logger.info("Starting ONNX Export Phase...")
    export_easyocr_craft()
    export_manga_ocr()
    export_lama()
    logger.info("ONNX Export Script Completed.")
