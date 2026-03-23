import argparse
import os
import time
from PIL import Image

from jully_engine.engine_models.stable_diffusion import StableDiffusion

def main():
    parser = argparse.ArgumentParser(description="Teste Arquiteto - Stable Diffusion (LCM + IP-Adapters)")
    parser.add_argument("--prompt", type=str, required=True, help="O prompt de texto (Ex: 'office woman')")
    parser.add_argument("--neg", type=str, default="", help="Prompt negativo opcional")
    parser.add_argument("--ref", type=str, help="Caminho da imagem de estilo/composição (IP-Adapter Plus)")
    parser.add_argument("--face", type=str, help="Caminho da imagem do rosto (IP-Adapter FaceID)")
    parser.add_argument("--out", type=str, default="july_test.jpg", help="Caminho de saída da imagem gerada")
    
    args = parser.parse_args()

    # Usando raw strings (r"") para blindar os caminhos do Windows
    MODEL_PATH = r"E:\ai\stable-diffusion-webui\models\Stable-diffusion\epicphotogasm_ultimateFidelity.safetensors"
    LORA_DIR = r"E:\ai\stable-diffusion-webui\models\Lora"

    print(f"\n[*] Iniciando Motor Stable Diffusion")
    print(f"[*] Prompt: '{args.prompt}'")
    
    # Carregamento seguro das imagens opcionais
    ref_img = None
    if args.ref and os.path.exists(args.ref):
        print(f"[*] Imagem de Estilo ativada: {args.ref}")
        ref_img = Image.open(args.ref).convert("RGB")
        
    face_img = None
    if args.face and os.path.exists(args.face):
        print(f"[*] Identidade Facial ativada: {args.face}")
        face_img = Image.open(args.face).convert("RGB")

    print("\n[1/3] Subindo pesos na VRAM (Pode levar alguns segundos na primeira vez)...")
    start_time = time.time()
    
    # Instancia a engine (Note que passamos LORA_DIR, a pasta mãe)
    engine = StableDiffusion(model_path=MODEL_PATH, lora_path=LORA_DIR, backend='gpu')
    
    print(f"[2/3] Motor carregado em {time.time() - start_time:.2f}s. Gerando (LCM 6 steps)...")
    gen_start = time.time()
    
    try:
        # Chama a inferência passando todos os parâmetros (None é tratado nativamente)
        result_image = engine.generate(
            prompt=args.prompt,
            negative_prompt=args.neg,
            reference_image=ref_img,
            face_image=face_img,
            steps=6,
            cfg_scale=1.5
        )
        
        print(f"[3/3] Inferência concluída em {time.time() - gen_start:.2f}s!")
        
        result_image.save(args.out)
        print(f"[*] SUCESSO! Arquivo salvo em: {args.out}")
        
    except Exception as e:
        print(f"[!] ERRO FATAL DURANTE A GERAÇÃO: {e}")
        
    finally:
        print("\n[*] Descarregando motor e acionando Garbage Collector...")
        engine.unload()
        print("[*] VRAM 100% liberada. Fim do teste.")

if __name__ == "__main__":
    main()