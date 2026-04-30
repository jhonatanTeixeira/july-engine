import base64
import logging
import sys
import tempfile
from ascii_magic import AsciiArt
import colorama

# Configura o logger raiz para printar tudo do nível INFO para cima direto no terminal
logging.basicConfig(
    level=logging.INFO,  # Troque para logging.DEBUG se quiser ver até os escovadores de bit
    format="%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

import asyncio
import argparse
import sys
import json

from july_engine.domain.brain import Brain
# from july_engine.model_loader import model_loader
from july_engine.persistence import get_backend


async def main():
    # 1. Configurando o CLI (argparse)
    parser = argparse.ArgumentParser(description="Testador CLI Bare-Metal para o Brain e July MCP")
    parser.add_argument("--model-alias", type=str, required=True, help="O alias do modelo configurado no banco (ex: qwen-06b)")
    parser.add_argument("--stream", action="store_true", help="Ativa o modo stream (AsyncGenerator)")
    parser.add_argument("--prompt", type=str, default="gere a imagem de um mustang 1967", help="O prompt de teste")
    parser.add_argument("--layers", type=int, default=15, help="Força a quantidade de camadas na GPU (ex: 15)")
    parser.add_argument("--num-ctx", type=int, default=2048, help="Define o tamanho da janela de contexto")
    args = parser.parse_args()

    print(f"🚀 Iniciando teste do Brain | Modelo: {args.model_alias} | Stream: {args.stream}")

    # 2. Emulando o _enrich_headers_and_payload (Busca no Banco)
    backend_db = get_backend()
    text_presets = backend_db.get_setting("TEXT_PRESETS") or []
    
    config = next((p for p in text_presets if p.get("alias") == args.model_alias), None)
    if not config and text_presets:
        print(f"⚠️ Alias '{args.model_alias}' não encontrado. Usando o default: {text_presets[0].get('alias')}")
        config = text_presets[0]
    
    if not config:
        print("❌ Erro: Nenhum TEXT_PRESET encontrado no banco de dados.")
        sys.exit(1)

    backend_type = config.get("backend", "gpu")
    model_name = config.get("model", args.model_alias)

    # 3. Montando os Headers e o Payload (A Mágica do MCP)
    headers = {
        "x-enable-internal-mcp": "1",  # A CHAVE QUE LIGA A ENGINE XML BARE-METAL!
        "x-backend": backend_type
    }
    
    if "base_url" in config:
        headers["x-base-url"] = config["base_url"]
    if "api_key" in config and config["api_key"]:
        headers["x-api-key"] = config["api_key"]
        headers["authorization"] = f"Bearer {config['api_key']}"
        
    headers['x-context-window'] = args.num_ctx

    payload = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": args.prompt}
        ],
        "stream": args.stream,
        "temperature": 0.7,
        "headers": headers,
        "num_layers": args.layers
    }

    print(f"🔧 Configuração resolvida: Backend={backend_type}, Model={model_name}")
    print(f"💬 Prompt: '{args.prompt}'")
    print("-" * 50)

    # 4. Instanciando o Cérebro e Executando
    try:
        # brain = model_loader.get_brain(backend=backend_type, model_tag=model_name)
        brain = Brain(backend=backend_type, model_tag=model_name)
        response = await brain.chat(payload)

        # 5. Tratando o Retorno (Stream vs Dict)
        if args.stream:
            print("🌊 Stream iniciado:\n")
            async for chunk in response:
                # O chunk pode ser um dicionário padrão da OpenAI
                if isinstance(chunk, dict):
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content", "")
                    rcontent = delta.get("reasoning_content", "")
                    img_content = delta.get("image_url", "")
                    
                    if rcontent:
                        print('\033[94m' + rcontent, end="", flush=True)
                        
                    if content:
                        print('\033[95m' + content, end="", flush=True)
                    
                    if img_content:
                        if "," in img_content and img_content.startswith("data:image"):
                            img_content = img_content.split(",", 1)[1]

                        # 2. A Mágica do Padding: Adiciona os '=' finais que se perderam na rede
                        missing_padding = len(img_content) % 4
                        if missing_padding:
                            img_content += '=' * (4 - missing_padding)
                        
                        img_bytes = base64.b64decode(img_content) 

                        with tempfile.NamedTemporaryFile(delete=False, suffix=".png", prefix="july_engine_") as temp_file:
                            temp_file.write(img_bytes)

                        print('\033[94m' + 'image created', flush=True)
                        my_art = AsciiArt.from_image(temp_file.name)
                        colorama.init()
                        my_art.to_terminal()
                        
                else:
                    # Caso vaze algum objeto cru, printamos para debug
                    print(f"\n[Objeto não tratado no stream: {chunk}]")
            print("\n")
        else:
            print("📦 Resposta Completa (JSON):\n")
            print(json.dumps(response, indent=2, ensure_ascii=False))

    except Exception as e:
        print(f"\n❌ Erro crítico durante a execução do Brain")
        raise e

if __name__ == "__main__":
    # Roda o event loop do asyncio
    asyncio.run(main())