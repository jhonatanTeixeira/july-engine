# July Engine: Guia de Codificação para IA

## 🧩 Princípios de Design
1.  **Nenhum Mock nos Testes**: Todos os testes em `tests/test_integration.py` devem ser reais. Use modelos leves (Qwen 0.6B, Moondream, Nanonets) para validação.
2.  **Abstração via Bridge**: Nunca chame um orquestrador diretamente nos routers. Use sempre a classe `Bridge`.
3.  **Hibridismo de Backend**: Sempre considere que uma funcionalidade pode rodar em `cpu`, `gpu` ou `api`. O header `x-backend` é o mestre.
4.  **Gerenciamento de Recursos**: Ao adicionar modelos de GPU, sempre registre-os no `GpuOrchestrator` e chame `resource_manager.clear_memory()` após o uso se forem modelos pesados.

## 📁 Estrutura de Pastas Crítica
-   `jully_engine/engine_models/`: Implementações de baixo nível (bibliotecas de ML).
-   `jully_engine/domain/`: Lógica de negócio e roteamento de estratégia (Brain, Eyes, etc.).
-   `jully_engine/orchestrators/`: Gerenciamento de filas e hardware.
-   `storage/temp/`: Local obrigatório para arquivos temporários de áudio/imagem.

## 📝 Padrões de API
-   **OpenAI Parity**: Mantenha os DTOs em `openai.py` compatíveis com a versão mais recente da API OpenAI.
-   **Anthropic Parity**: O arquivo `anthropic.py` deve espelhar as capacidades do OpenAI, mesmo que a Anthropic não suporte nativamente (ex: TTS).
-   **Tuning Parameters**: Todo endpoint de Chat deve suportar `temperature`, `top_p`, `max_tokens`, `num_ctx`, etc. Filtre `None` antes de passar para bibliotecas como `llama-cpp`.

## 🗣️ Regras de TTS e Vozes
-   **Configuração de Vozes**: Utilize `voices.json` para vozes estáticas e `uploaded_voices.json` (não `uploaded.json`) para vozes de usuários.
-   **Dual Path**: Mantenha sempre suporte aos dois campos:
    -   `path`: Relativo à `storage/voices/` (usado por XTTS2 como referência `.wav`).
    -   `piper_path`: Caminho no formato Hugging Face (usado por Piper para baixar `.onnx` e `.onnx.json`).
-   **Resolução Dinâmica**: A classe `Mouth` tenta adivinhar o caminho se o ID contiver hifens ou barras, mas a preferência é sempre o mapeamento nos arquivos JSON.

## ⚙️ Correção de Swagger (Dica Técnica)
Para que os exemplos de saída apareçam no Swagger:
1.  Use `response_model` no decorador do router.
2.  Defina `examples` dentro da classe `Config` (Pydantic v1) ou `model_config` (Pydantic v2) dos seus DTOs de resposta.
3.  Para campos específicos, use `Field(..., examples=["exemplo"])`.
