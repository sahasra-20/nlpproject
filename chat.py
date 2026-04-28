# chat.py
import torch

from config import Config
from rag_inference import answer_with_rag, build_retriever, load_model_and_tokenizer


def main():
    print("========================================")
    print("      Agricultural RAG QA Chatbot       ")
    print("========================================")
    print("Initializing RAG system... (please wait)")

    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, tokenizer = load_model_and_tokenizer(config, device)
    retriever = build_retriever(model, tokenizer, config, device)

    print("\nChatbot is ready! Type 'quit' or 'exit' to stop.")
    print("-" * 40)

    while True:
        try:
            question = input("\nFarmer > ").strip()
            if question.lower() in ["quit", "exit"]:
                print("Goodbye!")
                break

            if not question:
                continue

            result = answer_with_rag(
                question,
                model,
                tokenizer,
                retriever,
                config,
                device,
                beam_width=4,
                verbose=True,
            )

            print(f"Bot    > {result['answer']}")
            print(f"RAG    > {'used' if result['used_rag'] else 'not used'} ({len(result['context'])} chunks)")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as exc:
            print(f"\n[Error] {exc}")


if __name__ == "__main__":
    main()
