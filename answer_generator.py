from typing import List
from english_helper import get_word_info
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    AutoConfig,
)
import torch


class AnswerGenerator:
    def __init__(self, model_name: str = "cahya/gpt2-small-turkish", cache_dir: str = None):
        """Load the model (Seq2Seq or CausalLM) and move it to the available device."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
        if getattr(config, "is_encoder_decoder", False):
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=cache_dir)
            self.is_seq2seq = True
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
            self.is_seq2seq = False
        self.model.to(self.device)
        self.model.eval()

    def generate(
        self,
        context_chunks: List[str],
        question: str,
        grade: int = 1,
        subject: str = "Türkçe",
        max_length: int = 150,
    ) -> str:
        """Generate an answer using the provided context."""
        context = "\n".join(context_chunks)

        # Special handling for English words
        if subject == "İngilizce":
            word_info = get_word_info(question.strip())
            if word_info:
                if grade <= 4:
                    return word_info + " \U0001F60A"
                return word_info

        if subject == "İngilizce":
            prompt = (
                f"{context}\n\nSoru: {question}\nCevap Türkçe ver. "
                "Kelimenin anlamını, örnek cümle ve telaffuz ipucu ekle."
            )
        else:
            prompt = f"{context}\n\nSoru: {question}\nCevap:"
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=inputs["input_ids"].shape[1] + max_length,
                do_sample=True,
                top_p=0.95,
                top_k=50,
            )
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if not self.is_seq2seq:
            answer = answer.split("Cevap:")[-1]
        return self._format_answer(answer.strip(), grade)

    @staticmethod
    def _format_answer(answer: str, grade: int) -> str:
        """Format answer according to grade level."""
        if grade <= 4:
            return f"{answer} \U0001F44D"
        return answer
