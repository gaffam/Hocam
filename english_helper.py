WORDS = {
    "cat": {
        "meaning": "kedi",
        "example": "The cat is sleeping on the sofa.",
        "pronunciation": "kæt",
    },
    "run": {
        "meaning": "koşmak",
        "example": "I like to run every morning.",
        "pronunciation": "rʌn",
    },
    "school": {
        "meaning": "okul",
        "example": "We go to school five days a week.",
        "pronunciation": "skuːl",
    },
    "table": {
        "meaning": "masa",
        "example": "He put the books on the table.",
        "pronunciation": "ˈteɪb(ə)l",
    },
    "book": {
        "meaning": "kitap",
        "example": "This book is very interesting.",
        "pronunciation": "bʊk",
    },
    "computer": {
        "meaning": "bilgisayar",
        "example": "The computer is on the desk.",
        "pronunciation": "kəmˈpjuːtə",
    },
    "dog": {
        "meaning": "köpek",
        "example": "The dog barked loudly.",
        "pronunciation": "dɒg",
    },
    "happy": {
        "meaning": "mutlu",
        "example": "She feels happy today.",
        "pronunciation": "ˈhæpi",
    },
}

def get_word_info(word: str) -> str:
    info = WORDS.get(word.lower())
    if not info:
        return None
    return (
        f"**{word}** anlamı **{info['meaning']}**.\n"
        f"Örnek: {info['example']}\n"
        f"Telaffuz: /{info['pronunciation']}/"
    )
