# Hocam

Bu proje, PDF dosyalarından Türkçe metin çıkararak FAISS vektör veritabanına kaydeden ve dil modeli ile sorulara cevap veren bir Streamlit uygulamasıdır.

## Kurulum

```bash
pip install -r requirements.txt
```

Python 3.8+ gereklidir. Tüm bağımlılıklar kurulduktan sonra ilk çalıştırmada gerekli modeller Hugging Face üzerinden indirilerek `~/.cache/huggingface` klasörüne kaydedilir. Bu indirme işlemini ilk çalıştırmadan önce yapmak isterseniz aşağıdaki komutu çalıştırabilirsiniz:

```bash
python - <<'PY'
from transformers import AutoTokenizer, AutoModel
for model in [
    "dbmdz/gpt2-turkish",
    "AI4Turk/ke-t5-small-tr",
    "cahya/gpt2-small-turkish",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
]:
    AutoTokenizer.from_pretrained(model)
    AutoModel.from_pretrained(model)
PY
```

Bu sayede uygulama çevrimdışı ortamlarda da çalıştırılabilir.

## Kullanım

```bash
streamlit run app.py
```

Arayüzde önce sınıf ve ders seçimi yaparak PDF yükleyip **İndeks Oluştur** butonuna basın. Sorularınızı yazıp **Cevapla** butonu ile yanıtları görebilirsiniz. Her sınıf ve ders için ayrı `index/`, `embeddings/` ve `chunks/` klasörlerinde dosyalar tutulur.

### Özellikler

- PDF yüklemeden soru sorulmaya çalışıldığında kullanıcıya kısa bir uyarı gösterilir.
- `Model seç` menüsünden GPT-2 veya T5 tabanlı Türkçe modelleri seçebilirsiniz.
- Seçilen modelin türü otomatik algılanır ve uygun şekilde yüklenir.
- GPU varsa modeller otomatik olarak GPU'ya taşınır.
- Arama sonuçları ilk 10 parça içinden yeniden sıralanarak en alakalı 3 parça kullanılır.
- Tüm veriler yerel dizinde saklandığı için uygulama internet bağlantısı olmadan da çalışabilir.
- Her sınıf ve ders için ayrı indeks dosyaları tutulur.
- İngilizce dersinde kelime bazlı sorular için Türkçe açıklamalı özel yanıt verilir.
- PDF metinleri artık tokenizer kullanılarak token bazlı parçalara ayrılır.
- ``extract_chunks`` fonksiyonu embedding modelinin tokenizer'ını
  kullanacak şekilde güncellenmiştir ve parçalara üst üste binme
  (``chunk_overlap``) ekleme seçeneği sunar.
- Sesli soru sorma ve gTTS ile sesli yanıt alma desteği eklenmiştir.
- Kullanıcı adı ile giriş yapıldığında öğrenci puan kazanır, veliler puan tablosunu görüntüler.
- Puanlar basit bir SQLite veritabanında saklanır.
- Embedding modeli varsayılan olarak `paraphrase-multilingual-MiniLM-L12-v2` kullanır, `bert-base-turkish-cased` ile değiştirmek mümkündür.
