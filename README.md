# 🌟 RAG_pipeline

Aby odpalić projekt należy wykonać poniższe kroki (Linux).

---

## 📘 0. Wprowadzenie

1. Zapoznaj się z prezentacją (RAGdocs), w której omówiono zaimplementowaną architekturę RAG.  
   Z prezentacji dowiesz się również, na jakim etapie jest projekt tworzenia pipelinu:  
   [RAGdocs](https://drive.google.com/drive/folders/1mkRFEg8djZen7azcsVCMNuwvDMf4FTym?usp=sharing)
   
   Ponadto pobierz stamtąd dane data_docs i wstaw do głównego pliku projektu.
---

## 🛠️ 1. Instalacja uv

1. Zainstalować `uv` (jeśli nie jest zainstalowane):  
   `curl -LsSf https://astral.sh/uv/install.sh | sh`  

   Patrz: [Instrukcja instalacji uv](https://docs.astral.sh/uv/getting-started/installation/#installation-methods)

2. Sprawdzić działanie `uv`:  
   `uv --version`

---

## 📦 2. Pobranie i przygotowanie projektu

1. Pobierz repozytorium (git clone).
2. Przejdź do katalogu projektu (tam, gdzie znajduje się plik `pyproject.toml`):  
   `cd /sciezka/do/projektu`
3. Zsynchronizować środowisko i zainstalować zależności:  
   `uv sync`
4. Aktywuj środowisko:  
   `source .venv/bin/activate`

---

## 🔑 3. Konfiguracja API

1. Wygeneruj własny klucz API u dostawcy modelu (np. Gemini) – darmowa wersja.  

   [Generate_API_KEY](https://ai.google.dev/gemini-api/docs/pricing?hl=en)  
   Wybierz: **Get started for free**

2. Ustaw zmienną środowiskową z kluczem API (przykład):  

   `export OPENAI_API_KEY="TWOJ_KLUCZ_API"`

---

## ▶️ 4. Uruchomienie

1. Uruchom projekt:  
   `python RAG.py`
2. Po wykonaniu tych kroków system RAG ruszy.  
   Zostanie zbudowana baza wektorowa na podstawie jednej bajki wygenerowanej przez ChatGPT.  
   Bajka jest dobrym narzędziem do oceny systemu RAG (szczególnie demo), bo jest małe prawdopodobieństwo, że jest ona gdzieś dostępna w internecie, inaczej niż w przypadku pytań o wiedzę specjalistyczną, np. czym jest sieć neuronowa.

---

## 🧪 5. Testowanie – pytania do RAG

Możesz przetestować RAG na podstawie pytań:

1. **Jak miała na imię dziewczynka mieszkająca w krzywym domu?**  
   Odp. Hania.

2. **Kim był dziadek Hani z zawodu?**  
   Odp. Stolarzem.

3. **Dlaczego dach w pokoju Hani przeciekał?**  
   Odp. Bo dom był stary i tęsknił za lasem – kiedy „płakał”, z sufitu kapała woda.

4. **Jakie drzewko Hania posadziła przed domem?**  
   Odp. Lipę.

5. **Gdzie Hania kupiła młode drzewko?**  
   Odp. Na rynku, na stoisku starszej kobiety sprzedającej rośliny.

6. **Dlaczego dom czasem płakał, czyli kapała z niego woda?**  
   Odp. Bo tęsknił za lasem i za innymi drzewami, z których kiedyś powstał.

7. **Co sprawiło, że dom przestał kapać w pokoju Hani?**  
   Odp. Dotrzymał obietnicy po posadzeniu lipy – przestał „płakać”, a krople omijały dziurawe dachówki.

8. **Jak zareagował dziadek na pomysł posadzenia lipy?**  
   Odp. Na początku miał wątpliwości, ale zgodził się i pomógł ją posadzić.

9. **W jaki sposób lipa pomagała domowi czuć się mniej samotnym?**  
   Odp. Lipa „opowiadała” mu o deszczu, chmurach i gwiazdach – dom znów czuł się jak wśród drzew.

10. **Jak miała na imię kobieta sprzedająca drzewka na rynku?**  
    Odp. W bajce nie podano jej imienia.
