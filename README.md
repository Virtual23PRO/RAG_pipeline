#  RAG_pipeline

Aby odpalić projekt należy wykonać poniższe kroki (Linux).

---

##  0. Wprowadzenie

1. Zapoznaj się z prezentacją (RAGdocs), w której omówiono zaimplementowaną architekturę RAG.  
   Z prezentacji dowiesz się również, na jakim etapie jest projekt tworzenia pipelinu:  
   [RAGdocs](https://drive.google.com/drive/folders/1mkRFEg8djZen7azcsVCMNuwvDMf4FTym?usp=sharing)
   
   Ponadto pobierz stamtąd dane **data_docs** i wstaw do głównego pliku projektu.

---

##  1. Instalacja uv

1. Zainstalować `uv` (jeśli nie jest zainstalowane):  
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

   Patrz: [Instrukcja instalacji uv](https://docs.astral.sh/uv/getting-started/installation/#installation-methods)

2. Sprawdzić działanie `uv`:  
   ```bash
   uv --version
   ```

---

##  2. Pobranie i przygotowanie projektu

1. Pobierz repozytorium (git clone).
2. Przejdź do katalogu projektu (tam, gdzie znajduje się plik `pyproject.toml`):  
   ```bash
   cd /sciezka/do/projektu
   ```
3. Zsynchronizować środowisko i zainstalować zależności:  
   ```bash
   uv sync
   ```
4. Aktywuj środowisko:  
   ```bash
   source .venv/bin/activate
   ```

---

##  3. Konfiguracja API

1. Wygeneruj własny klucz API u dostawcy modelu (np. Gemini) – darmowa wersja.  

   [Generate_API_KEY](https://ai.google.dev/gemini-api/docs/pricing?hl=en)  
   Wybierz: **Get started for free**

2. Ustaw zmienną środowiskową z kluczem API (przykład):  

   ```bash
   export GEMINI_API_KEY="TWOJ_KLUCZ_API"
   ```

---

##  4. Milvus – lokalna baza wektorowa

Do przechowywania wektorów używamy **Milvusa** uruchomionego w trybie standalone w kontenerze Dockera.

### 4.1. Wymagania

- Zainstalowany **Docker** oraz **docker compose**  
  (np. pakiet `docker-ce`; dla nowszych wersji komenda to `docker compose`, nie `docker-compose`).

Sprawdź, czy Docker działa:

```bash
docker --version
docker compose version   
```

### 4.2. Utworzenie katalogu na konfigurację Milvusa

W dowolnym miejscu (np. w katalogu domowym) utwórz katalog na pliki Milvusa:

```bash
mkdir milvus_db
cd milvus_db
```

### 4.3. Pobranie pliku docker-compose

Pobierz przygotowany plik `docker-compose.yml` dla Milvus Standalone:

```bash
wget https://github.com/milvus-io/milvus/releases/download/v2.4.13-hotfix/milvus-standalone-docker-compose.yml -O docker-compose.yml
```

Po tym kroku w katalogu `milvus_db` powinien znajdować się plik `docker-compose.yml`.

### 4.4. Uruchomienie Milvusa

Wciąż będąc w katalogu `milvus_db`, uruchom kontenery:

```bash
docker compose up -d
```

Docker pobierze odpowiednie obrazy i wystartuje Milvusa w tle.

Sprawdź, czy kontenery działają:

```bash
docker ps
```

Powinieneś zobaczyć kontenery o nazwach zbliżonych do:

- `milvus-standalone`
- `etcd`
- `minio`

Domyślnie Milvus nasłuchuje na porcie **19530** (gRPC) i **9091** (HTTP).

> Milvus musi działać w tle **przed uruchomieniem skryptu RAG**, inaczej aplikacja nie będzie miała dokąd zapisywać wektorów.

Aby zatrzymać Milvusa:

```bash
docker compose down
```

---

##  5. Uruchomienie

1. Upewnij się, że:
   - środowisko `.venv` jest aktywne,
   - kontenery Milvusa działają (`docker ps`),
   - zmienna `OPENAI_API_KEY` jest ustawiona.

2. Uruchom projekt:  
   ```bash
   python RAG.py
   ```

3. Po wykonaniu tych kroków system RAG ruszy.  
   Zostanie zbudowana baza wektorowa na podstawie jednej bajki wygenerowanej przez ChatGPT.  
   Bajka jest dobrym narzędziem do oceny systemu RAG (szczególnie demo), bo jest małe prawdopodobieństwo, że jest ona gdzieś dostępna w internecie, inaczej niż w przypadku pytań o wiedzę specjalistyczną, np. czym jest sieć neuronowa.

---

##  6. Testowanie – pytania do RAG

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
