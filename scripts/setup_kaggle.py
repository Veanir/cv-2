#!/usr/bin/env python3
"""
Skrypt pomocniczy do konfiguracji Kaggle API dla pobierania HAM10000
Używa pliku .env zamiast ~/.kaggle/kaggle.json
"""
import os
import json
from pathlib import Path

def setup_kaggle_env():
    """Konfiguracja Kaggle API przez plik .env"""
    print("🔧 Konfiguracja Kaggle API dla HAM10000 (plik .env)")
    print("=" * 50)
    
    project_root = Path.cwd()
    env_file = project_root / ".env"
    
    # Sprawdź czy już skonfigurowane
    if env_file.exists():
        with open(env_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        if "KAGGLE_USERNAME" in content and "KAGGLE_KEY" in content:
            print("✅ Kaggle API już skonfigurowane w .env!")
            print(f"📁 Plik: {env_file}")
            return True
        else:
            print("⚠️ Plik .env istnieje, ale brak konfiguracji Kaggle")
    
    print("📋 Instrukcje konfiguracji:")
    print()
    print("1. Utwórz konto na https://www.kaggle.com")
    print("2. Zaloguj się i idź do Account Settings")
    print("3. Przewiń do sekcji 'API'")
    print("4. Kliknij 'Create New API Token'")
    print("5. Pobierz plik kaggle.json")
    print()
    
    # Zapytaj użytkownika o dane API
    print("Podaj dane z pliku kaggle.json:")
    print()
    
    username = input("Kaggle username: ").strip()
    key = input("Kaggle API key: ").strip()
    
    if not username or not key:
        print("❌ Username i API key są wymagane!")
        return False
    
    # Przygotuj zawartość .env
    env_content = ""
    
    # Zachowaj istniejące linie jeśli plik istnieje
    if env_file.exists():
        with open(env_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Usuń stare linie Kaggle (jeśli istnieją)
        new_lines = []
        for line in lines:
            if not line.startswith(('KAGGLE_USERNAME=', 'KAGGLE_KEY=')):
                new_lines.append(line)
        
        env_content = "".join(new_lines)
        
        # Dodaj separator jeśli plik nie jest pusty
        if env_content and not env_content.endswith('\n'):
            env_content += '\n'
    
    # Dodaj nowe dane Kaggle
    env_content += f"\n# Kaggle API Credentials\n"
    env_content += f"KAGGLE_USERNAME={username}\n"
    env_content += f"KAGGLE_KEY={key}\n"
    
    # Zapisz .env
    with open(env_file, 'w', encoding='utf-8') as f:
        f.write(env_content)
    
    print("✅ Kaggle API skonfigurowane w .env!")
    print(f"📁 Plik: {env_file}")
    print("🔒 Plik .env jest już w .gitignore")
    
    return True

def test_kaggle_connection():
    """Testuje połączenie z Kaggle API używając .env"""
    print("\n🧪 Testowanie połączenia z Kaggle...")
    
    try:
        # Załaduj zmienne z .env
        from dotenv import load_dotenv
        load_dotenv()
        
        kaggle_username = os.getenv('KAGGLE_USERNAME')
        kaggle_key = os.getenv('KAGGLE_KEY')
        
        if not kaggle_username or not kaggle_key:
            print("❌ Brak KAGGLE_USERNAME lub KAGGLE_KEY w .env")
            return False
        
        print(f"✅ Znaleziono credentials w .env dla: {kaggle_username}")
        
        # Ustaw zmienne środowiskowe dla kaggle
        os.environ['KAGGLE_USERNAME'] = kaggle_username
        os.environ['KAGGLE_KEY'] = kaggle_key
        
        import kaggle
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        api = KaggleApi()
        api.authenticate()
        
        print("✅ Połączenie z Kaggle działa!")
        
        # Sprawdź czy HAM10000 jest dostępny
        ham_dataset = "kmader/skin-cancer-mnist-ham10000"
        try:
            dataset_info = api.dataset_view(ham_dataset)
            print(f"✅ Dataset HAM10000 znaleziony!")
            print(f"   Tytuł: {dataset_info.title}")
            print(f"   Rozmiar: {dataset_info.size} bajów")
            print(f"   Ostatnia aktualizacja: {dataset_info.lastUpdated}")
            return True
            
        except Exception as e:
            print(f"⚠️ Nie można załadować info o HAM10000: {e}")
            return False
            
    except ImportError as e:
        if "dotenv" in str(e):
            print("❌ Pakiet 'python-dotenv' nie jest zainstalowany")
            print("   Zainstaluj: pip install python-dotenv")
        elif "kaggle" in str(e):
            print("❌ Pakiet 'kaggle' nie jest zainstalowany")
            print("   Zainstaluj: pip install kaggle")
        else:
            print(f"❌ Błąd importu: {e}")
        return False
    except Exception as e:
        print(f"❌ Błąd połączenia z Kaggle: {e}")
        return False

def show_env_example():
    """Pokazuje przykład pliku .env"""
    print("\n📝 Przykład pliku .env:")
    print("=" * 30)
    print("# Kaggle API Credentials")
    print("KAGGLE_USERNAME=twoj_username")  
    print("KAGGLE_KEY=abc123def456ghi789...")
    print("=" * 30)
    print()

def main():
    """Główna funkcja"""
    print("🚀 Setup Kaggle API dla HAM10000 Dataset (.env)")
    print()
    
    show_env_example()
    
    # Konfiguruj API
    if setup_kaggle_env():
        # Testuj połączenie
        if test_kaggle_connection():
            print("\n🎯 Następne kroki:")
            print("1. Uruchom: python src/data/dataset_downloader.py")
            print("2. Lub użyj: from src.data.dataset_downloader import download_ham10000")
            print("3. Dataset zostanie pobrany do folderu 'data/'")
        else:
            print("\n⚠️ Konfiguracja zapisana, ale test połączenia nieudany")
            print("💡 Sprawdź czy klucze API są poprawne")
        
    else:
        print("\n❌ Konfiguracja nieudana!")

if __name__ == "__main__":
    main() 