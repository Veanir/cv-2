import argparse
import base64
import os
import subprocess
import sys
from pathlib import Path
from typing import List

try:
    from dotenv import load_dotenv
except ImportError:
    print("Moduł python-dotenv nie jest zainstalowany.", file=sys.stderr)
    print("Proszę go zainstalować komendą: pip install -r requirements.txt", file=sys.stderr)
    sys.exit(1)

try:
    from vastai_sdk import VastAI
except ImportError:
    print("Moduł vastai-sdk nie jest zainstalowany.", file=sys.stderr)
    print("Proszę go zainstalować komendą: pip install -r requirements.txt", file=sys.stderr)
    sys.exit(1)


def run_remote_command(sdk: VastAI, instance_id: int, command: str, workdir: str = None):
    """
    Wykonuje komendę na zdalnej instancji Vast.ai i wyświetla jej output.
    """
    if workdir:
        command = f"cd {workdir} && {command}"
    
    print(f"-> Wykonywanie na instancji {instance_id}: '{command}'")
    
    try:
        output = sdk.execute(id=instance_id, command=command)
        if isinstance(output, list) and output:
            output_str = "".join(output)
            print("<- Output:")
            print(output_str)
        elif isinstance(output, str):
            print("<- Output:")
            print(output)
        
        # Prosta weryfikacja błędu
        if output and any(err in str(output).lower() for err in ["error", "fatal", "denied"]):
            print(f"!!! Ostrzeżenie: Wykryto potencjalny błąd w odpowiedzi. Sprawdź logi.", file=sys.stderr)
            return None

        return output

    except Exception as e:
        print(f"!!! Wystąpił błąd podczas wykonywania komendy: {e}", file=sys.stderr)
        return None


def setup_ssh_key(repo_url: str) -> Path:
    """
    Generuje klucz SSH lokalnie, jeśli nie istnieje, i instruuje użytkownika,
    aby dodał go do Deploy Keys na GitHubie. Zwraca ścieżkę do klucza prywatnego.
    """
    keys_dir = Path(".keys")
    keys_dir.mkdir(exist_ok=True)
    
    private_key_path = keys_dir / "vast_deploy_key"
    public_key_path = private_key_path.with_suffix(".pub")

    if not private_key_path.exists():
        print("--- KROK PRZYGOTOWAWCZY: Generowanie klucza SSH do wdrożeń ---")
        print(f"Nie znaleziono klucza w '{private_key_path}'. Generowanie nowego...")
        subprocess.run(
            [
                "ssh-keygen",
                "-t", "ed25519",
                "-f", str(private_key_path),
                "-N", "",  # Puste hasło
                "-C", "vast-ai-deploy-key"
            ],
            check=True,
            capture_output=True
        )
        print(f"Klucz SSH został wygenerowany i zapisany w katalogu '{keys_dir}'.")
    
    public_key = public_key_path.read_text().strip()

    print("\n" + "="*70)
    print("!!! AKCJA WYMAGANA (jednorazowo) !!!")
    print("Aby umożliwić instancji klonowanie repozytorium, dodaj poniższy")
    print("klucz publiczny jako 'Deploy Key' w ustawieniach Twojego repozytorium na GitHub.")
    print(f"\nLink do dodania klucza: {repo_url.replace('.git', '/settings/keys/new')}")
    print("\n--- Początek klucza publicznego ---")
    print(public_key)
    print("--- Koniec klucza publicznego ---\n")
    print("Porada: Nazwij klucz 'Vast.ai Deploy Key', aby łatwo go zidentyfikować w przyszłości.")
    print("Upewnij się, że NIE zaznaczasz opcji 'Allow write access', jeśli klucz ma służyć tylko do odczytu.")
    print("="*70 + "\n")
    
    input("Naciśnij Enter, gdy klucz zostanie dodany do GitHub...")
    
    return private_key_path


def main():
    """
    Główna funkcja skryptu do wdrażania i uruchamiania projektu na Vast.ai.
    """
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Skrypt do automatycznego wdrażania projektu CV na instancję Vast.ai.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("VAST_API_KEY"),
        help="Twój klucz API z panelu Vast.ai (można też ustawić w .env)."
    )
    parser.add_argument(
        "--instance-id",
        default=os.getenv("VAST_INSTANCE_ID"),
        help="ID Twojej instancji na Vast.ai (można też ustawić w .env)."
    )
    parser.add_argument(
        "--repo-url",
        default=os.getenv("GIT_REPO_URL"),
        help="URL do repozytorium Git w formacie SSH (można też ustawić w .env)."
    )
    parser.add_argument(
        "--run",
        choices=['test', 'comparison', 'experiment'],
        default=None,
        help="Określa, który serwis docker-compose uruchomić."
    )

    args = parser.parse_args()

    # Walidacja, czy zmienne są dostępne
    if not args.api_key:
        sys.exit("Błąd: Nie znaleziono klucza API. Ustaw VAST_API_KEY w pliku .env lub podaj go za pomocą --api-key.")
    if not args.instance_id:
        sys.exit("Błąd: Nie znaleziono ID instancji. Ustaw VAST_INSTANCE_ID w pliku .env lub podaj go za pomocą --instance-id.")
    if not args.repo_url:
        sys.exit("Błąd: Nie znaleziono URL repozytorium. Ustaw GIT_REPO_URL w pliku .env lub podaj go za pomocą --repo-url.")

    args.instance_id = int(args.instance_id)
    repo_name = args.repo_url.split('/')[-1].replace('.git', '')

    print("--- Rozpoczynanie wdrożenia na Vast.ai ---")

    # Krok 1: Przygotowanie klucza SSH
    private_key_path = setup_ssh_key(args.repo_url)
    private_key_content = private_key_path.read_text()

    # Krok 2: Konfiguracja instancji
    print("\n--- KROK 1: Inicjalizacja klienta i konfiguracja instancji ---")
    vast_sdk = VastAI(api_key=args.api_key)

    # Przygotowanie katalogu .ssh
    run_remote_command(vast_sdk, args.instance_id, "mkdir -p /root/.ssh && chmod 700 /root/.ssh")

    # Wgranie klucza prywatnego w bezpieczny sposób (przez base64)
    private_key_b64 = base64.b64encode(private_key_content.encode('utf-8')).decode('utf-8')
    key_upload_cmd = f"echo '{private_key_b64}' | base64 -d > /root/.ssh/id_ed25519"
    run_remote_command(vast_sdk, args.instance_id, key_upload_cmd)
    
    # Ustawienie prawidłowych uprawnień dla klucza
    run_remote_command(vast_sdk, args.instance_id, "chmod 600 /root/.ssh/id_ed25519")

    # Dodanie github.com do znanych hostów, aby uniknąć pytania o zaufanie
    run_remote_command(vast_sdk, args.instance_id, "ssh-keyscan github.com >> /root/.ssh/known_hosts")

    # Krok 3: Klonowanie repozytorium
    print(f"\n--- KROK 2: Klonowanie repozytorium '{repo_name}' ---")
    run_remote_command(vast_sdk, args.instance_id, f"rm -rf {repo_name} && git clone {args.repo_url}")
    
    # Krok 4: Budowanie kontenerów Docker
    print("\n--- KROK 3: Budowanie obrazów Docker ---")
    run_remote_command(vast_sdk, args.instance_id, "docker-compose build", workdir=repo_name)
    
    # Krok 5: Uruchomienie zadania
    if args.run:
        print(f"\n--- KROK 4: Uruchamianie zadania '{args.run}' ---")
        run_remote_command(vast_sdk, args.instance_id, f"docker-compose run --rm {args.run}", workdir=repo_name)
        print(f"\n--- Zadanie '{args.run}' zakończone. ---")
    else:
        print("\n--- KROK 4: Pominięto uruchomienie zadania ---")

    print("\n--- Wdrożenie zakończone pomyślnie! ---")


if __name__ == "__main__":
    main() 