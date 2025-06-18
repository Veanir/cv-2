@echo off
REM Skrypt Windows do uruchamiania eksperymentów w Docker

echo.
echo ====================================================
echo      Vision Transformer vs CNN - Docker Runner
echo ====================================================
echo.

if "%1"=="build" goto build
if "%1"=="test" goto test
if "%1"=="cnn-quick" goto cnn_quick
if "%1"=="vit-quick" goto vit_quick
if "%1"=="comparison" goto comparison
if "%1"=="shell" goto shell
if "%1"=="clean" goto clean
goto menu

:build
echo 📦 Budowanie obrazu Docker...
docker-compose build
echo ✅ Obraz zbudowany pomyślnie!
goto end

:test
echo 🧪 Uruchamianie testów...
docker-compose run --rm test
goto end

:cnn_quick
echo 🔬 Uruchamianie eksperymentu CNN (ResNet18, 10%% danych)...
docker-compose run --rm experiment python main.py --mode single --model_type cnn --model_name resnet18 --fraction 0.1
goto end

:vit_quick
echo 🔬 Uruchamianie eksperymentu ViT (10%% danych)...
docker-compose run --rm experiment python main.py --mode single --model_type vit --model_name google/vit-base-patch16-224 --fraction 0.1
goto end

:comparison
echo 📊 Uruchamianie pełnego porównania (może trwać długo!)
echo ⚠️  To może zająć kilka godzin w zależności od sprzętu
set /p answer=Czy kontynuować? (y/N): 
if /I "%answer%"=="y" (
    docker-compose run --rm comparison
) else (
    echo Anulowano.
)
goto end

:shell
echo 💻 Uruchamianie interaktywnej sesji...
docker-compose run --rm -it vit-cnn-research bash
goto end

:clean
echo 🧹 Czyszczenie kontenerów...
docker-compose down
docker system prune -f
echo ✅ Wyczyszczono!
goto end

:menu
echo Dostępne opcje:
echo.
echo 1. 📦 Zbuduj obraz Docker
echo 2. 🧪 Uruchom testy  
echo 3. 🔬 Eksperyment CNN (ResNet18, 10%% danych)
echo 4. 🔬 Eksperyment ViT (10%% danych)
echo 5. 📊 Pełne porównanie (DŁUGIE!)
echo 6. 💻 Terminal interaktywny
echo 7. 🧹 Wyczyść kontenery
echo 8. ❌ Wyjście
echo.

set /p choice=Wybierz opcję (1-8): 

if "%choice%"=="1" goto build
if "%choice%"=="2" goto test
if "%choice%"=="3" goto cnn_quick
if "%choice%"=="4" goto vit_quick
if "%choice%"=="5" goto comparison
if "%choice%"=="6" goto shell
if "%choice%"=="7" goto clean
if "%choice%"=="8" goto exit

echo ❌ Nieprawidłowy wybór. Spróbuj ponownie.
echo.
goto menu

:end
echo.
echo Gotowe! Aby uruchomić ponownie, użyj: docker_run.bat
pause

:exit
echo.
echo �� Do widzenia!
pause 