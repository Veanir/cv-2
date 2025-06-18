#!/bin/bash
# Skrypt do łatwego uruchamiania eksperymentów w Docker

set -e

# Kolory dla outputu
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🐳 Vision Transformer vs CNN - Docker Runner${NC}"
echo -e "${BLUE}================================================${NC}"

# Funkcje pomocnicze
build_image() {
    echo -e "${YELLOW}📦 Budowanie obrazu Docker...${NC}"
    docker-compose build
    echo -e "${GREEN}✅ Obraz zbudowany pomyślnie!${NC}"
}

run_tests() {
    echo -e "${YELLOW}🧪 Uruchamianie testów...${NC}"
    docker-compose run --rm test
}

run_experiment() {
    local model_type=${1:-cnn}
    local model_name=${2:-resnet18}
    local fraction=${3:-0.1}
    
    echo -e "${YELLOW}🔬 Uruchamianie eksperymentu: ${model_type} ${model_name} (${fraction} danych)${NC}"
    docker-compose run --rm experiment python main.py \
        --mode single \
        --model_type $model_type \
        --model_name $model_name \
        --fraction $fraction
}

run_comparison() {
    echo -e "${YELLOW}📊 Uruchamianie pełnego porównania (może trwać długo!)${NC}"
    echo -e "${RED}⚠️  To może zająć kilka godzin w zależności od sprzętu${NC}"
    read -p "Czy kontynuować? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker-compose run --rm comparison
    else
        echo -e "${YELLOW}Anulowano.${NC}"
    fi
}

interactive_shell() {
    echo -e "${YELLOW}💻 Uruchamianie interaktywnej sesji...${NC}"
    docker-compose run --rm -it vit-cnn-research bash
}

# Menu główne
show_menu() {
    echo -e "\n${BLUE}Wybierz opcję:${NC}"
    echo "1) 📦 Zbuduj obraz Docker"
    echo "2) 🧪 Uruchom testy"
    echo "3) 🔬 Eksperyment CNN (ResNet18, 10% danych)"
    echo "4) 🔬 Eksperyment ViT (10% danych)"
    echo "5) 🔬 Własny eksperyment"
    echo "6) 📊 Pełne porównanie (DŁUGIE!)"
    echo "7) 💻 Terminal interaktywny"
    echo "8) 🧹 Wyczyść kontenery"
    echo "9) ❌ Wyjście"
    echo
}

# Main logic
case ${1:-menu} in
    "build")
        build_image
        ;;
    "test")
        run_tests
        ;;
    "cnn-quick")
        run_experiment "cnn" "resnet18" "0.1"
        ;;
    "vit-quick")
        run_experiment "vit" "google/vit-base-patch16-224" "0.1"
        ;;
    "comparison")
        run_comparison
        ;;
    "shell")
        interactive_shell
        ;;
    "clean")
        echo -e "${YELLOW}🧹 Czyszczenie kontenerów...${NC}"
        docker-compose down
        docker system prune -f
        echo -e "${GREEN}✅ Wyczyszczono!${NC}"
        ;;
    "menu"|*)
        while true; do
            show_menu
            read -p "Wybierz opcję (1-9): " choice
            case $choice in
                1)
                    build_image
                    ;;
                2)
                    run_tests
                    ;;
                3)
                    run_experiment "cnn" "resnet18" "0.1"
                    ;;
                4)
                    run_experiment "vit" "google/vit-base-patch16-224" "0.1"
                    ;;
                5)
                    echo -e "${BLUE}Własny eksperyment:${NC}"
                    read -p "Model type (cnn/vit): " model_type
                    read -p "Model name: " model_name
                    read -p "Data fraction (0.1-1.0): " fraction
                    run_experiment "$model_type" "$model_name" "$fraction"
                    ;;
                6)
                    run_comparison
                    ;;
                7)
                    interactive_shell
                    ;;
                8)
                    echo -e "${YELLOW}🧹 Czyszczenie kontenerów...${NC}"
                    docker-compose down
                    docker system prune -f
                    echo -e "${GREEN}✅ Wyczyszczono!${NC}"
                    ;;
                9)
                    echo -e "${GREEN}👋 Do widzenia!${NC}"
                    exit 0
                    ;;
                *)
                    echo -e "${RED}❌ Nieprawidłowy wybór. Spróbuj ponownie.${NC}"
                    ;;
            esac
        done
        ;;
esac 