#!/bin/bash

# Script to run I2I transcreation for all or specific countries
# 
# Usage: 
#   ./run_all_countries.sh <model_name> [options]
#
# Options:
#   --countries "country1 country2"  Process only specific countries
#   --skip "country1 country2"       Skip specific countries
#   --continue-on-error              Continue processing even if one country fails
#
# Examples:
#   ./run_all_countries.sh instructpix2pix
#   ./run_all_countries.sh sdxl-instructpix2pix --countries "japan brazil"
#   ./run_all_countries.sh instructpix2pix --skip "united-states"
#   ./run_all_countries.sh instructpix2pix --continue-on-error

# Default values
MODEL=""
CONFIG_DIR="configs/part1"
SCRIPT="I2I_trancreation.py"
SPECIFIC_COUNTRIES=""
SKIP_COUNTRIES=""
CONTINUE_ON_ERROR=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --countries)
            SPECIFIC_COUNTRIES="$2"
            shift 2
            ;;
        --skip)
            SKIP_COUNTRIES="$2"
            shift 2
            ;;
        --continue-on-error)
            CONTINUE_ON_ERROR=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 <model_name> [options]"
            echo ""
            echo "Options:"
            echo "  --countries \"country1 country2\"  Process only specific countries"
            echo "  --skip \"country1 country2\"       Skip specific countries"
            echo "  --continue-on-error              Continue processing even if one fails"
            echo ""
            echo "Available models:"
            echo "  - instructpix2pix"
            echo "  - sdxl-instructpix2pix"
            echo "  - cosxl-edit"
            echo "  - magicbrush"
            echo ""
            echo "Examples:"
            echo "  $0 instructpix2pix"
            echo "  $0 sdxl-instructpix2pix --countries \"japan brazil\""
            echo "  $0 instructpix2pix --skip \"united-states\""
            echo "  $0 instructpix2pix --continue-on-error"
            exit 0
            ;;
        *)
            if [ -z "$MODEL" ]; then
                MODEL="$1"
            else
                echo "Error: Unknown argument '$1'"
                exit 1
            fi
            shift
            ;;
    esac
done

# Check if model name is provided
if [ -z "$MODEL" ]; then
    echo "Error: Model name is required"
    echo "Usage: $0 <model_name> [options]"
    echo "Run with --help for more information"
    exit 1
fi

# Check if config directory exists
if [ ! -d "$CONFIG_DIR" ]; then
    echo "Error: Config directory '$CONFIG_DIR' not found"
    exit 1
fi

# Check if the script exists
if [ ! -f "$SCRIPT" ]; then
    echo "Error: Script '$SCRIPT' not found"
    exit 1
fi

# Build list of configs to process
if [ -n "$SPECIFIC_COUNTRIES" ]; then
    # Process only specific countries
    CONFIG_FILES=""
    for country in $SPECIFIC_COUNTRIES; do
        config_file="$CONFIG_DIR/${country}.yaml"
        if [ -f "$config_file" ]; then
            CONFIG_FILES="$CONFIG_FILES $config_file"
        else
            echo "Warning: Config file not found for country '$country'"
        fi
    done
else
    # Get all config files
    CONFIG_FILES=$(ls $CONFIG_DIR/*.yaml 2>/dev/null)
fi

if [ -z "$CONFIG_FILES" ]; then
    echo "Error: No config files to process"
    exit 1
fi

# Filter out skipped countries
if [ -n "$SKIP_COUNTRIES" ]; then
    FILTERED_CONFIGS=""
    for config in $CONFIG_FILES; do
        country=$(basename "$config" .yaml)
        skip=false
        for skip_country in $SKIP_COUNTRIES; do
            if [ "$country" = "$skip_country" ]; then
                skip=true
                echo "Skipping: $country"
                break
            fi
        done
        if [ "$skip" = false ]; then
            FILTERED_CONFIGS="$FILTERED_CONFIGS $config"
        fi
    done
    CONFIG_FILES=$FILTERED_CONFIGS
fi

# Count total configs
TOTAL=$(echo "$CONFIG_FILES" | wc -w)
CURRENT=0
SUCCESS=0
FAILED=0

echo "=========================================="
echo "Running I2I Transcreation"
echo "=========================================="
echo "Model: $MODEL"
echo "Total countries: $TOTAL"
echo "Continue on error: $CONTINUE_ON_ERROR"
echo "Started at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
echo ""

# Arrays to track results
FAILED_COUNTRIES=()
SUCCESS_COUNTRIES=()

# Process each config file
for CONFIG in $CONFIG_FILES; do
    CURRENT=$((CURRENT + 1))
    COUNTRY=$(basename "$CONFIG" .yaml)
    
    echo "[$CURRENT/$TOTAL] Processing: $COUNTRY"
    echo "Config: $CONFIG"
    echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "----------------------------------------"
    
    # Run the script
    if python "$SCRIPT" --config "$CONFIG" --model "$MODEL"; then
        echo "✓ Success: $COUNTRY ($(date '+%H:%M:%S'))"
        SUCCESS=$((SUCCESS + 1))
        SUCCESS_COUNTRIES+=("$COUNTRY")
    else
        echo "✗ Failed: $COUNTRY ($(date '+%H:%M:%S'))"
        FAILED=$((FAILED + 1))
        FAILED_COUNTRIES+=("$COUNTRY")
        
        if [ "$CONTINUE_ON_ERROR" = false ]; then
            echo ""
            echo "Error: Processing failed for $COUNTRY"
            echo "Use --continue-on-error to continue despite failures"
            break
        fi
    fi
    
    echo ""
done

# Print summary
echo "=========================================="
echo "Summary"
echo "=========================================="
echo "Model: $MODEL"
echo "Total countries: $TOTAL"
echo "Successful: $SUCCESS"
echo "Failed: $FAILED"
echo ""

if [ $SUCCESS -gt 0 ]; then
    echo "Successful countries:"
    for country in "${SUCCESS_COUNTRIES[@]}"; do
        echo "  ✓ $country"
    done
    echo ""
fi

if [ $FAILED -gt 0 ]; then
    echo "Failed countries:"
    for country in "${FAILED_COUNTRIES[@]}"; do
        echo "  ✗ $country"
    done
    echo ""
fi

echo "Output directory: ./outputs/part1/$MODEL/"
echo "Completed at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

# Exit with error code if any failed
if [ $FAILED -gt 0 ]; then
    exit 1
else
    exit 0
fi
