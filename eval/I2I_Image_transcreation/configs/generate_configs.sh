#!/bin/bash

# Define the countries
countries=("brazil" "india" "japan" "nigeria" "portugal" "turkey" "united-states")
country_names=("Brazil" "India" "Japan" "Nigeria" "Portugal" "Turkey" "the United States")

# Template for the config file
config_template="source_countries: SOURCE_COUNTRIES
source_data_path: ./data/part1
target_country: COUNTRY_NAME
prompt: make the image culturally relevant to COUNTRY_NAME
seed: 0
image_guidance: 1.5
num_inference_steps: 100
text_guidance: 7.5
debug: False"

# Create the output directory for configs
mkdir -p ./part1

# Iterate over the countries and create config files
for index in "${!countries[@]}"
do
    # get country and country name
    country=${countries[$index]}
    country_name=${country_names[$index]}
    # source countries is a list of all countries except the current country
    source_countries=(${countries[@]})
    unset source_countries[$index]

    # Create source countries string
    source_countries_str="["
    for source_country in "${source_countries[@]}"
    do
        source_countries_str="$source_countries_str'$source_country', "
    done
    source_countries_str="${source_countries_str::-2}]"

    echo "Source countries for $country_name: $source_countries_str"

    # Create the config file
    country_config="${config_template//COUNTRY_NAME/$country_name}"
    country_config="${country_config//SOURCE_COUNTRIES/$source_countries_str}"

    echo "Creating config file for $country_name"
    echo "$country_config"
    # Write the config to a file
    echo "$country_config" > "./part1/$country.yaml"
done

echo "Config files created for all countries."
