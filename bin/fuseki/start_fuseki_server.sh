# ARG 1 is the path to the database created using ./load_data.sh
fuseki-server -v --debug --loc=$1 /heritage-connector