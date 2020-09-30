# ARG 1 is an empty directory that you want to store the persistent Fuseki database in
# ARG 2 is the path of an RDF file 
echo "Exporting from Elasticsearch index to file: $2"
cd ../../smg_jobs
python es_to_csv.py nt $2

echo "Loading into TDB2 database: $1"
tdbloader2 --loc $1 $2