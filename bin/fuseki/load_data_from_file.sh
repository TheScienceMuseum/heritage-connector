# ARG 1 is an empty directory that you want to store the persistent Fuseki database in
# ARG 2 is the path of an RDF file 

echo "Loading into TDB2 database: $1"
tdbloader2 --loc $1 $2