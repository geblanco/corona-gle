# Scraping DOIs elsevier
The script for scraping has been coded to write the results in a file, or in a database.

In both versions, the input file is declared in a variable called "file\_"

## File output
Writes the output in a file named [input_file]_res.csv.
Because this version is really slow, it's recomended to use the version that writes into a database

## Database output
This version is much faster.
In a first phase it populates the database with the DOIs and the sha values of the papers. The conditions to execute this phase are:
- the number of lines of the input file are different than the number of documents in the database
- it only inserts if the document it hasn't been registered in the database already (this can be improved to make it faster)

It uses a config file located in the root of the folder "scrapers".
