# SMG Jobs
A list of SMG-specific jobs using Heritage Connector which will form the basis of the API.

Run `python run.py` to get a list of jobs. Need to run from inside the smg_jobs folder for the path to the *heritageconnector* module to correctly resolve.

## List of jobs
The name of each can be used as an argument to run the job, e.g. `python run.py lookup`.

- **lookup:** perform lookup on all free text & URL fields in people data and save the resulting dataframe to *GITIGNORE_DATA/lookup_result.pkl*.