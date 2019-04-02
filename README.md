**Download the whole stuff from git-hub:**
- git https://github.com/tomerm/MLClassificationDocker

**Build docker:**
- sudo docker build -t gunicorn-ml-image  ./

**Run it:**
- sudo docker run -d --name ML-SERVICE -p 80:80 gunicorn-ml-image

**Test it (from tests directory invoke):**

- curl -X POST -k "http://localhost:80/api/v1/analyze" -H "Content-Type:application/json" -d @ar.json
- curl -X GET -k "http://localhost:80/api/v1/healthcheck" -H "Content-Type:application/json"

**Resources**

All resources, needed for classification documents on the base of the pipe, created in the training and testing phase,
should be placed into the folder /app/resources. 

Resources collected by the process _Collector_, configured with the option `saveResources=yes`.
