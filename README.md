**Download the whole stuff from git-hub:**
- git https://github.com/tomerm/MLClassificationDocker

**Build docker:**
- sudo docker build -t gunicorn-ml-image  ./

**Run it:**
- sudo docker run -d --name ML-SERVICE -p 80:80 -e GUNICORN_CONF="/app/custom_gunicorn_conf.py" gunicorn-ml-image

**Test it (from tests directory invoke):**

- curl -X POST -k "http://localhost:80/api/v1/analyze" -H "Content-Type:application/json"  -H "Expect:" -d @ar.json
- curl -X GET -k "http://localhost:80/api/v1/healthcheck" -H "Content-Type:application/json"

**Debugging**
Uncomment logging pertinent lines in 'main.py' and following lines in 'custom_gunicorn_conf.py'
#errorlog = "gunicorn_error.log"
#accesslog = "gunicorn_access.log"
#access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"

Send requests, then log into docker container by
- docker exec -it docker-container-id /bin/bash
then view content of /app/gunicorn_error.log, /app/gunicorn_access.log

Or alternatively, invoke in somewhere
- docker logs -f docker-container-id &> docker.log &
Send requests.
Observer content of 'docker.log' file under folder from where 'docker logs' command has been started

**Installation and configuring outside docker**
Tested on Ubuntu 14
- install all requirements mentioned in Dockerfile
- install 'gunicorn' and 'meinheld' wsgi servers:
pip install -U meinheld
pip install -U gunicorn
- from under 'app' folder run:
gunicorn -k egg:meinheld#gunicorn_worker -c custom_gunicorn_conf.py main:app

**Resources**

All resources, needed for classification documents on the base of the pipe, created in the training and testing phase,
should be placed into the folder **/app/resources**. Sample content of this folder is shown on the image below.

![image](https://user-images.githubusercontent.com/5329257/55479242-5406f980-5626-11e9-9df2-752dce940e94.png)

 Please observe that content of this folder should manually copied from training / testing environment upon completion of training / testing process. For more details please see: https://github.com/tomerm/MLClassification

Resources collected by the process _Collector_, configured with the option `saveResources=yes`.
