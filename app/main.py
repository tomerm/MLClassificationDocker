from flask import Flask, request, jsonify
import json
import os.path
import jsonschema
import traceback
import socket
from classification.predictor import Predictor

#References the flask app
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

##Configuration
request_schema_fn=os.path.join(os.path.dirname(__file__),"resources", "json", "request.jsonschema")
assert os.path.isfile(request_schema_fn), "%s does not exist"%request_schema_fn
request_schema = json.load(open(request_schema_fn))
assert jsonschema.Draft4Validator.check_schema(request_schema) is None, "%s is not valid V1 schema"%request_schema_fn
payloadSize = 10280 #2056 #KB
payloadUnit = "KB"

predictor = Predictor()

""" The Health Check API returns the current health state of the server. """
@app.route("/api/v1/healthcheck", methods=["GET"])
def healthcheck():
    resp = app.make_response((json.dumps({"status": "healthy"}), 200,))
    resp.mimetype = 'application/json'
    return resp

def handle_accept(request):
    if "Accept" in request.headers:
        accept = request.headers["Accept"]
        if not accept or "*/*" in accept or "application/json" in accept:
            accept = "application/json"
        else:
            m = "Invalid Accept header: '%s'. Pick 'application/json'." % accept
            resp = app.make_response((json.dumps({"user_message": m}), 406,))
            resp.mimetype = 'application/json'
            return (None, resp)
    else:
        accept = "application/json"
    return (accept, None)

def extract_input(request):
    '''
    Extract the text to analyze in whichever format it's delivered.
    '''
    # Read the content type or provide a default if empty.
    if "Content-type" in request.headers and request.headers["Content-type"]:
        ctype = request.headers["Content-type"]
    else:
        ctype = "application/json"

    # Process according to the correct content type.
    if "application/json" in ctype:
        try:
            input = request.get_json(force=True)
            jsonschema.validate(input, request_schema)
            return (input, None)
        except Exception as e:
            print("Invalid json input for request.") #, e.message)
            # Now, answer the request.
            m = "Invalid JSON input:" + request.get_json(force=True) # % e.message
            resp = app.make_response((json.dumps({"user_message": m}), 400,))
            resp.mimetype = 'application/json'
            return (None, resp)
    else:
        m = "Unsupported media type: '%s'." % ctype
        resp = app.make_response((json.dumps({"user_message":m}),415,))
        resp.mimetype = 'application/json'
        return (None, resp)

@app.route("/api/v1/analyze", methods=["POST"])
def analyze():
    cl = request.content_length
    if cl is not None and cl > payloadSize * 1024:
        # The content length is over the configured maximum size in kilobytes.
        err_message = "Payload size: %s, is greater than supported size of %s%s" % (cl, payloadSize, payloadUnit)
        resp = (json.dumps({"user_message": err_message}), 413, {"mimetype": 'application/json'})
        return app.make_response(resp)

    # Handle the accept header.
    (accept, error_response) = handle_accept(request)
    if error_response:
        return error_response

    # Extract the set of sentences, based on the appropriate content type.
    (input, error_response) = extract_input(request)
    if error_response:
        return error_response

    sentences = input["text"]
    rid = input["id"] if "id" in input else None
    parameters = input["parameters"] if "parameters" in input else None
    synchronous = input["synchronous"] if "synchronous" in input else None # if empty just respond, otherwise send responce to submitted email
    if len(sentences) == 0 or rid == None:
        m = {"user_message": "Either no request ID or empty text has been submitted for classification."}
        resp = app.make_response((json.dumps(m), 500,))
        resp.mimetype = 'application/json'
        return resp
    else:
        try:
            error = "OK"
            # Invoke classification 
            error, resp = predictor.predict(sentences)
            if error == "OK":
               ret = assemble_response(sentences, resp, rid)
               if synchronous == None or len(synchronous) == 0:
                  return ret
               else:
                  return send_responce_to_remote_server(parameters, sentences, resp, rid)
            else:
               m = {"user_message": "Following error occured: %s." % error}
               resp = app.make_response((json.dumps(m), 500,))
               resp.mimetype = 'application/json'
               return resp
        except Exception as e:
            m = {"user_message": "An unexpected error has occurred while classification. " + str(e)}
            resp = app.make_response((json.dumps(m), 500,))
            resp.mimetype = 'application/json'
            return resp

def send_responce_to_remote_server(parameters, sentences, resp, rid):
    targetServer = parameters['targetServer']
    targetPort = int(parameters['targetPort'])
    if targetServer == None or targetPort == None or len(targetServer) == 0:
      m = {"user_message": "Either 'tartget server URL' or 'target port' parameters are not filled."}
      resp = app.make_response((json.dumps(m), 500,))
      resp.mimetype = 'application/json'
      return resp
    try:
      res = 'Request Id: ' + rid + ' ,Sentences: ' + sentences + ' ,Detected classes: ' + resp	  
      s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
      s.connect((targetServer, targetPort))
      s.sendall(res.encode())
      s.close()	
	  
      m = {"user_message": "Responce has been sent to %s." % targetServer}
      resp = app.make_response((json.dumps(m), 500,))
      resp.mimetype = 'application/json'
      return resp
    except Exception as e:
      m = {"user_message": "Failed to send result to specified server. Exception occured: " + str(e)}
      resp = app.make_response((json.dumps(m), 500,))
      resp.mimetype = 'application/json'
      return resp
	
def assemble_response(sentences, results, rid):
    ret = {}
    if rid:
        ret["Request Id"] = rid
    ret["Sentences"] = sentences
    ret["Detected classes"] = results
    return jsonify(ret)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
