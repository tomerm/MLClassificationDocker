from flask import Flask, request, jsonify
import json
import os.path
import jsonschema
import traceback

#References the flask app
app = Flask(__name__)

app.config['JSON_AS_ASCII'] = False

##Configuration
MAX_ATTEMPTS = 10
WAIT_TIME = 30
request_schema_fn=os.path.join(os.path.dirname(__file__),"resources", "json", "request.jsonschema")
assert os.path.isfile(request_schema_fn), "%s does not exist"%request_schema_fn
request_schema = json.load(open(request_schema_fn))
assert jsonschema.Draft4Validator.check_schema(request_schema) is None, "%s is not valid V1 schema"%request_schema_fn
payloadSize = 2056 #128 #KB
payloadUnit = "KB"


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
        # This can be replaced, later, for a json-schema validator. For now, simplicity is better.
        try:
            input = request.get_json(force=True)
            jsonschema.validate(input, request_schema)
            return (input, None)
        except Exception as e:
            # First, log the error.
            #app.logger.error("Invalid json input for request. Check result '%s'" % e.message)

            # Now, answer the request.
            m = "Invalid JSON input:" + request.get_json(force=True) # % e.message
            resp = app.make_response((json.dumps({"user_message": m}), 400,))
            resp.mimetype = 'application/json'
            return (None, resp)
    else:
        # First, log the error.
        m = "Unsupported media type: '%s'." % ctype
        resp = app.make_response((json.dumps({"user_message":m}),415,))
        resp.mimetype = 'application/json'
        return (None, resp)

@app.route("/api/v1/analyze", methods=["POST"])
def tokenize():
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
    # This could be an exception for a better design.
    (input, error_response) = extract_input(request)
    if error_response:
        return error_response

    sentences = input["text"]
    rid = input["id"] if "id" in input else None
    parameters = input["parameters"] if "parameters" in input else None
    language = input["language"]

    # Return now for empty requests.
    if len(sentences) == 0:
        res = []
        try:
            return assemble_response_v2(sentences, res, rid)
        except Exception as e:
            # First, log the error.
            trace = traceback.format_exc()
            exc_message = \
                "An unexpected error has occurred for request: '%s' and stack trace: %s" % (repr(e), trace)

            # Now, answer the request.
            m = {"user_message": "An unexpected error has occurred."}
            resp = app.make_response((json.dumps(m), 500,))
            resp.mimetype = 'application/json'
            return resp
    else:
        try:
            # Invoke the model summarization.
            res=[]
            for s in sentences:
                error = False #shensis error, r = predictor.predict(s,language)
                r = 'processed' #shensis temp 
                if not error:
                    res.append(r)
                else:
                    m = {"user_message": "Unsupport language %s." % language}
                    resp = app.make_response((json.dumps(m), 500,))
                    resp.mimetype = 'application/json'
                    return resp

        except Exception as e:
            # First, log the error.
            trace = traceback.format_exc()
            exc_message = \
                "An unexpected error has occurred for request: '%s' and stack trace: %s" % (repr(e), trace)
            #app.logger.error(exc_message)

            # Now, answer the request.
            m = {"user_message": "An unexpected error has occurred."}
            resp = app.make_response((json.dumps(m), 500,))
            resp.mimetype = 'application/json'
            return resp

        # Assemble the returned result.
        try:
            return assemble_response_v2(sentences, res, rid)
        except Exception as e:
            # First, log the error.
            trace = traceback.format_exc()
            exc_message = \
                "An unexpected error has occurred for request: '%s' and stack trace: %s" % (repr(e), trace)

            # Now, answer the request.
            m = {"user_message": "An unexpected error has occurred."}
            resp = app.make_response((json.dumps(m), 500,))
            resp.mimetype = 'application/json'
            return resp

def assemble_response_v2(sentences, results, rid):
    # Round up to 6th digit and just keep emotions around.
    # if "application/json" in accept:
    ret = {}
    if rid:
        ret["id"]=rid
    ret["sentences"] = [{"tokens": [ri for ri in r], "sentence": s} for s, r in zip(sentences, results)]
    # Return the message.
    return jsonify(ret)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
