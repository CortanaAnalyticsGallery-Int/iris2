from azure.ml.api.schema.schemaUtil import *
import azure.ml.api.schema.utilities as util
from azure.ml.api.exceptions.BadRequest import BadRequestException
from score import init as user_init_func
from score import run as user_run_func

def run(http_body):
    if aml_service_schema is not None:
        arguments = parse_service_input(http_body, aml_service_schema.input)
        try:
            return_obj = user_run_func(**arguments)
        except TypeError as exc:
            raise BadRequestException(str(exc))
    else:
        return_obj = user_run_func(http_body)
    return return_obj


def init():
    global aml_service_schema
    schema_file = "schema.json"
    if schema_file:
        aml_service_schema = load_service_schema(schema_file)
    else:
        aml_service_schema = None
    user_init_func()
