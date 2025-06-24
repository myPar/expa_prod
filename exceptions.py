from fastapi.responses import JSONResponse
from fastapi import status


class FatalServerException(Exception):
    def __init__(self, *args):
        super().__init__(*args)

    def get_json(self) -> JSONResponse:
        msg = str(self)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": {
                    "message": msg,
                    "type": "fatal error"
                }
            }
        )  

class FormatServerException(Exception):
    def __init__(self, *args):
        super().__init__(*args)

    def get_json(self) -> JSONResponse:
        msg = str(self)
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": {
                    "message": msg,
                    "type": "format error"
                }
            }
        )        