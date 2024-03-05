import sys
from src.logger import logger

def error_message_detail(error) -> str:
    _, _, exc_tb = sys.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_msg = f"Error occurred in python file: {file_name} at line: {exc_tb.tb_lineno} with error: {error}"
    return error_msg

class ExceptionHandler(Exception):
    def __init__(self, error_message) -> None:
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message)

    def __str__(self) -> str:
        return super().__str__() + f" with error message: {self.error_message}"

if __name__ == "__main__":
    try:
        # raise Exception("This is a test exception")
        a = 1 / 0
    except Exception as e:
        error_message = str(ExceptionHandler(str(e)))
        logger.error(error_message)
        raise ExceptionHandler(e)
