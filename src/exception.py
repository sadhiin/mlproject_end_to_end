import sys

def error_message_detail(error, error_detail: sys) -> str:
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_msg = "Error occured in python file: " + file_name + \
        " at line: " + str(exc_tb.tb_lineno) + " with error: " + str(error)
    return error_msg


class ExceptionHandler(Exception):
    def __init__(self, error_message, error_detail: sys) -> None:
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self) -> str:
        return super().__str__() + " with error message: " + self.error_message


if __name__ == "__main__":
    try:
        raise Exception("This is a test exception")
    except Exception as e:
        raise ExceptionHandler(str(e), sys) from e